//! 27.5 §10-2：llama.cpp EmbeddingGemma-2 嵌入实现（调用序复刻 27.2a gemma_probe.cpp）。
//!
//! 加载：`llama_backend_init` → 模型（`n_gpu_layers` Vulkan 探测值，失败退 0）
//! → context（embeddings + pooling MEAN, n_ctx/n_batch 2048, kv_unified）
//! → mtmd（mmproj, use_gpu, image_max_tokens）。
//! 编码：每输入一个独立 sequence，position 从 0 起，`llama_process(ENCODE)` 后
//! `llama_get_embeddings_seq(0)` 取 mean-pool 向量并 L2 归一化。
//! 图片走**像素**（`mtmd_bitmap_init(nx, ny, rgb)`），不经 PNG 编解码（方案 §1）。

use crate::ffi::*;
use std::ffi::CString;
use std::ptr;

pub struct LlamaEmbedder {
    model: *mut llama_model,
    ctx: *mut llama_context,
    mctx: *mut mtmd_context,
    batch: *mut llama_batch_ext,
    vocab: *const llama_vocab,
    n_embd_out: usize,
    n_embd_inp: usize,
    n_tokens: usize,
    pub runtime: String,
}

// 单线程使用（worker 内串行）；指针不跨线程共享。
unsafe impl Send for LlamaEmbedder {}

fn cstring(s: &str) -> Result<CString, String> {
    CString::new(s).map_err(|_| format!("路径含 NUL 字节: {s}"))
}

impl LlamaEmbedder {
    /// 加载模型与多模态上下文。`gpu_layers` 为 0 时纯 CPU（Vulkan 探测失败的回退）。
    pub fn load(
        model_path: &str,
        mmproj_path: &str,
        gpu_layers: i32,
        image_max_tokens: i32,
    ) -> Result<Self, String> {
        unsafe {
            llama_backend_init();
            let mp = llama_model_default_params();
            // 只改本 worker 关心的字段，其余保持 C 默认值。
            let mut mp = mp;
            mp.n_gpu_layers = gpu_layers;
            let c_model = cstring(model_path)?;
            let model = llama_model_load_from_file(c_model.as_ptr(), mp);
            if model.is_null() {
                return Err(format!("模型加载失败: {model_path}"));
            }

            let mut cp = llama_context_default_params();
            cp.n_ctx = 2048;
            cp.n_batch = 2048;
            cp.n_ubatch = 2048;
            cp.n_seq_max = 16;
            cp.kv_unified = true;
            cp.embeddings = true;
            cp.pooling_type = LLAMA_POOLING_MEAN;            let ctx = llama_init_from_model(model, cp);
            if ctx.is_null() {
                llama_model_free(model);
                return Err("context 初始化失败".to_string());
            }

            let mut mcp = mtmd_context_params_default();
            mcp.use_gpu = gpu_layers > 0;
            mcp.warmup = true;
            mcp.image_min_tokens = 1;
            mcp.image_max_tokens = image_max_tokens;
            let c_mmproj = cstring(mmproj_path)?;
            let mctx = mtmd_init_from_file(c_mmproj.as_ptr(), model, mcp);
            if mctx.is_null() {
                llama_free(ctx);
                llama_model_free(model);
                return Err(format!("mmproj 加载失败: {mmproj_path}"));
            }

            let batch = llama_batch_ext_init(ctx);
            if batch.is_null() {
                mtmd_free(mctx);
                llama_free(ctx);
                llama_model_free(model);
                return Err("batch 初始化失败".to_string());
            }
            let vocab = llama_model_get_vocab(model);
            let n_embd_out = llama_model_n_embd_out(model).max(0) as usize;
            let n_embd_inp = llama_model_n_embd_inp(model).max(0) as usize;
            let runtime = format!("llama-cpp-b11669-{}", if gpu_layers > 0 { "vulkan" } else { "cpu" });
            Ok(Self {
                model,
                ctx,
                mctx,
                batch,
                vocab,
                n_embd_out,
                n_embd_inp,
                n_tokens: 0,
                runtime,
            })
        }
    }

    fn reset(&mut self) {
        unsafe {
            llama_batch_ext_clear(self.batch);
        }
        self.n_tokens = 0;
    }

    /// 当前位置放置（position 从 0 起，与 probe 一致）。
    fn place(&mut self, idx: i32) -> bool {
        unsafe {
            let pos = self.n_tokens as llama_pos;
            if !llama_batch_ext_set_pos(self.batch, idx, &pos) {
                return false;
            }
            if !llama_batch_ext_set_output_embd(self.batch, idx, true) {
                return false;
            }
        }
        self.n_tokens += 1;
        true
    }

    fn add_token(&mut self, tok: llama_token) -> bool {
        unsafe {
            let idx = llama_batch_ext_add_token(self.batch, 0, tok);
            if idx < 0 {
                return false;
            }
            self.place(idx)
        }
    }

    fn add_rows(&mut self, rows: &[f32], count: usize) -> bool {
        for r in 0..count {
            let e = llama_embd {
                data: unsafe { rows.as_ptr().add(r * self.n_embd_inp) },
                n_rows: 1,
                n_embd: self.n_embd_inp,
            };
            unsafe {
                let idx = llama_batch_ext_add_embd(self.batch, 0, e);
                if idx < 0 {
                    return false;
                }
                if !self.place(idx) {
                    return false;
                }
            }
        }
        true
    }

    fn run_seq(&mut self) -> Vec<f32> {
        unsafe {
            if llama_process(self.ctx, LLAMA_PROCESS_ENCODE, self.batch) != 0 {
                return Vec::new();
            }
            let p = llama_get_embeddings_seq(self.ctx, 0);
            if p.is_null() {
                return Vec::new();
            }
            let mut v = std::slice::from_raw_parts(p, self.n_embd_out).to_vec();
            let s: f64 = v.iter().map(|x| f64::from(*x) * f64::from(*x)).sum();
            let k = if s > 0.0 { (1.0 / s.sqrt()) as f32 } else { 0.0 };
            for x in v.iter_mut() {
                *x *= k;
            }
            v
        }
    }

    pub fn embed_text(&mut self, text: &str) -> Result<Vec<f32>, String> {
        self.reset();
        let c = cstring(text)?;
        let mut buf: Vec<i32> = Vec::new();
        unsafe {
            let n = llama_tokenize(
                self.vocab,
                c.as_ptr(),
                text.len() as i32,
                ptr::null_mut(),
                0,
                true,
                true,
            );
            if n >= 0 {
                return Err("tokenize 未返回所需长度".to_string());
            }
            buf.resize((-n) as usize, 0);
            let n = llama_tokenize(
                self.vocab,
                c.as_ptr(),
                text.len() as i32,
                buf.as_mut_ptr(),
                buf.len() as i32,
                true,
                true,
            );
            if n < 0 {
                return Err(format!("tokenize 失败: {n}"));
            }
            buf.truncate(n as usize);
        }
        for t in &buf {
            if !self.add_token(*t) {
                return Err("添加 token 失败".to_string());
            }
        }
        if self.n_tokens == 0 {
            return Err("空输入".to_string());
        }
        Ok(self.run_seq())
    }

    /// 图片：`rgb` 为缩放后的行优先 h×w×3 像素（主进程 SharedPixels）。
    pub fn embed_image(&mut self, h: u32, w: u32, rgb: &[u8]) -> Result<Vec<f32>, String> {
        if rgb.len() != (h as usize) * (w as usize) * 3 {
            return Err("像素长度与尺寸不符".to_string());
        }
        unsafe {
            let bmp = mtmd_bitmap_init(w, h, rgb.as_ptr());
            if bmp.is_null() {
                return Err("bitmap 初始化失败".to_string());
            }
            let chunks = mtmd_input_chunks_init();
            let marker = mtmd_default_marker();
            let text = mtmd_input_text {
                text: marker,
                length: std::ffi::CStr::from_ptr(marker).to_bytes().len(),
                add_special: true,
                parse_special: true,
            };
            let bmps: [*mut mtmd_bitmap; 1] = [bmp];
            let st = mtmd_tokenize(self.mctx, chunks, &text, bmps.as_ptr(), 1);
            if st != 0 {
                mtmd_input_chunks_free(chunks);
                mtmd_bitmap_free(bmp);
                return Err(format!("mtmd_tokenize 失败: {st}"));
            }
            self.reset();
            let mut ok = true;
            let count = mtmd_input_chunks_size(chunks);
            for i in 0..count {
                let ch = mtmd_input_chunks_get(chunks, i);
                if ch.is_null() {
                    ok = false;
                    break;
                }
                let ct = mtmd_input_chunk_get_type(ch);
                if ct == 0 {
                    // 文本 chunk（marker 前后的特殊 token）
                    let mut n: usize = 0;
                    let toks = mtmd_input_chunk_get_tokens_text(ch, &mut n);
                    if toks.is_null() {
                        ok = false;
                        break;
                    }
                    for k in 0..n {
                        let t = *toks.add(k);
                        if !self.add_token(t) {
                            ok = false;
                            break;
                        }
                    }
                } else if ct == 1 {
                    // 图片 chunk：encoder rows → batch embd（复刻 probe）
                    if mtmd_encode_chunk(self.mctx, ch) != 0 {
                        ok = false;
                        break;
                    }
                    let rows = mtmd_get_output_embd(self.mctx);
                    if rows.is_null() {
                        ok = false;
                        break;
                    }
                    let n_rows = mtmd_input_chunk_get_n_tokens(ch);
                    let inp = self.n_embd_inp;
                    let slice = std::slice::from_raw_parts(rows, n_rows * inp);
                    if !self.add_rows(slice, n_rows) {
                        ok = false;
                        break;
                    }
                }
                if !ok {
                    break;
                }
            }
            mtmd_input_chunks_free(chunks);
            mtmd_bitmap_free(bmp);
            if !ok {
                return Err("图片 chunk 处理失败".to_string());
            }
            if self.n_tokens == 0 {
                return Err("空图片输入".to_string());
            }
        }
        Ok(self.run_seq())
    }
}

impl Drop for LlamaEmbedder {
    fn drop(&mut self) {
        unsafe {
            if !self.batch.is_null() {
                llama_batch_ext_free(self.batch);
            }
            if !self.mctx.is_null() {
                mtmd_free(self.mctx);
            }
            if !self.ctx.is_null() {
                llama_free(self.ctx);
            }
            if !self.model.is_null() {
                llama_model_free(self.model);
            }
        }
    }
}
