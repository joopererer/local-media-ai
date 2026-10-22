//! 27.5 §10-2：llama.cpp C ABI 最小绑定（b11669，EmbeddingGemma-2 嵌入路径专用）。
//!
//! 只声明本 worker 实际调用的函数（调用序列复刻 llama.cpp 官方示例（`gemma_probe.cpp`））。
//! 结构体按 `include/llama.h` / `tools/mtmd/mtmd.h` 的字段顺序手写。
//!
//! **布局校验**：`build.rs` 在有头文件时（`FO_LLAMA_SRC`）编译一个 C 断言程序，
//! 比对 `sizeof(llama_model_params/context_params/mtmd_context_params)` 与本文件的
//! `assert_layout!` 期望；不一致则构建失败（防止 pin 漂移导致 ABI 错位）。
//! 链接：build.rs 按 `FO_LLAMA_LIB_DIR`找 `llama.lib` /
//! `mtmd.lib` / `ggml.lib`（打包脚本生成，见 `scripts/fetch_llama_runtime.ps1`）。

#![allow(non_camel_case_types)]

use std::ffi::c_void;
use std::os::raw::{c_char, c_int};

pub type llama_model = c_void;
pub type llama_context = c_void;
pub type llama_batch_ext = c_void;
pub type llama_vocab = c_void;
pub type mtmd_context = c_void;
pub type mtmd_bitmap = c_void;
pub type mtmd_input_chunks = c_void;
pub type mtmd_input_chunk = c_void;

pub type llama_token = i32;
pub type llama_seq_id = i32;
pub type llama_pos = i32;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct llama_embd {
    pub data: *const f32,
    pub n_rows: usize,
    pub n_embd: usize,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct mtmd_input_text {
    pub text: *const c_char,
    pub length: usize,
    pub add_special: bool,
    pub parse_special: bool,
}

pub const LLAMA_POOLING_MEAN: c_int = 1; // LLAMA_POOLING_TYPE_MEAN（头：NONE=0,MEAN=1,CLS=2,LAST=3）
pub const LLAMA_PROCESS_ENCODE: c_int = 0;

#[link(name = "llama")]
#[link(name = "mtmd")]
#[link(name = "ggml")]
unsafe extern "C" {
    pub fn llama_backend_init();

    /// b11669 起 ggml 后端以动态模块分发（`ggml-cpu-*.dll` / `ggml-vulkan.dll`），
    /// 必须显式加载否则模型加载报 "no backends are loaded"。
    pub fn ggml_backend_load_all();
    /// 从指定目录加载全部后端（打包布局：`<app>/llama/`）。
    pub fn ggml_backend_load_all_from_path(dir_path: *const c_char);
    /// 已注册后端数量（日志用）。
    pub fn ggml_backend_reg_count() -> usize;
    pub fn llama_model_default_params() -> LlamaModelParams;
    pub fn llama_model_load_from_file(
        path: *const c_char,
        params: LlamaModelParams,
    ) -> *mut llama_model;
    pub fn llama_model_free(model: *mut llama_model);
    pub fn llama_context_default_params() -> LlamaContextParams;
    pub fn llama_init_from_model(
        model: *mut llama_model,
        params: LlamaContextParams,
    ) -> *mut llama_context;
    pub fn llama_free(ctx: *mut llama_context);
    pub fn llama_model_get_vocab(model: *const llama_model) -> *const llama_vocab;
    pub fn llama_model_n_embd_out(model: *const llama_model) -> c_int;
    pub fn llama_model_n_embd_inp(model: *const llama_model) -> c_int;
    pub fn llama_tokenize(
        vocab: *const llama_vocab,
        text: *const c_char,
        text_len: i32,
        tokens: *mut llama_token,
        n_tokens_max: i32,
        add_special: bool,
        parse_special: bool,
    ) -> c_int;
    pub fn llama_batch_ext_init(ctx: *mut llama_context) -> *mut llama_batch_ext;
    pub fn llama_batch_ext_free(batch: *mut llama_batch_ext);
    pub fn llama_batch_ext_clear(batch: *mut llama_batch_ext);
    pub fn llama_batch_ext_add_token(
        batch: *mut llama_batch_ext,
        seq_id: llama_seq_id,
        id: llama_token,
    ) -> i32;
    pub fn llama_batch_ext_add_embd(
        batch: *mut llama_batch_ext,
        seq_id: llama_seq_id,
        embd: llama_embd,
    ) -> i32;
    pub fn llama_batch_ext_set_pos(
        batch: *mut llama_batch_ext,
        idx: i32,
        pos: *const llama_pos,
    ) -> bool;
    pub fn llama_batch_ext_set_output_embd(
        batch: *mut llama_batch_ext,
        idx: i32,
        value: bool,
    ) -> bool;
    pub fn llama_process(ctx: *mut llama_context, ptype: c_int, batch: *mut llama_batch_ext) -> i32;
    pub fn llama_get_embeddings_seq(ctx: *mut llama_context, seq_id: llama_seq_id) -> *mut f32;
    pub fn mtmd_context_params_default() -> MtmdParams;
    pub fn mtmd_init_from_file(
        mmproj: *const c_char,
        model: *const llama_model,
        params: MtmdParams,
    ) -> *mut mtmd_context;
    pub fn mtmd_free(ctx: *mut mtmd_context);
    pub fn mtmd_bitmap_init(nx: u32, ny: u32, data: *const u8) -> *mut mtmd_bitmap;
    pub fn mtmd_bitmap_free(bitmap: *mut mtmd_bitmap);
    pub fn mtmd_default_marker() -> *const c_char;
    pub fn mtmd_input_chunks_init() -> *mut mtmd_input_chunks;
    pub fn mtmd_input_chunks_free(chunks: *mut mtmd_input_chunks);
    pub fn mtmd_input_chunks_size(chunks: *const mtmd_input_chunks) -> usize;
    pub fn mtmd_input_chunks_get(
        chunks: *const mtmd_input_chunks,
        idx: usize,
    ) -> *const mtmd_input_chunk;
    pub fn mtmd_tokenize(
        ctx: *mut mtmd_context,
        out: *mut mtmd_input_chunks,
        text: *const mtmd_input_text,
        bitmaps: *const *mut mtmd_bitmap,
        n_bitmaps: usize,
    ) -> c_int;
    pub fn mtmd_input_chunk_get_type(chunk: *const mtmd_input_chunk) -> c_int;
    pub fn mtmd_input_chunk_get_tokens_text(
        chunk: *const mtmd_input_chunk,
        n_out: *mut usize,
    ) -> *const llama_token;
    pub fn mtmd_input_chunk_get_n_tokens(chunk: *const mtmd_input_chunk) -> usize;
    pub fn mtmd_encode_chunk(ctx: *mut mtmd_context, chunk: *const mtmd_input_chunk) -> i32;
    pub fn mtmd_get_output_embd(ctx: *mut mtmd_context) -> *mut f32;
}

/// `llama_model_params`（b11669 头文件顺序；bool 组按头文件 6 个声明）。
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct LlamaModelParams {
    pub devices: *mut c_void,
    pub tensor_buft_overrides: *const c_void,
    pub n_gpu_layers: i32,
    pub split_mode: i32,
    pub load_mode: i32,
    pub lazy_mode: i32,
    pub main_gpu: i32,
    pub tensor_split: *const f32,
    pub progress_callback: *mut c_void,
    pub progress_callback_user_data: *mut c_void,
    pub kv_overrides: *const c_void,
    pub vocab_only: bool,
    pub check_tensors: bool,
    pub use_extra_bufts: bool,
    pub no_host: bool,
    pub no_alloc: bool,
    pub load_mtp: bool,
}

/// `llama_context_params`（b11669 头文件顺序；bool 组在末尾，见头文件注释）。
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct LlamaContextParams {
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_ubatch: u32,
    pub n_seq_max: u32,
    pub n_rs_seq: u32,
    pub n_outputs_max: u32,
    pub n_outputs_max_per_seq: u32,
    pub n_threads: i32,
    pub n_threads_batch: i32,
    pub ctx_type: i32,
    pub rope_scaling_type: i32,
    pub pooling_type: i32,
    pub attention_type: i32,
    pub flash_attn_type: i32,
    pub rope_freq_base: f32,
    pub rope_freq_scale: f32,
    pub yarn_ext_factor: f32,
    pub yarn_attn_factor: f32,
    pub yarn_beta_fast: f32,
    pub yarn_beta_slow: f32,
    pub yarn_orig_ctx: u32,
    pub defrag_thold: f32,
    pub cb_eval: *mut c_void,
    pub cb_eval_user_data: *mut c_void,
    pub type_k: i32,
    pub type_v: i32,
    pub moe_cache_size: usize,
    pub abort_callback: *mut c_void,
    pub abort_callback_data: *mut c_void,
    pub embeddings: bool,
    pub offload_kqv: bool,
    pub no_perf: bool,
    pub op_offload: bool,
    pub swa_full: bool,
    pub kv_unified: bool,
    pub samplers: *mut c_void,
    pub n_samplers: usize,
    pub ctx_other: *mut llama_context,
}

/// `mtmd_context_params`（b11669 头文件顺序）。
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MtmdParams {
    pub use_gpu: bool,
    pub device: *mut c_void,
    pub print_timings: bool,
    pub n_threads: c_int,
    pub image_marker: *const c_char,
    pub media_marker: *const c_char,
    pub flash_attn_type: c_int,
    pub warmup: bool,
    pub image_min_tokens: c_int,
    pub image_max_tokens: c_int,
    pub cb_eval: *mut c_void,
    pub cb_eval_user_data: *mut c_void,
    pub batch_max_tokens: i32,
    pub progress_callback: *mut c_void,
    pub progress_callback_user_data: *mut c_void,
}
