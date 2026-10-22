//! 27.5: `llama-embed-worker` —— llama.cpp 嵌入工作进程（stdio 帧服务）。
//!
//! 用法：`llama-embed-worker --model <gguf> --mmproj <mmproj> [--gpu-layers 99]`
//! stdin 读 [`Request`] 帧，stdout 写响应帧，日志全部走 stderr（另可重定向到文件）。
//! 图片输入为主进程预处理后的 `SharedPixels`（h×w×3 RGB），不走 PNG 编解码。
//!
//! 嵌入实现见 `llama_impl`（构建时若找到 llama.cpp 导入库则链接；
//! 否则退化为骨架模式：帧协议可用、hello 报告 vulkan=false、嵌入返回错误）。

use embed_core::worker_frame::*;
use std::io::{BufReader, BufWriter, Write};

#[cfg(llama_linked)]
mod ffi;
#[cfg(llama_linked)]
mod llama_impl;

const BACKEND_TAG: &str = "b11669";

fn usage() -> ! {
    eprintln!("用法: llama-embed-worker --model <gguf> --mmproj <mmproj> [--gpu-layers 99]");
    std::process::exit(2);
}

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model = arg(&args, "--model").unwrap_or_else(|| usage());
    let mmproj = arg(&args, "--mmproj").unwrap_or_else(|| usage());
    let gpu_layers: i32 = arg(&args, "--gpu-layers").map_or(99, |s| s.parse().unwrap_or(99));

    eprintln!("[worker] backend_tag={BACKEND_TAG} gpu_layers={gpu_layers}");
    eprintln!("[worker] model={model} mmproj={mmproj}");

    // 加载模型：Vulkan 加载失败 → gpu_layers=0 重试（CPU 回退，方案 §8）。
    #[cfg(llama_linked)]
    let mut embedder = match llama_impl::LlamaEmbedder::load(&model, &mmproj, gpu_layers, 1120) {
        Ok(e) => Some(e),
        Err(e) if gpu_layers > 0 => {
            eprintln!("[worker] Vulkan 加载失败（{e}），回退 CPU");
            llama_impl::LlamaEmbedder::load(&model, &mmproj, 0, 1120)
                .map_err(|e2| {
                    eprintln!("[worker] CPU 加载也失败: {e2}");
                    std::process::exit(3);
                })
                .ok()
        }
        Err(e) => {
            eprintln!("[worker] 加载失败: {e}");
            std::process::exit(3);
        }
    };
    #[cfg(not(llama_linked))]
    let mut embedder: Option<()> = None;

    #[cfg(llama_linked)]
    let (vulkan, device) = {
        let e = embedder.as_ref().expect("loaded");
        (e.runtime.ends_with("vulkan"), e.runtime.clone())
    };
    #[cfg(not(llama_linked))]
    let (vulkan, device) = (false, "skeleton(no llama libs)".to_string());

    let hello_json = serde_json::json!({
        "backend_tag": BACKEND_TAG,
        "vulkan": vulkan,
        "device": device,
        "tokenizer_sha256": "n/a",
    })
    .to_string();

    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut rin = BufReader::new(stdin.lock());
    let mut wout = BufWriter::new(stdout.lock());

    loop {
        let (req_id, req) = match read_request(&mut rin, 64 * 1024 * 1024) {
            Ok(v) => v,
            Err(e) => {
                // stdin EOF（router 退出）→ 正常退出；其余为协议错误，记日志后退出。
                eprintln!("[worker] stdin 结束/错误 ({e})，退出");
                break;
            }
        };
        match req {
            Request::Hello => {
                if let Err(e) = write_hello_response(&mut wout, req_id, &hello_json) {
                    eprintln!("[worker] 写 hello 失败: {e}");
                    break;
                }
            }
            Request::Shutdown => {
                eprintln!("[worker] 收到 shutdown，退出");
                break;
            }
            Request::Cancel(id) => {
                // 本阶段单线程串行：cancel 仅记日志（§10-3 超时强杀由 router 执行）。
                eprintln!("[worker] cancel req={id}（串行执行中无法抢占）");
            }
            Request::Text(t) => {
                let sw = std::time::Instant::now();
                let (vec_res, ms) = embed_text(&mut embedder, &t, sw);
                match vec_res {
                    Ok(v) => {
                        if let Err(e) = write_response(&mut wout, req_id, Ok((&v, ms))) {
                            eprintln!("[worker] 写响应失败: {e}");
                            break;
                        }
                    }
                    Err(msg) => {
                        eprintln!("[worker] 文本嵌入失败: {msg}");
                        if write_response(&mut wout, req_id, Err(&msg)).is_err() {
                            break;
                        }
                    }
                }
            }
            Request::Image { h, w, soft, rgb } => {
                let sw = std::time::Instant::now();
                let n = rgb.len();
                eprintln!("[worker] image {h}x{w} soft={soft} rgb={n}B");
                let res = embed_image(&mut embedder, h, w, &rgb);
                let ms = sw.elapsed().as_secs_f64() * 1000.0;
                match res {
                    Ok(v) => {
                        if let Err(e) = write_response(&mut wout, req_id, Ok((&v, ms))) {
                            eprintln!("[worker] 写响应失败: {e}");
                            break;
                        }
                    }
                    Err(msg) => {
                        eprintln!("[worker] 图片嵌入失败: {msg}");
                        if write_response(&mut wout, req_id, Err(&msg)).is_err() {
                            break;
                        }
                    }
                }
            }
        }
    }
    let _ = std::io::stderr().flush();
}

#[cfg(llama_linked)]
fn embed_text(
    e: &mut Option<llama_impl::LlamaEmbedder>,
    text: &str,
    _sw: std::time::Instant,
) -> (Result<Vec<f32>, String>, f64) {
    let sw = std::time::Instant::now();
    match e.as_mut() {
        Some(x) => (x.embed_text(text), sw.elapsed().as_secs_f64() * 1000.0),
        None => (Err("worker 未加载模型（骨架模式）".to_string()), 0.0),
    }
}

#[cfg(not(llama_linked))]
fn embed_text(
    _e: &mut Option<()>,
    _text: &str,
    _sw: std::time::Instant,
) -> (Result<Vec<f32>, String>, f64) {
    (
        Err(
            "未链接 llama.cpp（骨架模式：请用 scripts/fetch_llama_runtime.ps1 生成导入库）"
                .to_string(),
        ),
        0.0,
    )
}

#[cfg(llama_linked)]
fn embed_image(
    e: &mut Option<llama_impl::LlamaEmbedder>,
    h: u32,
    w: u32,
    rgb: &[u8],
) -> Result<Vec<f32>, String> {
    match e.as_mut() {
        Some(x) => x.embed_image(h, w, rgb),
        None => Err("worker 未加载模型（骨架模式）".to_string()),
    }
}

#[cfg(not(llama_linked))]
fn embed_image(_e: &mut Option<()>, _h: u32, _w: u32, _rgb: &[u8]) -> Result<Vec<f32>, String> {
    Err("未链接 llama.cpp（骨架模式）".to_string())
}
