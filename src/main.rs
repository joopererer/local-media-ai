//! 27.5: `fo-llama-worker` —— llama.cpp 嵌入工作进程（stdio 帧服务）。
//!
//! 用法：`fo-llama-worker --model <gguf> --mmproj <mmproj> [--gpu-layers 99]`
//! stdin 读 [`Request`] 帧，stdout 写响应帧，日志全部走 stderr（另可重定向到文件）。
//! 图片输入为主进程预处理后的 `SharedPixels`（h×w×3 RGB），不走 PNG 编解码。
//!
//! 注意：llama.cpp C ABI 绑定在实现 §10-2 时接入（见 `TODO_llama`）；本步先落
//! 帧循环骨架 + hello/错误路径（用占位向量证明协议），保证 router 可联调。

use embed_core::worker_frame::*;
use std::io::{BufReader, BufWriter, Read, StdinLock, StdoutLock, Write};

const BACKEND_TAG: &str = "b11669";

fn usage() -> ! {
    eprintln!("用法: fo-llama-worker --model <gguf> --mmproj <mmproj> [--gpu-layers 99]");
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

    // TODO_llama(§10-2): 在此加载 llama.cpp（Vulkan 探测 → gpu_layers / 0 回退），
    // 记录实际 vulkan/device/tokenizer_sha256；失败则 exit 3（router 视为不可用）。
    eprintln!("[worker] backend_tag={BACKEND_TAG} gpu_layers={gpu_layers}");
    eprintln!("[worker] model={model} mmproj={mmproj}");
    let vulkan = gpu_layers > 0;
    let hello_json =
        serde_json::json!({"backend_tag": BACKEND_TAG, "vulkan": vulkan, "device": "probe-pending",
            "tokenizer_sha256": "pending"})
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
                // TODO_llama(§10-2): 真实文本嵌入；当前返回占位零向量（仅协议联调）。
                eprintln!("[worker] text len={}（占位）", t.len());
                let v = vec![0f32; 768];
                if let Err(e) = write_response(&mut wout, req_id, Ok((&v, 0.0))) {
                    eprintln!("[worker] 写响应失败: {e}");
                    break;
                }
            }
            Request::Image { h, w, soft, rgb } => {
                // TODO_llama(§10-2): mtmd_bitmap_init(rgb) + 真实图片嵌入；当前占位。
                eprintln!("[worker] image {h}x{w} soft={soft} rgb={}B（占位）", rgb.len());
                let v = vec![0f32; 768];
                if let Err(e) = write_response(&mut wout, req_id, Ok((&v, 0.0))) {
                    eprintln!("[worker] 写响应失败: {e}");
                    break;
                }
            }
        }
    }
    let _ = std::io::stderr().flush();
}

#[allow(dead_code)]
fn _use_types(_r: StdinLock<'_>, _w: StdoutLock<'_>, _b: impl Read) {}
