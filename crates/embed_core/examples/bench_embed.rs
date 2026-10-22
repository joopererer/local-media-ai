//! 27.2b：ORT 嵌入基准（预热后的逐项耗时），供三后端对比。
//!
//! 用法：
//!   bench_embed --ort_lib <dll> --model_dir <dir> [--threads 4] [--soft_tokens 140]
//!               --image_dir <dir> --files a.jpg,b.jpg --texts "t1|t2" [--repeat 1]
//! 输出 JSON：{backend, threads, soft_tokens, load_s, images:{ms:[...]}, texts:{ms:[...]}}
//! 预热：先跑一张图/一条文本（不计时）。

use embed_core::{init_ort, Embedder};
use std::path::PathBuf;
use std::time::Instant;

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let need = |n: &str| arg(&args, n).unwrap_or_else(|| panic!("缺少参数 {n}"));
    let threads: usize = arg(&args, "--threads").map_or(4, |s| s.parse().unwrap());
    let soft: u32 = arg(&args, "--soft_tokens").map_or(140, |s| s.parse().unwrap());
    let repeat: usize = arg(&args, "--repeat").map_or(1, |s| s.parse().unwrap());
    let image_dir = PathBuf::from(need("--image_dir"));
    let files: Vec<String> = need("--files").split(',').map(str::to_string).collect();
    let texts: Vec<String> = need("--texts")
        .split('|')
        .map(str::to_string)
        .filter(|s| !s.is_empty())
        .collect();

    init_ort(&PathBuf::from(need("--ort_lib")))?;
    let t0 = Instant::now();
    let embedder = Embedder::new(&PathBuf::from(need("--model_dir")), threads)?;
    let load_s = t0.elapsed().as_secs_f64();

    // 预热：视觉会话懒加载 + 主干图缓存，均不计时。
    if let Some(f) = files.first() {
        let _ = embedder.embed_image_file(&image_dir.join(f), soft, false)?;
    }
    if let Some(t) = texts.first() {
        let _ = embedder.embed(t, "")?;
    }

    let mut image_ms = Vec::new();
    for _ in 0..repeat {
        for f in &files {
            let t = Instant::now();
            embedder.embed_image_file(&image_dir.join(f), soft, false)?;
            image_ms.push(t.elapsed().as_secs_f64() * 1000.0);
        }
    }
    let mut text_ms = Vec::new();
    for _ in 0..repeat {
        for t in &texts {
            let s = Instant::now();
            embedder.embed(t, "")?;
            text_ms.push(s.elapsed().as_secs_f64() * 1000.0);
        }
    }

    let out = serde_json::json!({
        "backend": "ort-cpu", "threads": threads, "soft_tokens": soft,
        "load_s": load_s,
        "images": { "n": image_ms.len(), "ms": image_ms },
        "texts": { "n": text_ms.len(), "ms": text_ms },
    });
    println!("{}", serde_json::to_string(&out)?);
    Ok(())
}
