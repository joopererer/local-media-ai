// 27.3b：Rust OCR 评测（PP-OCRv5 mobile）。
//
// 用法：ocr_bench --ort_lib <dll> --model_dir <dir含det/cls/rec.onnx> <img1> <img2> ...
// 输出每张图的 lines + bbox + 各阶段耗时 JSON（供 Python 对比与耗时统计）。
use ocr_core::PaddleEngine;
use std::path::{Path, PathBuf};

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let need = |n: &str| arg(&args, n).unwrap_or_else(|| panic!("缺少 {n}"));
    let threads: usize = arg(&args, "--threads").map_or(4, |s| s.parse().unwrap());
    let text_score: f32 = arg(&args, "--text_score").map_or(0.5, |s| s.parse().unwrap());

    embed_like_init(&need("--ort_lib"))?;
    let eng = PaddleEngine::load(&PathBuf::from(need("--model_dir")), threads)?;

    let imgs: Vec<String> = args
        .iter()
        .skip(1)
        .filter(|a| !a.starts_with("--"))
        .filter(|a| a.ends_with(".png") || a.ends_with(".jpg") || a.ends_with(".jpeg"))
        .cloned()
        .collect();
    assert!(!imgs.is_empty(), "请给出图片路径");
    for img in imgs {
        let t = std::time::Instant::now();
        let (res, det_ms, cls_ms, rec_ms) = eng.recognize_file_with_timing(Path::new(&img))?;
        let total_ms = t.elapsed().as_secs_f64() * 1000.0;
        let kept: Vec<serde_json::Value> = res
            .lines
            .iter()
            .filter(|l| l.confidence >= text_score)
            .map(|l| {
                serde_json::json!({
                    "text": l.text, "conf": (l.confidence * 1000.0).round() / 1000.0,
                    "bbox": l.bbox,
                })
            })
            .collect();
        println!(
            "{}",
            serde_json::json!({
                "file": img, "engine": res.engine,
                "n_empty_early_exit": res.lines.is_empty(),
                "n_lines": kept.len(),
                "det_ms": (det_ms * 10.0).round() / 10.0,
                "cls_ms": (cls_ms * 10.0).round() / 10.0,
                "rec_ms": (rec_ms * 10.0).round() / 10.0,
                "total_ms": (total_ms * 10.0).round() / 10.0,
                "lines": kept,
            })
        );
    }
    Ok(())
}

fn embed_like_init(ort_lib: &str) -> Result<(), Box<dyn std::error::Error>> {
    // 与 embed_core 共用 ORT 初始化（同进程单例）：用环境同名路径加载。
    let lib = PathBuf::from(ort_lib);
    // ocr_core 不依赖 embed_core；此处直接调 ort 的动态加载。
    let b = ort::init_from(&lib).map_err(|e| e.to_string())?;
    if b.with_name("fo_ocr").commit() {
        Ok(())
    } else {
        Err("ORT 初始化失败".into())
    }
}
