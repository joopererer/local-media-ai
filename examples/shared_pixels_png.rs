//! 27.2b：导出"共享预处理像素"（后端边界）为 PNG，供 llama.cpp 等后端做一致性/基准对比。
//!
//! 现在直接调用 [`embed_core::image_prep::shared_pixels_file`]（唯一的缩放实现），
//! 不再手工从 pixel_values 反推（27.2a 的临时做法）。
//!
//! 用法：
//!   shared_pixels_png --image_dir <dir> --files a.jpg,b.jpg --soft_tokens 140 --out_dir <dir>

use embed_core::image_prep::shared_pixels_file;
use std::path::PathBuf;

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let need = |n: &str| arg(&args, n).unwrap_or_else(|| panic!("缺少参数 {n}"));
    let soft: u32 = arg(&args, "--soft_tokens")
        .map_or(140, |s| s.parse().expect("--soft_tokens"));
    let image_dir = PathBuf::from(need("--image_dir"));
    let files: Vec<String> = need("--files").split(',').map(str::to_string).collect();
    let out_dir = PathBuf::from(need("--out_dir"));
    std::fs::create_dir_all(&out_dir)?;

    let mut manifest = Vec::new();
    for f in &files {
        let sp = shared_pixels_file(&image_dir.join(f), soft, false)?;
        let stem = PathBuf::from(f)
            .file_stem()
            .unwrap()
            .to_string_lossy()
            .into_owned();
        let out_name = format!("{stem}.png");
        std::fs::write(out_dir.join(&out_name), sp.to_png()?)?;
        let inputs = sp.patchify()?;
        manifest.push(serde_json::json!({
            "file": f, "prepped": out_name,
            "size_hw": [sp.size_hw.0, sp.size_hw.1],
            "soft_tokens": inputs.soft_tokens,
        }));
        println!(
            "{f} -> {out_name} hw=[{},{}] soft={}",
            sp.size_hw.0, sp.size_hw.1, inputs.soft_tokens
        );
    }
    std::fs::write(
        out_dir.join("manifest.json"),
        serde_json::to_string_pretty(&manifest)?,
    )?;
    Ok(())
}
