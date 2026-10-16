//! 27.2a：用 ORT（embed_core）的预处理产出 PNG，喂给 llama.cpp。
//!
//! 隔离"图片像素"与"token 预算"两个变量，判断图片不一致的根因。
//!
//! 用法：
//!   prepped_png --image_dir <dir> --files a.jpg,b.jpg --soft_tokens 140 --out_dir <dir>
//! 输出：每输入一张 `<stem>.ort.png`（ORT resized 实际尺寸）+ manifest.json。

use embed_core::image_prep::prepare_image_file;
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
        let inp = prepare_image_file(&image_dir.join(f), soft, false)?;
        let (th, tw) = inp.resized;
        // pixel_values 是「按补丁平铺」：patch(r,c) 在 idx=r*pw+c，每补丁内 16x16 像素行优先。
        // 必须按补丁坐标回填到整图，不能直接按数组顺序展平。
        let ps = 16usize;
        let ph = th as usize / ps;
        let pw = tw as usize / ps;
        let mut bytes = vec![0u8; th as usize * tw as usize * 3];
        for r in 0..ph {
            for c in 0..pw {
                let idx = r * pw + c;
                let src = &inp.pixel_values[idx * (ps * ps * 3)..(idx + 1) * (ps * ps * 3)];
                for py in 0..ps {
                    for px in 0..ps {
                        let s = (py * ps + px) * 3;
                        let di = (((r * ps + py) * tw as usize) + c * ps + px) * 3;
                        for k in 0..3 {
                            bytes[di + k] = (src[s + k] * 255.0).round().clamp(0.0, 255.0) as u8;
                        }
                    }
                }
            }
        }
        let img: image::RgbImage =
            image::ImageBuffer::from_raw(tw, th, bytes).expect("尺寸不匹配");
        let stem = PathBuf::from(f)
            .file_stem()
            .unwrap()
            .to_string_lossy()
            .into_owned();
        let out_name = format!("{stem}.ort.png");
        img.save(out_dir.join(&out_name))?;
        manifest.push(serde_json::json!({
            "file": f, "prepped": out_name,
            "resized_hw": [th, tw], "soft_tokens": inp.soft_tokens,
        }));
        println!("{f} -> {out_name} hw=[{th},{tw}] soft={}", inp.soft_tokens);
    }
    std::fs::write(
        out_dir.join("manifest.json"),
        serde_json::to_string_pretty(&manifest)?,
    )?;
    Ok(())
}
