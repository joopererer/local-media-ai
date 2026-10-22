// 内部诊断：打印 Rust DB 后处理的连通域框（含分数）。
use ocr_core::paddle::{prep_for_det, PaddleOcr, RgbImageData};
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let need = |n: &str| {
        let i = args.iter().position(|a| a == n).unwrap() + 1;
        args[i].clone()
    };
    let lib = PathBuf::from(need("--ort_lib"));
    let b = ort::init_from(&lib).map_err(|e| e.to_string())?;
    assert!(b.with_name("fo_ocr_dump").commit());
    let eng = PaddleOcr::load(&PathBuf::from(need("--model_dir")), 4, "dump")?;
    let img_path = args.last().unwrap();
    let img = image::open(img_path)?.to_rgb8();
    let (w, h) = img.dimensions();
    let data = prep_for_det(&RgbImageData {
        w,
        h,
        data: img.as_raw().to_vec(),
    });
    for (b, score) in eng.detect_debug(&data)? {
        println!(
            "{{\"bbox\":[{:.1},{:.1},{:.1},{:.1}],\"score\":{:.3}}}",
            b.x0, b.y0, b.x1, b.y1, score
        );
    }
    Ok(())
}
