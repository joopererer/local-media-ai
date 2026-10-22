//! 27.0：embed_core 独立性验证。
//!
//! 证明 `embed_core` 不依赖应用层：本测试只用文件路径/文本做输入，
//! 输出向量与分段耗时；FRB、`fo_core`、数据库概念均不出现。

use embed_core::{init_ort, Embedder};
use std::path::PathBuf;

fn ort_lib() -> PathBuf {
    PathBuf::from(std::env::var("FO_ORT_LIB").expect("FO_ORT_LIB"))
}

fn model_dir() -> PathBuf {
    PathBuf::from(std::env::var("FO_MODEL_DIR").expect("FO_MODEL_DIR"))
}

#[test]
#[ignore]
fn image_file_roundtrip_with_real_model() {
    init_ort(&ort_lib()).unwrap();
    let e = Embedder::new(&model_dir(), 4).unwrap();
    let img = PathBuf::from(std::env::var("FO_TEST_IMAGE").expect("FO_TEST_IMAGE"));
    let r = e.embed_image_file(&img, 140, false).unwrap();
    assert_eq!(r.vector.len(), embed_core::EMBED_DIM);
    // `max_soft_tokens` 是上限；实际值按长宽比计算（见 image_prep）。
    assert!(r.soft_tokens > 0 && r.soft_tokens <= 140, "n={}", r.soft_tokens);
    assert!(r.prep_secs > 0.0 && r.vision_secs >= 0.0 && r.backbone_secs > 0.0);
    let norm: f32 = r.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-3, "norm={norm}");
}
