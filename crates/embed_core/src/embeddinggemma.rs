//! EmbeddingGemma 2 嵌入实现（模型专属部分；Phase 25.0.4 → 27.0 迁入 embed_core）。
//!
//! ORT 以 load-dynamic 方式按绝对路径加载独立的新版运行库，与 Dart 插件附带的
//! ORT 1.15.1 并存：新版库放在独立目录、使用不同文件名（见 [`ort_library_file_name`]）。
//! 与应用无关：只接收文件路径/文本，返回向量；图片预处理见 [`crate::image_prep`]。

use ort::session::Session;
use ort::value::Tensor;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use thiserror::Error;
use tokenizers::{Tokenizer, TruncationParams};

pub const EMBED_DIM: usize = 768;
/// 视觉/文本主干之间的特征维度（纯文本输入时传入空的 (0, 512) 特征）。
const FEATURE_DIM: usize = 512;
/// 防止异常长输入占满内存的安全上限（Python 参考实现不截断，属有意偏离，见 onnx_rust_image_prep.md）。
/// 通过分词器截断，保留结尾的特殊符号。
pub const MAX_TEXT_TOKENS: usize = 2048;

pub const TOKENIZER_FILE: &str = "tokenizer.json";
pub const TEXT_MODEL_FILE: &str = "onnx/model_quantized.onnx";
pub const VISION_MODEL_FILE: &str = "onnx/vision_encoder_quantized.onnx";

#[derive(Debug, Error)]
pub enum EmbedError {
    #[error("ORT 运行库不存在: {0}")]
    OrtLibraryMissing(PathBuf),
    #[error("ORT 运行库加载失败: {0}")]
    OrtLoad(String),
    #[error("模型文件不存在: {0}")]
    ModelMissing(PathBuf),
    #[error("分词器错误: {0}")]
    Tokenizer(String),
    #[error("推理错误: {0}")]
    Inference(String),
    #[error("图片错误: {0}")]
    InvalidImage(String),
    #[error("不支持的图片格式: {0}")]
    UnsupportedImage(String),
    #[error("图片过大或分配超限: {0}")]
    ImageTooLarge(String),
}

/// 新版 ORT 运行库在各平台上的文件名。刻意不用默认名
/// （onnxruntime.dll / libonnxruntime.so / libonnxruntime.dylib），
/// 避免 Dart 插件按名称加载时拾取到它。
pub fn ort_library_file_name() -> &'static str {
    #[cfg(target_os = "windows")]
    {
        "onnxruntime_fo.dll"
    }
    #[cfg(target_os = "macos")]
    {
        "libonnxruntime_fo.dylib"
    }
    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
    {
        "libonnxruntime_fo.so"
    }
}

static ORT_INIT: OnceLock<Result<PathBuf, String>> = OnceLock::new();

/// 按绝对路径加载 ORT 运行库。进程内只会初始化一次，之后的调用沿用首次的结果。
pub fn init_ort(lib_path: &Path) -> Result<(), EmbedError> {
    if !lib_path.is_absolute() {
        return Err(EmbedError::OrtLoad(format!(
            "必须是绝对路径: {}",
            lib_path.display()
        )));
    }
    if !lib_path.is_file() {
        return Err(EmbedError::OrtLibraryMissing(lib_path.to_path_buf()));
    }
    let res = ORT_INIT.get_or_init(|| {
        let builder = ort::init_from(lib_path).map_err(|e| e.to_string())?;
        if builder.with_name("local_media_ai").commit() {
            Ok(lib_path.to_path_buf())
        } else {
            Err("ORT 环境初始化失败".to_string())
        }
    });
    res.as_ref()
        .map(|_| ())
        .map_err(|e| EmbedError::OrtLoad(e.clone()))
}

fn inf<E: std::fmt::Display>(e: E) -> EmbedError {
    EmbedError::Inference(e.to_string())
}

/// 文本与图片共用的嵌入引擎。文本主干只加载一份（图片向量也经过它）；视觉编码器在首次嵌入图片时才加载。
pub struct Embedder {
    tokenizer: Tokenizer,
    session: Mutex<Session>,
    vision: Mutex<Option<Session>>,
    model_dir: PathBuf,
    threads: usize,
}

/// 图片嵌入结果与分段耗时（秒）。
#[derive(Debug, Clone)]
pub struct ImageEmbedding {
    pub vector: Vec<f32>,
    pub soft_tokens: usize,
    /// 读取解码 + 预处理（含 EXIF 方向、转 RGB、缩放、切补丁）
    pub prep_secs: f64,
    pub vision_secs: f64,
    pub backbone_secs: f64,
}

fn build_session(path: &Path, threads: usize) -> Result<Session, EmbedError> {
    Session::builder()
        .map_err(inf)?
        .with_intra_threads(threads.max(1))
        .map_err(inf)?
        .with_inter_threads(1)
        .map_err(inf)?
        .commit_from_file(path)
        .map_err(inf)
}

impl Embedder {
    /// `model_dir` 为 Hub 目录布局：`tokenizer.json` 与 `onnx/*_quantized.onnx(_data)`。
    /// 调用前须已 [`init_ort`]。`threads` 为 ORT 算子内线程数（后台任务默认取 4）。
    pub fn new(model_dir: &Path, threads: usize) -> Result<Self, EmbedError> {
        let tok_path = model_dir.join(TOKENIZER_FILE);
        let model_path = model_dir.join(TEXT_MODEL_FILE);
        for p in [&tok_path, &model_path] {
            if !p.is_file() {
                return Err(EmbedError::ModelMissing(p.clone()));
            }
        }
        let mut tokenizer =
            Tokenizer::from_file(&tok_path).map_err(|e| EmbedError::Tokenizer(e.to_string()))?;
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: MAX_TEXT_TOKENS,
                ..Default::default()
            }))
            .map_err(|e| EmbedError::Tokenizer(e.to_string()))?;
        let session = build_session(&model_path, threads)?;
        Ok(Self {
            tokenizer,
            session: Mutex::new(session),
            vision: Mutex::new(None),
            model_dir: model_dir.to_path_buf(),
            threads,
        })
    }

    /// 分词结果（含 BOS/EOS，已按 [`MAX_TEXT_TOKENS`] 截断）。
    pub fn token_ids(&self, text: &str, prefix: &str) -> Result<Vec<i64>, EmbedError> {
        let full = format!("{prefix}{text}");
        let enc = self
            .tokenizer
            .encode(full.as_str(), true)
            .map_err(|e| EmbedError::Tokenizer(e.to_string()))?;
        Ok(enc.get_ids().iter().map(|&i| i as i64).collect())
    }

    /// 返回已 L2 归一化的 768 维向量（池化与归一化在模型图内完成）。
    /// `prefix` 默认应为空（实测不加前缀更好，见 docs/spikes/embeddinggemma2.md）。
    pub fn embed(&self, text: &str, prefix: &str) -> Result<Vec<f32>, EmbedError> {
        let ids = self.token_ids(text, prefix)?;
        self.run_backbone(ids, Vec::new())
    }

    /// 文本主干；`image_features` 为 (n,512) 行优先展开，纯文本传空。
    fn run_backbone(
        &self,
        ids: Vec<i64>,
        image_features: Vec<f32>,
    ) -> Result<Vec<f32>, EmbedError> {
        let n = ids.len();
        let mask = vec![1i64; n];
        let rows = image_features.len() / FEATURE_DIM;
        let empty = || Tensor::from_array(([0usize, FEATURE_DIM], Vec::<f32>::new())).map_err(inf);
        let mut session = self
            .session
            .lock()
            .map_err(|_| EmbedError::Inference("session 锁已损坏".into()))?;
        let outputs = session
            .run(ort::inputs! {
                "input_ids" => Tensor::from_array(([1usize, n], ids)).map_err(inf)?,
                "attention_mask" => Tensor::from_array(([1usize, n], mask)).map_err(inf)?,
                "image_features" => Tensor::from_array(([rows, FEATURE_DIM], image_features)).map_err(inf)?,
                "video_features" => empty()?,
                "audio_features" => empty()?,
            })
            .map_err(inf)?;
        let (shape, data) = outputs["sentence_embedding"]
            .try_extract_tensor::<f32>()
            .map_err(inf)?;
        if data.len() != EMBED_DIM {
            return Err(EmbedError::Inference(format!("输出维度异常: {shape:?}")));
        }
        Ok(data.to_vec())
    }

    /// 读取图片文件并嵌入（EXIF 方向、RGB 转换规则见 [`crate::image_prep::load_rgb_image`]）。
    pub fn embed_image_file(
        &self,
        path: &Path,
        max_soft_tokens: u32,
        skip_sniff: bool,
    ) -> Result<ImageEmbedding, EmbedError> {
        let t = std::time::Instant::now();
        let inputs = crate::image_prep::prepare_image_file(path, max_soft_tokens, skip_sniff)?;
        let prep_secs = t.elapsed().as_secs_f64();
        let mut out = self.embed_prepared_image(&inputs)?;
        out.prep_secs = prep_secs;
        Ok(out)
    }

    /// 对已预处理的输入做嵌入：视觉编码器 → 文本主干。`prep_secs` 在此处为 0。
    pub fn embed_prepared_image(
        &self,
        inputs: &crate::image_prep::ImageInputs,
    ) -> Result<ImageEmbedding, EmbedError> {
        let (features, vision_secs): (Vec<f32>, f64) = {
            let mut guard = self
                .vision
                .lock()
                .map_err(|_| EmbedError::Inference("vision 锁已损坏".into()))?;
            if guard.is_none() {
                let p = self.model_dir.join(VISION_MODEL_FILE);
                if !p.is_file() {
                    return Err(EmbedError::ModelMissing(p));
                }
                *guard = Some(build_session(&p, self.threads)?);
            }
            let session = guard.as_mut().expect("已加载");
            // 视觉耗时只计推理，不含首次懒加载会话的时间
            let t_run = std::time::Instant::now();
            let outputs = session
                .run(ort::inputs! {
                    "pixel_values" => Tensor::from_array(([1usize, inputs.max_patches, crate::image_prep::PATCH_DIM], inputs.pixel_values.clone())).map_err(inf)?,
                    "pixel_position_ids" => Tensor::from_array(([1usize, inputs.max_patches, 2], inputs.position_ids.clone())).map_err(inf)?,
                })
                .map_err(inf)?;
            let (shape, data) = outputs["image_features"]
                .try_extract_tensor::<f32>()
                .map_err(inf)?;
            if data.len() != inputs.soft_tokens * FEATURE_DIM {
                return Err(EmbedError::Inference(format!(
                    "视觉特征形状异常: {shape:?}，期望 {} 行",
                    inputs.soft_tokens
                )));
            }
            (data.to_vec(), t_run.elapsed().as_secs_f64())
        };
        let t = std::time::Instant::now();
        let vector = self.run_backbone(inputs.input_ids.clone(), features)?;
        Ok(ImageEmbedding {
            vector,
            soft_tokens: inputs.soft_tokens,
            prep_secs: 0.0,
            vision_secs,
            backbone_secs: t.elapsed().as_secs_f64(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn library_name_differs_from_default() {
        let n = ort_library_file_name();
        assert!(n.contains("onnxruntime_fo"));
        assert_ne!(n, "onnxruntime.dll");
    }

    #[test]
    fn init_rejects_relative_and_missing_paths() {
        assert!(matches!(
            init_ort(Path::new("onnxruntime_fo.dll")),
            Err(EmbedError::OrtLoad(_))
        ));
        let missing = std::env::temp_dir()
            .join("fo_no_such_dir")
            .join(ort_library_file_name());
        assert!(matches!(
            init_ort(&missing),
            Err(EmbedError::OrtLibraryMissing(_))
        ));
    }

    #[test]
    fn missing_model_dir_is_reported() {
        let dir = std::env::temp_dir().join("fo_no_such_model_dir");
        assert!(matches!(
            Embedder::new(&dir, 4),
            Err(EmbedError::ModelMissing(_))
        ));
    }

    // 需要本机模型与 ORT 库：设置 FO_MODEL_DIR 与 FO_ORT_LIB 后用 `cargo test -- --ignored` 运行。
    #[test]
    #[ignore]
    fn embeds_text_with_real_model() {
        let lib = PathBuf::from(std::env::var("FO_ORT_LIB").expect("FO_ORT_LIB"));
        let dir = PathBuf::from(std::env::var("FO_MODEL_DIR").expect("FO_MODEL_DIR"));
        init_ort(&lib).unwrap();
        let e = Embedder::new(&dir, 4).unwrap();
        let a = e.embed("a red jacket", "").unwrap();
        let b = e.embed("a red jacket", "").unwrap();
        assert_eq!(a.len(), EMBED_DIM);
        assert_eq!(a, b);
        let norm: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-3, "norm={norm}");
    }
}
