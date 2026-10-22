//! 嵌入路由与工作进程监管使用的错误类型。
//!
//! 该类型原先定义在应用绑定层（File Organizer 的 `fo_core::api::embedding`，与 Dart 侧的
//! FRB 接口同层）。拆分为独立 crate 后，路由层自带一份等价定义；应用侧仍在自己的
//! `api::embedding` 保留 FRB 可见的那份，并在边界处做 `From` 转换，以保证 Dart 侧
//! 枚举名与生成代码**完全不变**。

use embed_core::EmbedError;

/// 嵌入接口的错误分类（与 Dart 侧 `EmbeddingErrorKind` 一一对应）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingErrorKind {
    /// 模型文件或 ORT 运行库不存在（未下载）
    ModelNotDownloaded,
    /// 不支持的图片格式（如 HEIC/AVIF）
    UnsupportedImageFormat,
    /// 文件损坏（截断、解码失败、无法读取）
    CorruptFile,
    /// 图片过大（超出 image::Limits）
    ImageTooLarge,
    /// 推理或会话相关失败
    InferenceFailed,
    /// ORT 运行库加载失败 / 版本不兼容（环境问题，暂时性：换正确运行库后自动恢复）。
    RuntimeUnavailable,
    /// llama 工作进程不可用（未安装/启动失败/超时/崩溃/熔断中）。
    WorkerUnavailable,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbeddingApiError {
    pub kind: EmbeddingErrorKind,
    pub detail: String,
}

impl EmbeddingApiError {
    pub fn new(kind: EmbeddingErrorKind, detail: impl Into<String>) -> Self {
        Self {
            kind,
            detail: detail.into(),
        }
    }

    /// llama 工作进程不可用（未安装/启动失败/熔断中）。
    pub fn worker_unavailable(detail: impl Into<String>) -> Self {
        Self::new(EmbeddingErrorKind::WorkerUnavailable, detail)
    }

    /// 工作进程单次请求超时（已 cancel + kill 重启）。
    pub fn worker_timeout(req_id: u64) -> Self {
        Self::new(
            EmbeddingErrorKind::WorkerUnavailable,
            format!("worker 请求超时 req={req_id}"),
        )
    }

    /// 工作进程崩溃（exit code 见 detail）。
    pub fn worker_crashed(code: Option<i32>) -> Self {
        Self::new(
            EmbeddingErrorKind::WorkerUnavailable,
            format!("worker 崩溃 exit={code:?}"),
        )
    }
}

impl From<EmbedError> for EmbeddingApiError {
    fn from(e: EmbedError) -> Self {
        use EmbeddingErrorKind::*;
        let kind = match e {
            EmbedError::OrtLibraryMissing(_) | EmbedError::ModelMissing(_) => ModelNotDownloaded,
            EmbedError::UnsupportedImage(_) => UnsupportedImageFormat,
            EmbedError::ImageTooLarge(_) => ImageTooLarge,
            EmbedError::InvalidImage(_) => CorruptFile,
            EmbedError::OrtLoad(_) => EmbeddingErrorKind::RuntimeUnavailable,
            EmbedError::Tokenizer(_) | EmbedError::Inference(_) => InferenceFailed,
        };
        Self::new(kind, e.to_string())
    }
}
