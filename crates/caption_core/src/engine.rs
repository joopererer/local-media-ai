//! Trait and client definitions for executing VLM captioning on keyframes.

use std::path::Path;
use thiserror::Error;

use crate::model::ClipCaption;

#[derive(Debug, Error)]
pub enum CaptionError {
    #[error("I/O 错误: {0}")]
    Io(#[from] std::io::Error),

    #[error("图片文件不存在: {0}")]
    ImageNotFound(String),

    #[error("模型推理失败: {0}")]
    InferenceFailed(String),

    #[error("未找到 VLM 运行后端二进制: {0}")]
    BackendUnavailable(String),
}

/// 视频片段描述生成引擎抽象。
pub trait CaptionEngine: Send + Sync {
    /// 对传入的多张有序视频关键帧生成结构化剧情描述。
    fn caption_frames(
        &self,
        frames: &[&Path],
        custom_instruction: Option<&str>,
    ) -> Result<ClipCaption, CaptionError>;
}
