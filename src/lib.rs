//! video_frames — 与应用无关的视频抽帧（关键帧）。
//!
//! 设计约束（Phase 27）：
//! - 只接收视频路径与抽帧策略，返回带时间戳的帧；不依赖任何应用的数据库结构或业务概念。
//! - 通过外部 ffmpeg 二进制执行（路径由调用方传入），本 crate 不链接、不分发 ffmpeg。
//! - 本文件先定义接口与数据类型；实现于 27.4 落地。
//!
//! 本阶段（27.0）仅接口：帧抽取函数返回 [`VideoError::Unavailable`]，不产出伪造帧。

use std::path::{Path, PathBuf};

/// 抽帧策略。
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FrameStrategy {
    /// 固定间隔抽帧（毫秒）。
    Interval { every_ms: u64 },
    /// 镜头变化抽帧：帧间差异阈值（0..=1，越大越稀疏）。
    Scene { threshold: f32 },
}

/// 一帧：临时图片文件 + 它在视频中的时间戳。
#[derive(Debug, Clone, PartialEq)]
pub struct VideoFrame {
    /// 帧图落盘路径（调用方负责清理；通常为临时目录下的 PNG）。
    pub image_path: PathBuf,
    /// 时间戳（毫秒）。
    pub t_ms: u64,
}

/// 抽帧错误分类。
#[derive(Debug, thiserror::Error)]
pub enum VideoError {
    #[error("视频不存在: {0}")]
    NotFound(String),
    #[error("ffmpeg 不可用: {0}")]
    FfmpegUnavailable(String),
    #[error("ffmpeg 执行失败: {0}")]
    FfmpegFailed(String),
    #[error("抽帧未实现/该平台不支持: {0}")]
    Unavailable(String),
}

/// 使用外部 ffmpeg 抽取关键帧。
///
/// `ffmpeg_bin` 为 ffmpeg 可执行文件路径；`out_dir` 用于存放帧图。
pub fn extract_frames(
    ffmpeg_bin: &Path,
    video: &Path,
    strategy: FrameStrategy,
    out_dir: &Path,
) -> Result<Vec<VideoFrame>, VideoError> {
    if !video.is_file() {
        return Err(VideoError::NotFound(video.display().to_string()));
    }
    if !ffmpeg_bin.is_file() {
        return Err(VideoError::FfmpegUnavailable(
            ffmpeg_bin.display().to_string(),
        ));
    }
    // 27.4 落地：按策略构造 ffmpeg 调用（-vf fps / select scene）、解析输出、
    // 返回 (帧图, 时间戳)。本阶段明确返回未实现，避免静默返回空集。
    let _ = (strategy, out_dir);
    Err(VideoError::Unavailable(
        "extract_frames 将在 27.4 实现".into(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_video_is_reported() {
        let r = extract_frames(
            Path::new("ffmpeg"),
            Path::new("/nonexistent.mp4"),
            FrameStrategy::Interval { every_ms: 5000 },
            Path::new("."),
        );
        assert!(matches!(r, Err(VideoError::NotFound(_))));
    }
}
