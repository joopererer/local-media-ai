//! caption_core — 与应用无关的视频片段/多帧视觉语言模型（VLM）认知分析库。
//!
//! 支持使用 Qwen2-VL 系列模型输入多张视频关键帧，
//! 产出结构化的场景、主体、动作序列与叙事摘要。

pub mod engine;
pub mod model;
pub mod prompt;

pub use engine::{CaptionEngine, CaptionError};
pub use model::ClipCaption;
pub use prompt::PromptBuilder;
