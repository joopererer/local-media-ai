//! embed_core — 与应用无关的 EmbeddingGemma 2 嵌入（文本/图片向量）。
//!
//! 设计约束（Phase 27）：
//! - 只接收文件路径或文本，返回向量；不依赖任何应用的数据库结构或业务概念。
//! - 运行方式可替换（CPU ONNX Runtime / GPU / llama.cpp）由上游决定；本 crate 负责
//!   模型专属部分（分词、图片预处理）与 ORT 后端。
//! - 运行库以 `load-dynamic` 方式按绝对路径加载（见 [`init_ort`]），不链接、不下载二进制。
//!
//! 模块：
//! - [`embeddinggemma`]：EmbeddingGemma 2 专属实现（[`Embedder`]）。
//! - [`image_prep`]：图片预处理（缩放、切补丁、位置 id、token 序列），可单独测试。
//!
//! 兼容：下游应用可重新导出本 crate（例如 `app_core::embed`），原调用点路径不变。

pub mod embeddinggemma;
pub mod image_prep;
pub mod worker_frame;

pub use embeddinggemma::*;
