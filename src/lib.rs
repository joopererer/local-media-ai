//! ocr_core — 与应用无关的图片/截图文字识别（OCR）。
//!
//! 设计约束（Phase 27）：
//! - 只接收图片路径，返回识别文本与行框；不依赖任何应用的数据库结构或业务概念。
//! - 引擎可替换：Windows.Media.Ocr / PaddleOCR(ONNX) / macOS Vision（实现见 27.3）。
//! - 本文件先定义接口（trait + 数据类型）；引擎实现于 27.3 落地，届时按平台/配置选择。
//!
//! 本阶段（27.0）仅提供接口与占位实现：调用 [`OcrEngine::recognize_file`] 的实现前，
//! 先用 [`UnavailableEngine`] 表示"该平台/引擎未启用"，避免任何伪造的识别结果。

use std::path::Path;

/// 一行识别结果。
#[derive(Debug, Clone, PartialEq)]
pub struct OcrLine {
    /// 该行文本（引擎原始输出，不做拼接或纠错）。
    pub text: String,
    /// 行框 `[x0, y0, x1, y1]`，像素坐标，原点在左上。
    pub bbox: [f32; 4],
    /// 引擎置信度（0..=1；不支持时为 1.0）。
    pub confidence: f32,
}

/// 单张图片的识别结果。
#[derive(Debug, Clone, PartialEq)]
pub struct OcrResult {
    /// 全文（行以 `\n` 连接，保持引擎的行序）。
    pub text: String,
    /// 逐行结果与位置。
    pub lines: Vec<OcrLine>,
    /// 生成该结果的引擎标识（如 `"winrt-ocr"`、`"paddleocr-onnx"`、`"vision"`）。
    pub engine: String,
}

/// OCR 错误分类。
#[derive(Debug, thiserror::Error)]
pub enum OcrError {
    #[error("图片不存在: {0}")]
    NotFound(String),
    #[error("图片无法解码: {0}")]
    Decode(String),
    #[error("引擎不可用: {0}")]
    Unavailable(String),
    #[error("识别失败: {0}")]
    Failed(String),
}

/// 可替换的 OCR 引擎。
pub trait OcrEngine {
    /// 引擎标识（写入 [`OcrResult::engine`]）。
    fn engine_id(&self) -> &'static str;

    /// 识别单张图片文件。
    fn recognize_file(&self, path: &Path) -> Result<OcrResult, OcrError>;
}

/// 未启用/该平台不支持的引擎：始终返回 [`OcrError::Unavailable`]。
///
/// 存在的意义：让上层明确区分"不支持"与"识别到空文本"，不产出伪结果。
pub struct UnavailableEngine {
    pub reason: String,
}

impl OcrEngine for UnavailableEngine {
    fn engine_id(&self) -> &'static str {
        "unavailable"
    }

    fn recognize_file(&self, _path: &Path) -> Result<OcrResult, OcrError> {
        Err(OcrError::Unavailable(self.reason.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unavailable_engine_is_explicit() {
        let e = UnavailableEngine {
            reason: "not built for this platform".into(),
        };
        assert_eq!(e.engine_id(), "unavailable");
        let r = e.recognize_file(Path::new("/nonexistent.png"));
        assert!(matches!(r, Err(OcrError::Unavailable(_))));
    }
}
