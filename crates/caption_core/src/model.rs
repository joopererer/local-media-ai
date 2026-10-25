//! Data models and parsing for structured video clip captions.

use serde::{Deserialize, Serialize};

/// 视频片段结构化描述结果。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClipCaption {
    /// 场景环境（例如 "indoor meeting room", "outdoor park"）。
    pub scene: String,
    /// 核心主体（例如 "a woman wearing a blue jacket", "golden retriever"）。
    pub main_subject: String,
    /// 动作序列（例如 ["walking in", "sitting down", "drinking coffee"]）。
    pub key_actions: Vec<String>,
    /// 综合故事性一句话摘要（自然语言叙述）。
    pub summary: String,
    /// 模型原始返回文本（用于调试或备用解析）。
    pub raw_response: String,
}

impl ClipCaption {
    /// 构造用于 SQLite FTS5 全文索引检索的统一复合文本。
    pub fn to_search_document(&self) -> String {
        format!(
            "Scene: {}. Subject: {}. Actions: {}. Summary: {}",
            self.scene,
            self.main_subject,
            self.key_actions.join(", "),
            self.summary
        )
    }

    /// 从 VLM 原始文本中鲁棒解析出结构化 ClipCaption。
    /// 兼容 Markdown 围栏 (```json ... ```)、前导杂质文本与未转义字符。
    pub fn parse_robust(raw: &str) -> Self {
        let trimmed = raw.trim();

        // 尝试提取 JSON 区间
        if let Some(json_slice) = extract_json_slice(trimmed) {
            if let Ok(mut parsed) = serde_json::from_str::<ClipCaption>(json_slice) {
                parsed.raw_response = raw.to_string();
                return parsed;
            }

            // 宽容解析：部分模型可能使用下划线或大小写不一的字段
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(json_slice) {
                let scene = v.get("scene")
                    .and_then(|s| s.as_str())
                    .unwrap_or("unknown")
                    .to_string();
                let main_subject = v.get("main_subject")
                    .or_else(|| v.get("subject"))
                    .and_then(|s| s.as_str())
                    .unwrap_or("unknown")
                    .to_string();
                let key_actions = v.get("key_actions")
                    .or_else(|| v.get("actions"))
                    .and_then(|a| a.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|x| x.as_str().map(|s| s.to_string()))
                            .collect()
                    })
                    .unwrap_or_default();
                let summary = v.get("summary")
                    .or_else(|| v.get("description"))
                    .and_then(|s| s.as_str())
                    .unwrap_or(trimmed)
                    .to_string();

                return Self {
                    scene,
                    main_subject,
                    key_actions,
                    summary,
                    raw_response: raw.to_string(),
                };
            }
        }

        // 回退保底：无法解析为 JSON 时，整段作为 summary
        Self {
            scene: "general".to_string(),
            main_subject: "general".to_string(),
            key_actions: Vec::new(),
            summary: trimmed.to_string(),
            raw_response: raw.to_string(),
        }
    }
}

/// 提取文本中最外层的 JSON {...} 区间
fn extract_json_slice(text: &str) -> Option<&str> {
    let start = text.find('{')?;
    let end = text.rfind('}')?;
    if start < end {
        Some(&text[start..=end])
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_clean_json() {
        let raw = r#"{"scene": "indoor office", "main_subject": "man in suit", "key_actions": ["typing", "drinking coffee"], "summary": "A man is typing on his laptop in the office.", "raw_response": ""}"#;
        let c = ClipCaption::parse_robust(raw);
        assert_eq!(c.scene, "indoor office");
        assert_eq!(c.main_subject, "man in suit");
        assert_eq!(c.key_actions, vec!["typing", "drinking coffee"]);
        assert_eq!(c.summary, "A man is typing on his laptop in the office.");
    }

    #[test]
    fn parses_json_wrapped_in_markdown() {
        let raw = "Here is the video analysis:\n```json\n{\n  \"scene\": \"sunny park\",\n  \"main_subject\": \"cute dog\",\n  \"key_actions\": [\"running\", \"catching ball\"],\n  \"summary\": \"A dog runs and catches a ball in the park.\"\n}\n```\nHope this helps!";
        let c = ClipCaption::parse_robust(raw);
        assert_eq!(c.scene, "sunny park");
        assert_eq!(c.main_subject, "cute dog");
        assert_eq!(c.key_actions.len(), 2);
    }

    #[test]
    fn falls_back_gracefully_on_invalid_json() {
        let raw = "A young woman walks down the street on a rainy day holding an umbrella.";
        let c = ClipCaption::parse_robust(raw);
        assert_eq!(c.scene, "general");
        assert_eq!(c.summary, raw);
    }

    #[test]
    fn generates_fts_search_document() {
        let c = ClipCaption {
            scene: "beach".into(),
            main_subject: "surfer".into(),
            key_actions: vec!["paddling".into(), "riding wave".into()],
            summary: "A surfer rides a big wave.".into(),
            raw_response: String::new(),
        };
        let doc = c.to_search_document();
        assert!(doc.contains("Scene: beach"));
        assert!(doc.contains("Subject: surfer"));
        assert!(doc.contains("paddling, riding wave"));
    }
}
