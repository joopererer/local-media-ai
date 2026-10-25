//! Prompt builder and schema definitions for Qwen2-VL structured video clip analysis.

/// Prompt 构造器，用于生成针对 Qwen2-VL 视频片段分析的结构化 System 和 User Prompt。
pub struct PromptBuilder;

impl PromptBuilder {
    /// 默认的专家视频分析 System Prompt。
    /// 严格约束模型输出单层 JSON 格式，包含 scene, main_subject, key_actions, summary 字段。
        pub const DEFAULT_SYSTEM_PROMPT: &'static str =
        "You are an expert video clip analyst for professional digital asset management. Analyze the sequence of keyframes. \
        Respond ONLY with a valid JSON object matching this schema: \
        {\"scene\": \"<concise scene noun, <= 8 words>\", \"main_subject\": \"<core subject noun, <= 6 words>\", \"tags\": [\"<atomic keyword 1>\", \"<atomic keyword 2>\", \"<atomic keyword 3>\"], \"key_actions\": [\"<action 1>\", \"<action 2>\"], \"summary\": \"<one-sentence natural narrative>\"}. \
        CRITICAL FOR TAGS: Tags must be single atomic keywords (nouns or verbal nouns, 2-8 characters/words). NEVER output full clauses, sentences, or phrases containing commas, semicolons or punctuation. \
        Do not output any markdown formatting or extra text outside JSON.";

    /// 构造用户查询提示词。
    pub fn build_user_prompt(frame_count: usize, custom_instruction: Option<&str>) -> String {
        let base = format!(
            "Here are {frame_count} sequential keyframes from a video clip in chronological order. \
             Describe what happens across these frames."
        );
        if let Some(extra) = custom_instruction {
            format!("{base} Focus on: {extra}")
        } else {
            base
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_builder_formats_user_prompt() {
        let p = PromptBuilder::build_user_prompt(3, None);
        assert!(p.contains("3 sequential keyframes"));

        let p2 = PromptBuilder::build_user_prompt(2, Some("emotions"));
        assert!(p2.contains("Focus on: emotions"));
    }
}
