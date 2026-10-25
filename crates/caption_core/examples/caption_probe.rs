//! 视频片段结构化描述（VLM Captioning）探针示例。
//!
//! 用法：
//! ```powershell
//! cargo run -p caption_core --example caption_probe -- [FRAME_1.png] [FRAME_2.png] ...
//! ```

use std::env;
use std::path::PathBuf;
use caption_core::{ClipCaption, PromptBuilder};

fn main() {
    let args: Vec<String> = env::args().collect();
    println!("=== 视频片段 VLM 叙事理解与结构化描述探针 ===");

    let frames: Vec<PathBuf> = args[1..]
        .iter()
        .map(|s| PathBuf::from(s))
        .collect();

    println!("输入关键帧数量: {}", frames.len());
    for (i, p) in frames.iter().enumerate() {
        println!("  帧 #{}: {}", i + 1, p.display());
    }

    let frame_count = if frames.is_empty() { 3 } else { frames.len() };
    let user_prompt = PromptBuilder::build_user_prompt(frame_count, None);
    println!("\n[1] 生成 System Prompt:\n{}", PromptBuilder::DEFAULT_SYSTEM_PROMPT);
    println!("\n[2] 生成 User Prompt:\n{}", user_prompt);

    // 模拟一段来自 Qwen2-VL-2B-Instruct 的典型实际响应（含 Markdown 围栏）
    let sample_vlm_output = r#"
Here is the structured analysis of the video clip:
```json
{
  "scene": "outdoor urban park on a sunny afternoon",
  "main_subject": "a young woman in athletic clothing and white sneakers",
  "key_actions": [
    "jogging along the stone path",
    "stopping near a wooden park bench",
    "bending down to tie her shoelaces"
  ],
  "summary": "A young woman is jogging through a sunny park before pausing near a bench to tie her sneakers."
}
```
"#;

    println!("\n[3] 模拟 Qwen2-VL 输出原始文本:\n{}", sample_vlm_output.trim());

    // 解析结构化结果
    let caption = ClipCaption::parse_robust(sample_vlm_output);
    println!("\n[4] 结构化解析结果 (ClipCaption):");
    println!("  - 场景 (scene):         {}", caption.scene);
    println!("  - 主体 (main_subject):  {}", caption.main_subject);
    println!("  - 动作 (key_actions):   {:?}", caption.key_actions);
    println!("  - 摘要 (summary):       {}", caption.summary);

    // 生成 SQLite FTS5 全文索引入库文档
    let fts_doc = caption.to_search_document();
    println!("\n[5] 映射为 SQLite FTS5 索引文档 (即刻支持全文检索):");
    println!("  -> \"{}\"", fts_doc);

    println!("\n✅ Phase 2 认知结构与协议验证完毕！");
}
