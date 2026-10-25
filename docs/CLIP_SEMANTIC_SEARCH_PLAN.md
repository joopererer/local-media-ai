# Video Clip Semantic Search & Local-Media-AI Integration Plan

## 概述
为视频片段（本地/在线/流式）提供基于 EmbeddingGemma 2 的多模态语义搜索支持，以及基于轻量端侧视觉语言模型（Qwen2-VL-2B）的视频片段自然语言叙事与自动打标支持。

---

## 阶段规划与执行进度

### Phase 1: `local-media-ai` 视频区间与流式抽帧增强 (Rust 侧) [已完成 ✅]
- [x] **1.1 区间抽帧接口**：扩展 `crates/video_frames`，实现 `ClipRange` 结构与 `extract_clip_frames` 接口，支持毫秒级 $[T_{start}, T_{end}]$ 区间抽样。
- [x] **1.2 流式零下载支持**：支持 `http://` / `https://` / `m3u8` 直链，放宽路径存在性硬编码约束；提供 `stream_clip_extract` 测试探针 CLI。
- [x] **1.3 场景自适应抽帧 (Scene-aware Keyframing)**：实现 `FrameStrategy::Scene { threshold }`，采用 FFmpeg 镜头变化过滤器；单镜头静止片段自动压缩至 1 帧核心图。
- [x] **1.4 批量向量与帧特征封装**：完善 `embed_core`，增加 `VideoFrameEmbedding` 与 `embed_video_frames` 批量特征生成。

### Phase 2: `local-media-ai` 认知轨扩展 (Qwen2-VL 视频片段描述) [已完成 ✅]
- [x] **2.1 多图输入与结构化 Prompt**：新增独立 crate `crates/caption_core`，内置针对 Qwen2-VL 的 `PromptBuilder`（约束输出单层 JSON：scene, main_subject, key_actions, summary）。
- [x] **2.2 鲁棒解析与 FTS5 文档转换**：实现 `ClipCaption::parse_robust`（容错 markdown 围栏与前导脏字符）与 `to_search_document()`（自动格式化为全文检索复合文档）；提供 `caption_probe` 探针验证。

### Phase 3: 桥接层与 IPC 集成 [已完成 ✅]
- [x] **3.1 统一跨语言 Sidecar CLI**：在 `local-media-ai` 中实现 `crates/local_media_cli`（产出 `local-media-ai.exe`），暴露 `doctor`、`extract-clip`、`embed-text`、`embed-frames`、`caption` 标准 JSON 协议。
- [x] **3.2 Flutter 端服务封装与自检**：在 `video_clip_manager` 中封装 [`lib/services/ai_service.dart`](file:///d:/workspace/IdeaProjects/video_clip_manager/lib/services/ai_service.dart)，完成自动路径解析、FFmpeg 环境自检、抽帧与认知调用；单元测试全绿通过。

### Phase 4: `video_clip_manager` 帧池存储与增量区间引擎 (进行中 🚀)
- [ ] **4.1 绝对时间戳帧池**：建立 `video_frame_embeddings(video_path_hash, t_ms, vector)`，实现母视频帧特征的全局复用。
- [ ] **4.2 增量更新算法 (Delta Engine)**：Clip 拖拽微调时，缩短 0 耗时，扩大仅增量补算 $S_{missing} = S_{new} \setminus S_{cached}$。

### Phase 5: 混合检索与播放器精准起播
- [ ] **5.1 MaxSim 帧级检索与高亮**：计算与区间内帧的最大相似度，返回最优命中时间点 $t_{hit}$。
- [ ] **5.2 混合排序 (Hybrid Search)**：向量检索与 FTS5 全文搜索通过 RRF 算法融合。
- [ ] **5.3 播放器精准起播**：点击搜索卡片直接跳转至 $t_{hit}$ 起播。
