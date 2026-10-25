# Video Clip Semantic Search & Local-Media-AI Integration Plan

## 概述
为视频片段（本地/网络/流式）提供基于 EmbeddingGemma 2 的多模态语义检索支持，以及基于轻量级多模态视觉大模型（Qwen2-VL-2B）的视频片段剧情自然语言浓缩、稠密描述与关键动作自动打标支持。

---

## 阶段规划与执行进度

### 阶段 1：底层库增强 (`local-media-ai`) [已完成]
- [x] 在 `crates/video_frames` 中实现 `extract_clip_frames`，支持 `start_ms` 与 `end_ms` 区间精准抽帧。
- [x] 支持 HTTP/HTTPS/m3u8 在线流式零下载抽帧（基于 FFmpeg 前置 `-ss` 参数按需请求 Range 字节）。
- [x] 支持 `Scene` 场景切变抽帧策略 (`select='isnan(prev_selected_t)+gt(scene,threshold)'`)。
- [x] 在 `crates/embed_core` 中支持 `embed_image_files` 与 `embed_video_frames` 批量特征向量提取。
- [x] 编写探测示例 `examples/stream_clip_extract.rs` 并验证通过。

### 阶段 2：VLM 稠密描述与动作结构化 (`caption_core`) [已完成]
- [x] 创建 `crates/caption_core`。
- [x] 实现结构化提示词构造器 `PromptBuilder` 与解析引擎 `ClipCaption`。
- [x] 实现鲁棒 JSON 提取器（支持 Markdown 代码块解析与格式容错）。
- [x] 实现 `to_search_document` 导出供全文检索与语义检索复合使用的标准文档。
- [x] 编写 `examples/caption_probe.rs` 并通过测试。

### 阶段 3：统一跨进程 CLI 与 Flutter 桥接 (`local_media_cli` + `AiService`) [已完成]
- [x] 创建 `crates/local_media_cli`，输出独立命令行程序 `local-media-ai.exe`。
- [x] 实现 JSON 通信协议，包括 `doctor`, `extract-clip`, `embed-text`, `embed-frames`, `caption` 命令。
- [x] 进程隔离设计：C++/GPU 原生推理崩溃不会导致 Flutter 应用退出。
- [x] 在 `video_clip_manager` 中创建 `lib/services/ai_service.dart` 与单元测试 `test/ai_service_test.dart`（测试全数通过）。

### 阶段 4：帧特征池与增量叠加引擎 (`DeltaEngine` + `FrameEmbeddingPool`) [已完成]
- [x] 建立母视频绝对时间轴关键帧嵌入池 `lib/services/frame_embedding_pool.dart`。
  - 使用母视频路径 MD5 散列作为唯一键。
  - 帧记录按绝对时间戳 `tMs` 锚定并缓存，支持 JSON 持久化。
  - 实现余弦相似度计算 `cosineSimilarity`。
- [x] 实现增量叠加引擎 `lib/services/delta_engine.dart`。
  - 片段缩短/已抽帧区间：100% 缓存命中，0 抽帧开销，0 推理时间。
  - 片段延展/新区间：仅对缺失的时间区间进行增量抽帧和特征向量计算，并合并写回缓存池。
  - 实现 `computeMaxSim` 检索算法，计算文本向量与片段所有帧的最大相似度，并记录最佳命中帧时间戳 `hitTimestampMs`。
- [x] 编写并通过单元测试 `test/delta_engine_test.dart`（8/8 通过）。

### 阶段 5：混合语义检索与播放器/编辑界面联动 [已完成]
- [x] 扩展 `VideoClip` 数据模型：增加 `aiSummary`、`aiScene`、`aiActions` 和 `hitTimestampMs` 字段。
- [x] 在 `AppState` 中集成 `AiService`、`FrameEmbeddingPool` 与 `DeltaEngine`。
- [x] 扩展 `getFilteredClips`：
  - 关键字搜索自动覆盖 `aiSummary`、`aiScene`、`aiActions`，大幅提升常规搜索召回率。
  - 支持 `performSemanticSearch` 跨片段多模态语义检索，并按相似度得分排序输出。
- [x] 在片段编辑页面 `ClipEditorScreen` 中增加 `[✨ AI 智能解构]` 按钮：
  - 自动调用视觉模型生成剧情概要、场景信息并自动抽取推荐标签。
  - 片段保存时，后台异步调用 `DeltaEngine.syncClipRange` 增量更新母视频帧特征池。
- [x] 播放器 `ClipPlaybackScreen` 增加 `initialSeekTime` 参数：
  - 用户从语义检索结果点击片段时，播放器直接跳跃到最相似的画面关键帧时刻开始播放。
- [x] 片段卡片组件增加 `🎯 命中时刻: MM:SS` 徽标与 `✨ 剧情描述` 预览。
- [x] 编写并通过语义检索及数据状态联动测试 `test/app_state_ai_test.dart`（所有单元测试 17/17 通过）。
