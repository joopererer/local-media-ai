# Video Clip Semantic Search & Local-Media-AI Integration Plan

## 概述
为视频片段（本地/在线/流式）提供基于 EmbeddingGemma 2 的多模态语义搜索支持，以及基于轻量端侧视觉语言模型（Qwen2-VL-2B）的视频片段自然语言叙事与自动打标支持。

---

## 阶段规划

### Phase 1: `local-media-ai` 视频区间与流式抽帧增强 (Rust 侧)
- **1.1 区间抽帧接口**：扩展 `crates/video_frames`，提供 `extract_clip_range(path_or_url, start_ms, end_ms, strategy)` 接口。
- **1.2 流式零下载验证**：
  - HTTP Range MP4：验证 `-ss {start} -to {end} -i {url}` 零落地仅拉取数百 KB 切片。
  - HLS / m3u8：验证解析播放列表仅拉取目标分片（TS/fMP4），秒级出帧。
- **1.3 场景自适应抽帧 (Scene-aware Keyframing)**：集成关键帧 (I-frame) / 场景突变过滤，使 10s 以内的单镜头仅抽取 1~2 帧核心代表画面。
- **1.4 批量向量与探针测试**：完善 `embed_core` 批量帧推理接口，编写独立测试用例。

### Phase 2: `local-media-ai` 认知轨扩展 (Qwen2-VL 视频片段描述)
- **2.1 多图输入与结构化 Prompt**：在 `llama_worker` 扩展多图输入通道，输入 2~3 张关键帧，输出结构化英文描述（场景、主体、动作序列）。
- **2.2 资源消耗与延迟基准**：测试 Q4_K_M 量化版本在 CPU 与核显 Vulkan/DirectML 下的耗时与显存表现。

### Phase 3: 桥接层与 IPC 集成
- **3.1 跨语言绑定**：提供 `flutter_rust_bridge` 或轻量 Sidecar stdio 协议。
- **3.2 运行时门控与模型下载管理**：FFmpeg 环境自检、模型权重后台下载与 sha256 校验。

### Phase 4: `video_clip_manager` 帧池存储与增量区间引擎
- **4.1 绝对时间戳帧池**：建立 `video_frame_embeddings(video_path_hash, t_ms, vector)`，实现母视频帧特征的全局复用。
- **4.2 增量更新算法 (Delta Engine)**：Clip 拖拽微调时，缩短 0 耗时，扩大仅增量补算 $S_{missing} = S_{new} \setminus S_{cached}$。

### Phase 5: 混合检索与播放器精准起播
- **5.1 MaxSim 帧级检索与高亮**：计算与区间内帧的最大相似度，返回最优命中时间点 $t_{hit}$。
- **5.2 混合排序 (Hybrid Search)**：向量检索与 FTS5 全文搜索通过 RRF 算法融合。
- **5.3 播放器精准起播**：点击搜索卡片直接跳转至 $t_{hit}$ 起播。
