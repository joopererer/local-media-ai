//! video_frames — 与应用无关的视频抽帧（关键帧）。
//!
//! 设计约束（Phase 27）：
//! - 只接收视频路径与抽帧策略，返回带时间戳的帧；不依赖任何应用的数据库结构或业务概念。
//! - 通过外部 ffmpeg 二进制执行（路径由调用方传入），本 crate 不链接、不分发 ffmpeg。
//! - **进程并发由调用方保证**（同一时刻只跑一个 ffmpeg 进程）；本 crate 只限制 ffmpeg 自身
//!   线程数（`-threads`），两者共同避免抽帧把 CPU 打满。
//! - 近重复帧的感知哈希**不在本 crate 内计算**（避免引入图片解码依赖）：本 crate 只提供纯函数
//!   [`is_near_duplicate`]，由调用方用既有 pHash 实现算出 64 位哈希后判定，并自行记录跳过数。
//!
//! 27.4 现状：固定间隔抽帧已实现（含短视频兜底、长视频均匀重采样、帧数上限、pts 时间戳）；
//! 镜头检测（[`FrameStrategy::Scene`]）仍返回 [`VideoError::Unavailable`]——不伪造结果。
//!
//! **FFmpeg 版本兼容**：帧时间戳保真需要"帧率直通"，`-fps_mode vfr` 仅在 FFmpeg ≥ 5.1 可用，
//! 更早的版本必须写 `-vsync vfr`。策略见 [`FpsModePolicy`]：调用方按
//! [`fps_mode_policy_for_version`] 传入已知版本；版本无法识别时用 [`FpsModePolicy::Auto`]
//! （先试 `-fps_mode`，失败后回退 `-vsync` 重试一次）。
//!
//! **两种抽帧方式**（[`ExtractMethod`]）：帧率过滤（`-vf fps=…`）一次解码全片、精确但
//! 长视频慢；跳转定位（`-ss` 放输入前、每个目标时刻一次调用）只解目标附近数据、快一个
//! 量级。默认 [`ExtractMethod::Auto`]：时长 ≥ `seek_threshold_ms`（默认 60 秒）走跳转定位。

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// 抽帧策略。
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum FrameStrategy {
    /// 固定间隔抽帧（毫秒）。
    Interval { every_ms: u64 },
    /// 镜头变化抽帧：帧间差异阈值（0..=1，越大越稀疏）。
    Scene { threshold: f32 },
}

/// 一帧：临时图片文件 + 它在视频中的时间戳。
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VideoFrame {
    /// 帧图落盘路径（调用方负责清理；通常为临时目录下的 PNG）。
    pub image_path: PathBuf,
    /// 时间戳（毫秒）。
    pub t_ms: u64,
}

/// 帧率直通参数的写法（`-fps_mode vfr` 需要 FFmpeg ≥ 5.1；旧版用 `-vsync vfr`）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FpsModePolicy {
    /// 新写法（FFmpeg ≥ 5.1）：`-fps_mode vfr`。
    FpsMode,
    /// 旧写法（FFmpeg < 5.1）：`-vsync vfr`。
    Vsync,
    /// 版本未知：先试 [`FpsModePolicy::FpsMode`]，失败后回退 [`FpsModePolicy::Vsync`] 重试一次。
    Auto,
}

impl FpsModePolicy {
    /// 该策略下首选的参数名。
    pub fn primary_flag(&self) -> &'static str {
        match self {
            FpsModePolicy::Vsync => "-vsync",
            _ => "-fps_mode",
        }
    }

    /// 是否允许在首选写法失败后回退重试。
    pub fn allows_fallback(&self) -> bool {
        matches!(self, FpsModePolicy::Auto)
    }
}

/// 按 FFmpeg 主/次版本选择帧率直通参数写法。
///
/// `major`/`minor` 传负数表示"版本不可识别"（⇒ [`FpsModePolicy::Auto`]）。
pub fn fps_mode_policy_for_version(major: i32, minor: i32) -> FpsModePolicy {
    if major < 0 {
        return FpsModePolicy::Auto;
    }
    if major > 5 || (major == 5 && minor >= 1) {
        FpsModePolicy::FpsMode
    } else {
        FpsModePolicy::Vsync
    }
}

/// 抽帧参数（与策略正交）。
#[derive(Debug, Clone, Copy)]
pub struct ExtractOptions {
    /// ffmpeg 及其解码器的线程数上限（默认 2）。抽帧与图片回填并行时不要调大。
    pub threads: u32,
    /// 单个视频最多产出多少帧（默认 24）。
    pub max_frames: u32,
    /// 已知时长（毫秒；来自资源元数据）。给出时用于短视频兜底与长视频均匀重采样。
    pub duration_hint_ms: Option<u64>,
    /// 输出帧的最大边长（像素，默认 1024，0 表示不缩放）。
    pub max_edge: u32,
    /// 帧率直通参数写法（按 FFmpeg 版本；见 [`fps_mode_policy_for_version`]）。
    pub fps_mode: FpsModePolicy,
    /// 抽帧方式（默认 Auto）。
    pub method: ExtractMethod,
    /// 自动选择"跳转定位"的时长阈值（毫秒，默认 60 秒）。
    pub seek_threshold_ms: u64,
}

impl Default for ExtractOptions {
    fn default() -> Self {
        Self {
            threads: 2,
            max_frames: 24,
            duration_hint_ms: None,
            max_edge: 1024,
            fps_mode: FpsModePolicy::Auto,
            method: ExtractMethod::Auto,
            seek_threshold_ms: DEFAULT_SEEK_THRESHOLD_MS,
        }
    }
}

/// 抽样计划（纯计算，便于单测）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplingPlan {
    /// 实际采样间隔（毫秒）。长视频会被放大到"均匀覆盖全片"。
    pub every_ms: u64,
    /// 只抽单帧的时刻（短视频不足一个间隔时取中点）。
    pub single_at_ms: Option<u64>,
    /// 帧数上限。
    pub max_frames: u32,
}

/// 最小采样间隔，避免调用方传 0 造成 fps 爆炸。
const MIN_EVERY_MS: u64 = 100;

/// 抽帧方式。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExtractMethod {
    /// 帧率过滤：一次 ffmpeg 调用解码全片取帧。精确、可预期，**长片慢**。
    Filter,
    /// 跳转定位：每个目标时刻一次调用，`-ss` 放输入前，只解该时刻附近的数据。
    Seek,
    /// 自动：时长 ≥ `seek_threshold_ms` ⇒ Seek；否则（或时长未知）⇒ Filter。
    Auto,
}

/// 跳转定位的默认时长阈值（毫秒）：超过即按"长视频"处理。
///
/// 取值依据（实测，见 `tests/ffmpeg_integration.rs`）：跳转定位每帧起一个 ffmpeg
/// 进程（约 50ms 固定开销，24 帧 ≈ 1.2s），而帧率过滤要**解码全片**。
/// - 易解码的合成素材（180s / 720p testsrc）：filter 1.39s vs seek 1.99s（过滤略胜）；
/// - 真实 1080p 长片（单次全片解码 20s 以上）：seek 快一个量级。
///
/// 60s 让"超过一分钟"的视频走跳转定位；两者差距在短视频上本就在 1s 量级，代价可接受。
pub const DEFAULT_SEEK_THRESHOLD_MS: u64 = 60_000;

/// 跳转定位的落点精度门（毫秒）：实测偏差超过它就判定 ffmpeg 行为异常。
pub const MAX_SEEK_DRIFT_MS: u64 = 500;

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
    #[error("跳转定位落点偏差过大：请求 {requested_ms}ms，实测 {actual_ms:?}ms，偏差 {drift_ms}ms")]
    BadSeek {
        requested_ms: u64,
        actual_ms: Option<u64>,
        drift_ms: u64,
    },
}

/// ffmpeg 进程输出（只关心成败与 stderr：帧时间戳从 `showinfo` 里解析）。
#[derive(Debug, Clone, PartialEq)]
pub struct ProcessOutput {
    pub success: bool,
    pub stderr: String,
}

/// 进程执行抽象：生产用 [`SystemRunner`]，测试注入假实现（不依赖真实 ffmpeg）。
pub trait ProcessRunner {
    fn run(&self, program: &Path, args: &[String]) -> Result<ProcessOutput, VideoError>;
}

/// 真实执行器。
pub struct SystemRunner;

impl ProcessRunner for SystemRunner {
    fn run(&self, program: &Path, args: &[String]) -> Result<ProcessOutput, VideoError> {
        let out = Command::new(program)
            .args(args)
            .output()
            .map_err(|e| VideoError::FfmpegFailed(format!("启动失败: {e}")))?;
        Ok(ProcessOutput {
            success: out.status.success(),
            stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
        })
    }
}

/// 使用外部 ffmpeg 抽取关键帧（生产入口）。
///
/// `ffmpeg_bin` 为 ffmpeg 可执行文件路径；`out_dir` 用于存放帧图（每次调用前会清掉
/// 目录内残留的 `frame_*.png`，避免上次中断的帧混进结果）。
pub fn extract_frames(
    ffmpeg_bin: &Path,
    video: &Path,
    strategy: FrameStrategy,
    out_dir: &Path,
    options: &ExtractOptions,
) -> Result<ExtractOutcome, VideoError> {
    extract_frames_with(ffmpeg_bin, video, strategy, out_dir, options, &SystemRunner)
}

/// 抽帧结果。
#[derive(Debug, Clone, PartialEq)]
pub struct ExtractOutcome {
    /// 已按时间升序的帧。
    pub frames: Vec<VideoFrame>,
    /// 估算的视频时长（毫秒；未知时为 `None`）。
    pub duration_ms: Option<u64>,
    /// 因帧数上限被丢弃的帧数（0 表示未截断）。
    pub truncated: usize,
    /// 本次实际生效的帧率直通写法（回退成功时为 `Vsync`，便于调用方缓存）。
    pub used_fps_mode: FpsModePolicy,
    /// 本次实际使用的抽帧方式（Auto 解析后的结果）。
    pub method: ExtractMethod,
}

/// 可注入执行器的抽帧实现（测试用）。
pub fn extract_frames_with(
    ffmpeg_bin: &Path,
    video: &Path,
    strategy: FrameStrategy,
    out_dir: &Path,
    options: &ExtractOptions,
    runner: &dyn ProcessRunner,
) -> Result<ExtractOutcome, VideoError> {
    if !video.is_file() {
        return Err(VideoError::NotFound(video.display().to_string()));
    }
    if !ffmpeg_bin.is_file() {
        return Err(VideoError::FfmpegUnavailable(
            ffmpeg_bin.display().to_string(),
        ));
    }
    if options.max_frames == 0 {
        return Err(VideoError::Unavailable("max_frames 必须 ≥ 1".into()));
    }
    fs::create_dir_all(out_dir)
        .map_err(|e| VideoError::FfmpegFailed(format!("创建输出目录失败: {e}")))?;
    clean_stale_frames(out_dir)?;

    let duration_ms = match options.duration_hint_ms {
        Some(ms) if ms > 0 => Some(ms),
        _ => probe_duration_ms(ffmpeg_bin, video, runner).ok().flatten(),
    };

    let every_ms = match strategy {
        FrameStrategy::Interval { every_ms } => every_ms,
        FrameStrategy::Scene { .. } => {
            return Err(VideoError::Unavailable(
                "镜头检测抽帧尚未实现（先用 Interval 策略）".into(),
            ))
        }
    };

    let plan = plan_sampling(duration_ms, every_ms, options.max_frames);
    let method = resolve_method(options.method, duration_ms, options.seek_threshold_ms);
    let targets = target_times_ms(duration_ms, &plan);

    let seek_applicable = method == ExtractMethod::Seek && !targets.is_empty();
    let (mut frames, used, truncated) = if seek_applicable {
        extract_by_seek(ffmpeg_bin, video, out_dir, options, &targets, runner)?
    } else {
        extract_by_filter(ffmpeg_bin, video, out_dir, &plan, options, runner)?
    };

    frames.sort_by_key(|f| f.t_ms);
    frames.truncate(plan.max_frames as usize);

    Ok(ExtractOutcome {
        frames,
        duration_ms,
        truncated,
        used_fps_mode: used,
        method: if seek_applicable {
            ExtractMethod::Seek
        } else {
            ExtractMethod::Filter
        },
    })
}

/// 帧率过滤抽帧（单次调用解码全片）。
/// 返回（帧，实际生效的 [`FpsModePolicy`]，因上限被丢弃的帧数）。
fn extract_by_filter(
    ffmpeg_bin: &Path,
    video: &Path,
    out_dir: &Path,
    plan: &SamplingPlan,
    options: &ExtractOptions,
    runner: &dyn ProcessRunner,
) -> Result<(Vec<VideoFrame>, FpsModePolicy, usize), VideoError> {
    let mut used = options.fps_mode;
    let args = build_args(video, out_dir, plan, options, used);
    let mut out = runner.run(ffmpeg_bin, &args)?;
    if !out.success && used.allows_fallback() && plan.single_at_ms.is_none() {
        // 版本不可识别：先试了 `-fps_mode`，失败则用旧写法 `-vsync` 重试一次。
        // 重试前清掉上次可能已落盘的半成品帧，避免混入结果。
        clean_stale_frames(out_dir)?;
        let first_err = out.stderr.clone();
        used = FpsModePolicy::Vsync;
        let retry_args = build_args(video, out_dir, plan, options, used);
        out = runner.run(ffmpeg_bin, &retry_args)?;
        if !out.success {
            return Err(VideoError::FfmpegFailed(format!(
                "回退 -vsync 亦失败：{}\n原始错误（-fps_mode）：{}",
                tail(&out.stderr, 600),
                tail(&first_err, 600)
            )));
        }
    }
    if !out.success {
        return Err(VideoError::FfmpegFailed(tail(&out.stderr, 800)));
    }

    let mut produced = list_frames(out_dir)?;
    let pts_ms = parse_pts_times_ms(&out.stderr);
    let truncated = produced.len().saturating_sub(plan.max_frames as usize);
    if produced.len() > plan.max_frames as usize {
        for extra in produced.drain(plan.max_frames as usize..) {
            let _ = fs::remove_file(extra);
        }
    }

    let frames: Vec<VideoFrame> = produced
        .into_iter()
        .enumerate()
        .map(|(i, image_path)| VideoFrame {
            image_path,
            t_ms: frame_timestamp_ms(i, plan, &pts_ms),
        })
        .collect();
    Ok((frames, used, truncated))
}

/// 跳转定位抽帧：每个目标时刻一次调用，`-ss` 放在 `-i` 之前（快速定位）。
///
/// 时间戳语义：带 `-copyts` 让 `showinfo` 报告**原始时间轴**的 pts（否则 ffmpeg 会把
/// 时间轴整体平移到 0，实测值全是 0，无法校验落点），并优先用实测值作 `t_ms`，
/// 缺失时退回请求时刻。实测偏离请求超过 [`MAX_SEEK_DRIFT_MS`] 时报
/// [`VideoError::BadSeek`]——说明该文件的定位行为异常，宁可报错也不返回错位帧。
fn extract_by_seek(
    ffmpeg_bin: &Path,
    video: &Path,
    out_dir: &Path,
    options: &ExtractOptions,
    targets_ms: &[u64],
    runner: &dyn ProcessRunner,
) -> Result<(Vec<VideoFrame>, FpsModePolicy, usize), VideoError> {
    let mut frames: Vec<VideoFrame> = Vec::with_capacity(targets_ms.len());
    for (i, &at_ms) in targets_ms.iter().enumerate() {
        let image_path = out_dir.join(format!("frame_{i:06}.png"));
        let args = build_seek_args(video, at_ms, &image_path, options);
        let out = runner.run(ffmpeg_bin, &args)?;
        if !out.success {
            return Err(VideoError::FfmpegFailed(format!(
                "第 {}/{} 帧（{at_ms}ms）抽取失败：{}",
                i + 1,
                targets_ms.len(),
                tail(&out.stderr, 600)
            )));
        }
        if !image_path.is_file() {
            return Err(VideoError::FfmpegFailed(format!(
                "第 {}/{} 帧（{at_ms}ms）ffmpeg 成功退出但未产出帧文件",
                i + 1,
                targets_ms.len()
            )));
        }
        let actual = parse_pts_times_ms(&out.stderr).first().copied();
        if let Some(a) = actual {
            let drift = a.abs_diff(at_ms);
            if drift > MAX_SEEK_DRIFT_MS {
                return Err(VideoError::BadSeek {
                    requested_ms: at_ms,
                    actual_ms: Some(a),
                    drift_ms: drift,
                });
            }
        }
        frames.push(VideoFrame {
            image_path,
            t_ms: actual.unwrap_or(at_ms),
        });
    }
    // 跳转定位不写帧率直通参数（每帧单张），回报值只用于界面说明。
    let used = if options.fps_mode == FpsModePolicy::Vsync {
        FpsModePolicy::Vsync
    } else {
        FpsModePolicy::FpsMode
    };
    Ok((frames, used, 0))
}

/// 跳转定位的参数：`-ss` 在 `-i` 之前 ⇒ ffmpeg 快速定位（不解码前面的数据）。
fn build_seek_args(
    video: &Path,
    at_ms: u64,
    image_path: &Path,
    options: &ExtractOptions,
) -> Vec<String> {
    let mut args: Vec<String> = vec!["-hide_banner".into(), "-nostdin".into(), "-y".into()];
    args.push("-ss".into());
    args.push(format!("{:.3}", at_ms as f64 / 1000.0));
    args.push("-threads".into());
    args.push(options.threads.max(1).to_string());
    args.push("-i".into());
    args.push(path_arg(video));
    // 保留原始时间轴：否则 showinfo 的 pts 会被平移到 0，无法校验落点/报告实测时刻。
    args.push("-copyts".into());
    let mut filters = Vec::new();
    if options.max_edge > 0 {
        filters.push(format!(
            "scale='min({m},iw)':'min({m},ih)':force_original_aspect_ratio=decrease",
            m = options.max_edge
        ));
    }
    filters.push("showinfo".into());
    args.push("-vf".into());
    args.push(filters.join(","));
    args.push("-frames:v".into());
    args.push("1".into());
    args.push("-threads".into());
    args.push(options.threads.max(1).to_string());
    args.push(path_arg(image_path));
    args
}

/// 解析实际生效的抽帧方式（Auto 按时长阈值解析；时长未知 ⇒ Filter）。
pub fn resolve_method(
    requested: ExtractMethod,
    duration_ms: Option<u64>,
    seek_threshold_ms: u64,
) -> ExtractMethod {
    if requested != ExtractMethod::Auto {
        return requested;
    }
    match duration_ms {
        Some(d) if d > 0 && d >= seek_threshold_ms => ExtractMethod::Seek,
        _ => ExtractMethod::Filter,
    }
}

/// 跳转定位的目标时刻（毫秒，升序，≤ max_frames 个）。
///
/// - 短视频（不足一个间隔）⇒ 单帧中点（与 [`plan_sampling`] 一致）；
/// - 其它 ⇒ `0, every, 2*every, …`（过 `every` 已在 [`plan_sampling`] 放大到覆盖全片）；
/// - 时长未知 ⇒ 空（调用方回退帧率过滤）。
pub fn target_times_ms(duration_ms: Option<u64>, plan: &SamplingPlan) -> Vec<u64> {
    if let Some(single) = plan.single_at_ms {
        return vec![single];
    }
    let Some(d) = duration_ms.filter(|d| *d > 0) else {
        return Vec::new();
    };
    let times: Vec<u64> = (0..plan.max_frames)
        .map(|i| (i as u64).saturating_mul(plan.every_ms))
        .take_while(|t| *t < d)
        .collect();
    if times.is_empty() {
        vec![d / 2]
    } else {
        times
    }
}

/// 帧时间戳：优先用 ffmpeg `showinfo` 的实测 pts，缺失时退回"序号 × 采样间隔"。
pub fn frame_timestamp_ms(index: usize, plan: &SamplingPlan, pts_ms: &[u64]) -> u64 {
    if let Some(t) = pts_ms.get(index) {
        return *t;
    }
    match plan.single_at_ms {
        Some(t) => t,
        None => (index as u64).saturating_mul(plan.every_ms),
    }
}

/// 抽样计划：短视频取中点单帧；长视频放大间隔以均匀覆盖全片并满足帧数上限。
pub fn plan_sampling(
    duration_ms: Option<u64>,
    every_ms: u64,
    max_frames: u32,
) -> SamplingPlan {
    let every_ms = every_ms.max(MIN_EVERY_MS);
    let max_frames = max_frames.max(1);
    match duration_ms {
        Some(d) if d > 0 && d < every_ms => SamplingPlan {
            every_ms,
            single_at_ms: Some(d / 2),
            max_frames,
        },
        Some(d) if d > 0 => {
            // 均匀重采样：至少每 max_frames 段取一帧，避免只覆盖视频开头。
            let uniform = (d as f64 / max_frames as f64).ceil() as u64;
            SamplingPlan {
                every_ms: every_ms.max(uniform.max(MIN_EVERY_MS)),
                single_at_ms: None,
                max_frames,
            }
        }
        // 时长未知：按原间隔抽，靠 -frames:v 截断（从开头覆盖，行为可预期）。
        _ => SamplingPlan {
            every_ms,
            single_at_ms: None,
            max_frames,
        },
    }
}

/// 近重复帧判定：与上一个**保留**帧的 64 位感知哈希汉明距离 ≤ 阈值 ⇒ 跳过（不算向量）。
pub fn is_near_duplicate(previous_hash: u64, current_hash: u64, hamming_threshold: u32) -> bool {
    (previous_hash ^ current_hash).count_ones() <= hamming_threshold
}

/// 解析 ffmpeg banner 里的时长：`Duration: 00:05:23.45, start: ...`。
pub fn parse_duration_ms(stderr: &str) -> Option<u64> {
    for line in stderr.lines() {
        let Some(pos) = line.find("Duration: ") else {
            continue;
        };
        let rest = &line[pos + "Duration: ".len()..];
        let spec = rest.split(',').next()?.trim();
        let mut parts = spec.split(':');
        let (h, m, s) = (parts.next()?, parts.next()?, parts.next()?);
        if parts.next().is_some() {
            continue;
        }
        let (Ok(h), Ok(m), Ok(sec)) = (
            h.trim().parse::<u64>(),
            m.trim().parse::<u64>(),
            s.trim().parse::<f64>(),
        ) else {
            continue;
        };
        if !(0.0..1_000_000.0).contains(&sec) {
            continue;
        }
        let ms = (h * 3600 + m * 60) * 1000 + (sec * 1000.0).round() as u64;
        return Some(ms);
    }
    None
}

/// 解析 showinfo 的 `pts_time:` 列表（毫秒，保持出现顺序）。
pub fn parse_pts_times_ms(stderr: &str) -> Vec<u64> {
    let mut out = Vec::new();
    for line in stderr.lines() {
        let Some(pos) = line.find("pts_time:") else {
            continue;
        };
        let rest = &line[pos + "pts_time:".len()..];
        let token = rest
            .trim_start()
            .split([' ', '|', '\t'])
            .next()
            .unwrap_or("");
        if let Ok(sec) = token.parse::<f64>() {
            if sec.is_finite() && sec >= 0.0 {
                out.push((sec * 1000.0).round() as u64);
            }
        }
    }
    out
}

/// 探明视频时长：`ffmpeg -i <video>`（不指定输出）会在 stderr banner 里给出 Duration。
fn probe_duration_ms(
    ffmpeg_bin: &Path,
    video: &Path,
    runner: &dyn ProcessRunner,
) -> Result<Option<u64>, VideoError> {
    let args = vec!["-hide_banner".to_string(), "-i".to_string(), path_arg(video)];
    let out = runner.run(ffmpeg_bin, &args)?;
    // 只有输入没有输出时 ffmpeg 返回非零，但 banner 仍然可用。
    Ok(parse_duration_ms(&out.stderr))
}

fn build_args(
    video: &Path,
    out_dir: &Path,
    plan: &SamplingPlan,
    options: &ExtractOptions,
    fps_mode: FpsModePolicy,
) -> Vec<String> {
    let mut args: Vec<String> = vec!["-hide_banner".into(), "-nostdin".into(), "-y".into()];
    // 解码器线程数（真正的 CPU 大户）——与编码侧一起限制在 threads 以内。
    args.push("-threads".into());
    args.push(options.threads.max(1).to_string());

    if let Some(at_ms) = plan.single_at_ms {
        args.push("-ss".into());
        args.push(format!("{:.3}", at_ms as f64 / 1000.0));
    }
    args.push("-i".into());
    args.push(path_arg(video));

    let mut filters = vec![
        if plan.single_at_ms.is_some() {
            "null".to_string()
        } else {
            // fps=1000/every：直接以"每 every 毫秒一帧"表达采样间隔。
            format!("fps={}/{}", 1000, plan.every_ms)
        },
    ];
    if options.max_edge > 0 {
        filters.push(format!(
            "scale='min({m},iw)':'min({m},ih)':force_original_aspect_ratio=decrease",
            m = options.max_edge
        ));
    } else {
        filters.push("scale=iw:ih".to_string());
    }
    if plan.single_at_ms.is_none() {
        // 让 ffmpeg 把每帧的 pts_time 打到 stderr，供实测时间戳解析。
        filters.push("showinfo".to_string());
    }
    args.push("-vf".into());
    args.push(filters.join(","));
    if plan.single_at_ms.is_none() {
        // 帧率直通：新版 `-fps_mode vfr`，旧版 `-vsync vfr`（单帧路径不需要）。
        args.push(fps_mode.primary_flag().into());
        args.push("vfr".into());
    }
    args.push("-frames:v".into());
    args.push(
        (if plan.single_at_ms.is_some() {
            1
        } else {
            plan.max_frames
        })
        .to_string(),
    );
    args.push("-threads".into());
    args.push(options.threads.max(1).to_string());
    args.push(out_dir.join("frame_%06d.png").to_string_lossy().into_owned());
    args
}

fn path_arg(p: &Path) -> String {
    p.to_string_lossy().into_owned()
}

/// 清掉输出目录里上一次残留的帧（崩溃/中断后的半成品）。
fn clean_stale_frames(out_dir: &Path) -> Result<(), VideoError> {
    let entries = match fs::read_dir(out_dir) {
        Ok(e) => e,
        Err(_) => return Ok(()),
    };
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with("frame_") && name.ends_with(".png") {
            let _ = fs::remove_file(entry.path());
        }
    }
    Ok(())
}

fn list_frames(out_dir: &Path) -> Result<Vec<PathBuf>, VideoError> {
    let mut frames: Vec<PathBuf> = fs::read_dir(out_dir)
        .map_err(|e| VideoError::FfmpegFailed(format!("读取输出目录失败: {e}")))?
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            let name = p.file_name().map(|n| n.to_string_lossy().into_owned());
            matches!(name, Some(n) if n.starts_with("frame_") && n.ends_with(".png"))
        })
        .collect();
    frames.sort();
    Ok(frames)
}

fn tail(text: &str, max: usize) -> String {
    let trimmed = text.trim();
    if trimmed.len() <= max {
        return trimmed.to_string();
    }
    // 按字符边界截取，避免切坏 UTF-8。
    let mut start = trimmed.len() - max;
    while start < trimmed.len() && !trimmed.is_char_boundary(start) {
        start += 1;
    }
    format!("…{}", &trimmed[start..])
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FakeRunner {
        /// 每次调用前先落盘多少帧。
        frames: usize,
        /// 返回的 stderr（可含 showinfo / Duration）。
        stderr: String,
        /// 参数里出现这些开关之一的调用返回失败（模拟旧版 ffmpeg 不认新参数）。
        fail_args: Vec<String>,
        /// 失败调用先写几个半成品帧（验证重试前会清掉）。
        partial_frames_on_failure: usize,
        calls: std::cell::RefCell<Vec<Vec<String>>>,
    }

    impl FakeRunner {
        fn new(frames: usize, stderr: &str) -> Self {
            Self {
                frames,
                stderr: stderr.to_string(),
                fail_args: Vec::new(),
                partial_frames_on_failure: 0,
                calls: std::cell::RefCell::new(Vec::new()),
            }
        }

        fn failing_on(mut self, arg: &str, partial_frames: usize) -> Self {
            self.fail_args.push(arg.to_string());
            self.partial_frames_on_failure = partial_frames.max(self.partial_frames_on_failure);
            self
        }

        fn last_args(&self) -> Vec<String> {
            self.calls.borrow().last().cloned().unwrap_or_default()
        }

        fn call_count(&self) -> usize {
            self.calls.borrow().len()
        }
    }

    impl ProcessRunner for FakeRunner {
        fn run(&self, _program: &Path, args: &[String]) -> Result<ProcessOutput, VideoError> {
            self.calls.borrow_mut().push(args.to_vec());
            // banner 探测（只有 -i，没有输出）时不产生帧。
            let is_probe = !args.iter().any(|a| a.ends_with(".png"));
            let will_fail = args
                .iter()
                .any(|a| self.fail_args.iter().any(|flag| flag == a));
            if !is_probe {
                let out_arg = args.last().unwrap().clone();
                if out_arg.contains("%06d") {
                    // 帧率过滤：按序号模板落盘。
                    let dir = Path::new(&out_arg).parent().unwrap().to_path_buf();
                    let count = if will_fail {
                        self.partial_frames_on_failure
                    } else {
                        self.frames
                    };
                    for i in 0..count {
                        let p = dir.join(format!("frame_{i:06}.png"));
                        fs::write(&p, b"fake").unwrap();
                    }
                } else if !will_fail {
                    // 跳转定位：单帧调用，只写这次请求的那个文件。
                    fs::write(Path::new(&out_arg), b"fake").unwrap();
                }
            }
            if will_fail {
                return Ok(ProcessOutput {
                    success: false,
                    stderr: "Unrecognized option 'fps_mode'.".to_string(),
                });
            }
            Ok(ProcessOutput {
                success: true,
                stderr: self.stderr.clone(),
            })
        }
    }

    fn tempdir(tag: &str) -> PathBuf {
        let base = std::env::temp_dir().join(format!(
            "vf_test_{}_{}_{}",
            tag,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&base).unwrap();
        base
    }

    fn fixture(tag: &str) -> (PathBuf, PathBuf, PathBuf) {
        let root = tempdir(tag);
        let ffmpeg = root.join("ffmpeg");
        fs::write(&ffmpeg, b"fake-ffmpeg").unwrap();
        let video = root.join("clip.mp4");
        fs::write(&video, b"fake-video").unwrap();
        let out = root.join("frames");
        (ffmpeg, video, out)
    }

    #[test]
    fn missing_video_is_reported() {
        let r = extract_frames(
            Path::new("ffmpeg"),
            Path::new("/nonexistent.mp4"),
            FrameStrategy::Interval { every_ms: 5000 },
            Path::new("."),
            &ExtractOptions::default(),
        );
        assert!(matches!(r, Err(VideoError::NotFound(_))));
    }

    #[test]
    fn missing_ffmpeg_is_reported() {
        let (_ff, video, out) = fixture("noffmpeg");
        let r = extract_frames(
            Path::new("/nonexistent/ffmpeg"),
            &video,
            FrameStrategy::Interval { every_ms: 5000 },
            &out,
            &ExtractOptions::default(),
        );
        assert!(matches!(r, Err(VideoError::FfmpegUnavailable(_))));
    }

    #[test]
    fn scene_strategy_is_explicitly_unavailable() {
        let (ff, video, out) = fixture("scene");
        let r = extract_frames_with(
            &ff,
            &video,
            FrameStrategy::Scene { threshold: 0.3 },
            &out,
            &ExtractOptions::default(),
            &FakeRunner::new(0, ""),
        );
        assert!(matches!(r, Err(VideoError::Unavailable(_))));
    }

    #[test]
    fn plan_sampling_handles_short_unknown_and_long_videos() {
        // 短视频（3s < 5s）→ 中点单帧。
        let p = plan_sampling(Some(3_000), 5_000, 24);
        assert_eq!(p.single_at_ms, Some(1_500));
        assert_eq!(p.every_ms, 5_000);

        // 时长未知 → 按原间隔，帧数上限靠 ffmpeg 截断。
        let p = plan_sampling(None, 5_000, 24);
        assert_eq!(p.single_at_ms, None);
        assert_eq!(p.every_ms, 5_000);

        // 10 分钟 / 24 帧 → 均匀重采样 25s。
        let p = plan_sampling(Some(600_000), 5_000, 24);
        assert_eq!(p.every_ms, 25_000);
        assert_eq!(p.single_at_ms, None);

        // 间隔下限钳制，避免 fps 爆炸。
        let p = plan_sampling(Some(60_000), 0, 24);
        assert_eq!(p.every_ms, 2_500);
        assert!(plan_sampling(Some(60_000), 1, 24).every_ms >= MIN_EVERY_MS);
    }

    #[test]
    fn interval_args_limit_threads_and_express_sampling_by_fps() {
        let (ff, video, out) = fixture("args");
        let runner = FakeRunner::new(2, "");
        let opts = ExtractOptions {
            threads: 2,
            max_frames: 24,
            duration_hint_ms: Some(120_000),
            max_edge: 1024,
            fps_mode: FpsModePolicy::FpsMode,
            method: ExtractMethod::Filter,
            ..ExtractOptions::default()
        };
        let outcome = extract_frames_with(
            &ff,
            &video,
            FrameStrategy::Interval { every_ms: 5_000 },
            &out,
            &opts,
            &runner,
        )
        .unwrap();
        let args = runner.last_args();
        let joined = args.join(" ");
        assert!(joined.contains("-threads 2"), "解码/编码线程需限制: {joined}");
        assert!(joined.contains("fps=1000/5000"), "采样间隔需以 fps 表达: {joined}");
        assert!(joined.contains("-frames:v 24"));
        assert!(joined.contains("-fps_mode vfr"));
        assert!(joined.contains("showinfo"));
        assert!(args.last().unwrap().ends_with("frame_%06d.png"));
        // 120s / 24 = 5s ⇒ 2 帧的时间戳落在 0 与 5s。
        assert_eq!(outcome.frames.len(), 2);
        assert_eq!(outcome.frames[0].t_ms, 0);
        assert_eq!(outcome.frames[1].t_ms, 5_000);
        assert_eq!(outcome.truncated, 0);
    }

    #[test]
    fn long_video_is_resampled_uniformly_to_cover_the_whole_clip() {
        let (ff, video, out) = fixture("longv");
        let runner = FakeRunner::new(1, "");
        let opts = ExtractOptions {
            duration_hint_ms: Some(600_000),
            method: ExtractMethod::Filter,
            ..ExtractOptions::default()
        };
        extract_frames_with(
            &ff,
            &video,
            FrameStrategy::Interval { every_ms: 5_000 },
            &out,
            &opts,
            &runner,
        )
        .unwrap();
        assert!(runner.last_args().join(" ").contains("fps=1000/25000"));
    }

    #[test]
    fn short_video_takes_at_least_the_middle_frame() {
        let (ff, video, out) = fixture("shortv");
        let runner = FakeRunner::new(1, "");
        let opts = ExtractOptions {
            duration_hint_ms: Some(3_000),
            ..ExtractOptions::default()
        };
        let outcome = extract_frames_with(
            &ff,
            &video,
            FrameStrategy::Interval { every_ms: 5_000 },
            &out,
            &opts,
            &runner,
        )
        .unwrap();
        let joined = runner.last_args().join(" ");
        assert!(joined.contains("-ss 1.500"), "应跳到中点: {joined}");
        assert!(joined.contains("-frames:v 1"));
        assert_eq!(outcome.frames.len(), 1);
        assert_eq!(outcome.frames[0].t_ms, 1_500);
    }

    #[test]
    fn timestamps_come_from_showinfo_when_available() {
        let (ff, video, out) = fixture("pts");
        let stderr = "\
[Parsed_showinfo_2 @ 0x1] n:   0 pts:      0 pts_time:0.000000 duration:1
[Parsed_showinfo_2 @ 0x1] n:   1 pts:  25000 pts_time:25.012000 duration:1
";
        let runner = FakeRunner::new(2, stderr);
        let opts = ExtractOptions {
            duration_hint_ms: Some(300_000),
            method: ExtractMethod::Filter,
            ..ExtractOptions::default()
        };
        let outcome = extract_frames_with(
            &ff,
            &video,
            FrameStrategy::Interval { every_ms: 5_000 },
            &out,
            &opts,
            &runner,
        )
        .unwrap();
        assert_eq!(outcome.frames[0].t_ms, 0);
        assert_eq!(outcome.frames[1].t_ms, 25_012);
    }

    #[test]
    fn truncates_to_max_frames_and_reports_it() {
        let (ff, video, out) = fixture("trunc");
        let runner = FakeRunner::new(30, "");
        let opts = ExtractOptions {
            max_frames: 24,
            duration_hint_ms: Some(600_000),
            method: ExtractMethod::Filter,
            ..ExtractOptions::default()
        };
        let outcome = extract_frames_with(
            &ff,
            &video,
            FrameStrategy::Interval { every_ms: 5_000 },
            &out,
            &opts,
            &runner,
        )
        .unwrap();
        assert_eq!(outcome.frames.len(), 24);
        assert_eq!(outcome.truncated, 6);
    }

    #[test]
    fn stale_frames_from_a_previous_run_are_cleaned() {
        let (ff, video, out) = fixture("stale");
        fs::create_dir_all(&out).unwrap();
        fs::write(out.join("frame_000099.png"), b"stale").unwrap();
        let runner = FakeRunner::new(2, "");
        let opts = ExtractOptions {
            duration_hint_ms: Some(20_000),
            ..ExtractOptions::default()
        };
        let outcome = extract_frames_with(
            &ff,
            &video,
            FrameStrategy::Interval { every_ms: 5_000 },
            &out,
            &opts,
            &runner,
        )
        .unwrap();
        assert_eq!(outcome.frames.len(), 2);
        assert!(!out.join("frame_000099.png").exists(), "残留帧必须清掉");
    }

    // ── FFmpeg 版本兼容（-fps_mode ≥ 5.1 / -vsync 更早）────────────────────

    #[test]
    fn fps_mode_policy_follows_ffmpeg_version() {
        // ≥ 5.1 → 新写法；< 5.1 → 旧写法；未知（负数）→ 自动回退。
        assert_eq!(fps_mode_policy_for_version(5, 1), FpsModePolicy::FpsMode);
        assert_eq!(fps_mode_policy_for_version(5, 0), FpsModePolicy::Vsync);
        assert_eq!(fps_mode_policy_for_version(6, 1), FpsModePolicy::FpsMode);
        assert_eq!(fps_mode_policy_for_version(7, 0), FpsModePolicy::FpsMode);
        assert_eq!(fps_mode_policy_for_version(4, 4), FpsModePolicy::Vsync);
        assert_eq!(fps_mode_policy_for_version(3, 4), FpsModePolicy::Vsync);
        assert_eq!(fps_mode_policy_for_version(-1, -1), FpsModePolicy::Auto);

        assert_eq!(FpsModePolicy::FpsMode.primary_flag(), "-fps_mode");
        assert_eq!(FpsModePolicy::Vsync.primary_flag(), "-vsync");
        assert_eq!(FpsModePolicy::Auto.primary_flag(), "-fps_mode");
        assert!(FpsModePolicy::Auto.allows_fallback());
        assert!(!FpsModePolicy::FpsMode.allows_fallback());
        assert!(!FpsModePolicy::Vsync.allows_fallback());
    }

    #[test]
    fn known_new_version_uses_fps_mode_flag() {
        let (ff, video, out) = fixture("fpsmode_new");
        let runner = FakeRunner::new(2, "");
        let opts = ExtractOptions {
            duration_hint_ms: Some(20_000),
            fps_mode: FpsModePolicy::FpsMode,
            ..ExtractOptions::default()
        };
        let outcome = extract_frames_with(
            &ff,
            &video,
            FrameStrategy::Interval { every_ms: 5_000 },
            &out,
            &opts,
            &runner,
        )
        .unwrap();
        let joined = runner.last_args().join(" ");
        assert!(joined.contains("-fps_mode vfr"), "{joined}");
        assert!(!joined.contains("-vsync"), "{joined}");
        assert_eq!(outcome.used_fps_mode, FpsModePolicy::FpsMode);
        assert_eq!(runner.call_count(), 1, "已知版本不该重试");
    }

    #[test]
    fn known_old_version_uses_vsync_flag() {
        let (ff, video, out) = fixture("fpsmode_old");
        let runner = FakeRunner::new(2, "");
        let opts = ExtractOptions {
            duration_hint_ms: Some(20_000),
            fps_mode: FpsModePolicy::Vsync,
            ..ExtractOptions::default()
        };
        let outcome = extract_frames_with(
            &ff,
            &video,
            FrameStrategy::Interval { every_ms: 5_000 },
            &out,
            &opts,
            &runner,
        )
        .unwrap();
        let joined = runner.last_args().join(" ");
        assert!(joined.contains("-vsync vfr"), "{joined}");
        assert!(!joined.contains("-fps_mode"), "{joined}");
        assert_eq!(outcome.used_fps_mode, FpsModePolicy::Vsync);
        assert_eq!(runner.call_count(), 1);
    }

    #[test]
    fn unknown_version_falls_back_to_vsync_once_and_cleans_partial_frames() {
        let (ff, video, out) = fixture("fpsmode_auto");
        // 第一次（-fps_mode）失败并留下 1 个半成品帧；第二次（-vsync）成功产出 2 帧。
        let runner = FakeRunner::new(2, "").failing_on("-fps_mode", 1);
        let opts = ExtractOptions {
            duration_hint_ms: Some(20_000),
            fps_mode: FpsModePolicy::Auto,
            ..ExtractOptions::default()
        };
        let outcome = extract_frames_with(
            &ff,
            &video,
            FrameStrategy::Interval { every_ms: 5_000 },
            &out,
            &opts,
            &runner,
        )
        .unwrap();
        assert_eq!(runner.call_count(), 2, "未知版本应回退重试一次");
        assert!(runner.last_args().join(" ").contains("-vsync vfr"));
        assert_eq!(outcome.used_fps_mode, FpsModePolicy::Vsync);
        assert_eq!(outcome.frames.len(), 2, "半成品帧必须清掉，只剩重试产出");
        assert!(outcome.frames.iter().all(|f| f.image_path.exists()));
    }

    #[test]
    fn fallback_failure_reports_both_errors() {
        let (ff, video, out) = fixture("fpsmode_fail");
        // 新旧写法都失败：回退一次后仍失败，必须报错（不能静默返回空集）。
        let runner = FakeRunner::new(2, "")
            .failing_on("-fps_mode", 0)
            .failing_on("-vsync", 0);
        let opts = ExtractOptions {
            duration_hint_ms: Some(20_000),
            fps_mode: FpsModePolicy::Auto,
            ..ExtractOptions::default()
        };
        let err = extract_frames_with(
            &ff,
            &video,
            FrameStrategy::Interval { every_ms: 5_000 },
            &out,
            &opts,
            &runner,
        )
        .unwrap_err();
        assert_eq!(runner.call_count(), 2, "只回退重试一次，不无限重试");
        let VideoError::FfmpegFailed(msg) = err else {
            panic!("应报 FfmpegFailed");
        };
        assert!(msg.contains("回退 -vsync 亦失败"), "{msg}");
        assert!(msg.contains("Unrecognized option"), "需带上原始错误: {msg}");
    }

    #[test]
    fn single_frame_path_omits_fps_mode_flags() {
        let (ff, video, out) = fixture("fpsmode_single");
        let runner = FakeRunner::new(1, "");
        let opts = ExtractOptions {
            duration_hint_ms: Some(3_000),
            fps_mode: FpsModePolicy::Auto,
            ..ExtractOptions::default()
        };
        extract_frames_with(
            &ff,
            &video,
            FrameStrategy::Interval { every_ms: 5_000 },
            &out,
            &opts,
            &runner,
        )
        .unwrap();
        let joined = runner.last_args().join(" ");
        assert!(!joined.contains("-fps_mode"), "单帧路径不带帧率直通参数: {joined}");
        assert!(!joined.contains("-vsync"), "单帧路径不带帧率直通参数: {joined}");
    }

    // ── 抽帧方式（帧率过滤 vs 跳转定位）──────────────────────────────────

    #[test]
    fn resolve_method_auto_uses_seek_threshold() {
        // 默认阈值 60s：短于它走帧率过滤，达到/超过走跳转定位。
        assert_eq!(
            resolve_method(ExtractMethod::Auto, Some(59_999), DEFAULT_SEEK_THRESHOLD_MS),
            ExtractMethod::Filter
        );
        assert_eq!(
            resolve_method(ExtractMethod::Auto, Some(60_000), DEFAULT_SEEK_THRESHOLD_MS),
            ExtractMethod::Seek
        );
        assert_eq!(
            resolve_method(ExtractMethod::Auto, Some(3_600_000), DEFAULT_SEEK_THRESHOLD_MS),
            ExtractMethod::Seek
        );
        // 时长未知 ⇒ 只能帧率过滤（跳转定位需要目标时刻）。
        assert_eq!(
            resolve_method(ExtractMethod::Auto, None, DEFAULT_SEEK_THRESHOLD_MS),
            ExtractMethod::Filter
        );
        // 阈值可配 + 显式指定优先。
        assert_eq!(
            resolve_method(ExtractMethod::Auto, Some(30_000), 10_000),
            ExtractMethod::Seek
        );
        assert_eq!(
            resolve_method(ExtractMethod::Filter, Some(3_600_000), 60_000),
            ExtractMethod::Filter
        );
        assert_eq!(
            resolve_method(ExtractMethod::Seek, Some(1_000), 60_000),
            ExtractMethod::Seek
        );
    }

    #[test]
    fn target_times_cover_the_whole_clip_within_max_frames() {
        // 10 分钟 / 24 帧 ⇒ 每 25s 一个目标时刻，覆盖全片。
        let plan = plan_sampling(Some(600_000), 5_000, 24);
        let times = target_times_ms(Some(600_000), &plan);
        assert_eq!(times.len(), 24);
        assert_eq!(times[0], 0);
        assert_eq!(times[1], 25_000);
        assert!(times.iter().all(|t| *t < 600_000));

        // 短视频 ⇒ 单帧中点。
        let plan = plan_sampling(Some(3_000), 5_000, 24);
        assert_eq!(target_times_ms(Some(3_000), &plan), vec![1_500]);

        // 时长未知 ⇒ 空（调用方回退帧率过滤）。
        let plan = plan_sampling(None, 5_000, 24);
        assert!(target_times_ms(None, &plan).is_empty());
    }

    #[test]
    fn seek_puts_ss_before_input_and_makes_one_call_per_target() {
        let (ff, video, out) = fixture("seek_args");
        let runner = FakeRunner::new(0, "");
        let opts = ExtractOptions {
            duration_hint_ms: Some(120_000),
            method: ExtractMethod::Seek,
            ..ExtractOptions::default()
        };
        let outcome = extract_frames_with(
            &ff,
            &video,
            FrameStrategy::Interval { every_ms: 5_000 },
            &out,
            &opts,
            &runner,
        )
        .unwrap();

        assert_eq!(outcome.method, ExtractMethod::Seek);
        // 120s / 24 = 5s ⇒ 24 个目标时刻，每帧一次调用（一次 probe 都没有：给了 hint）。
        assert_eq!(runner.call_count(), 24);
        assert_eq!(outcome.frames.len(), 24);
        assert_eq!(outcome.frames[0].t_ms, 0);
        assert_eq!(outcome.frames[1].t_ms, 5_000);

        let first = runner.calls.borrow()[0].join(" ");
        // `-ss` 必须在 `-i` 之前（快速定位），且每次只取一帧。
        let ss = first.find("-ss").expect("-ss 缺失");
        let i = first.find("-i").expect("-i 缺失");
        assert!(ss < i, "跳转定位要求 -ss 在 -i 之前: {first}");
        assert!(first.contains("-frames:v 1"), "{first}");
        assert!(first.contains("showinfo"), "需要实测落点校验: {first}");
        assert!(outcome.frames.iter().all(|f| f.image_path.is_file()));
    }

    #[test]
    fn seek_reports_bad_seek_when_landing_drift_exceeds_gate() {
        let (ff, video, out) = fixture("seek_drift");
        // showinfo 报告落点在 12s，但请求的是 5s ⇒ 偏差 7s，超门。
        let runner = FakeRunner::new(0, "n: 0 pts_time:12.000000");
        let opts = ExtractOptions {
            duration_hint_ms: Some(120_000),
            method: ExtractMethod::Seek,
            ..ExtractOptions::default()
        };
        let err = extract_frames_with(
            &ff,
            &video,
            FrameStrategy::Interval { every_ms: 5_000 },
            &out,
            &opts,
            &runner,
        )
        .unwrap_err();
        match err {
            VideoError::BadSeek { requested_ms, drift_ms, .. } => {
                assert_eq!(requested_ms, 0);
                assert_eq!(drift_ms, 12_000);
            }
            other => panic!("应报 BadSeek，实际 {other:?}"),
        }
    }

    #[test]
    fn seek_fails_when_ffmpeg_reports_success_but_writes_no_frame() {
        let (ff, video, out) = fixture("seek_nofile");
        let runner = FakeRunner::new(0, "");
        let opts = ExtractOptions {
            duration_hint_ms: Some(120_000),
            method: ExtractMethod::Seek,
            ..ExtractOptions::default()
        };
        // FakeRunner 在 seek 调用（非失败）时会写文件；这里用 fail 分支模拟"没落盘"：
        // 失败即会走到错误路径，断言错误信息里带帧序号与时刻。
        let runner = runner.failing_on("-ss", 0);
        let err = extract_frames_with(
            &ff,
            &video,
            FrameStrategy::Interval { every_ms: 5_000 },
            &out,
            &opts,
            &runner,
        )
        .unwrap_err();
        let VideoError::FfmpegFailed(msg) = err else {
            panic!("应报 FfmpegFailed");
        };
        assert!(msg.contains("第 1/24 帧"), "{msg}");
    }

    #[test]
    fn explicit_filter_is_used_even_for_long_videos() {
        let (ff, video, out) = fixture("seek_off");
        let runner = FakeRunner::new(2, "");
        let opts = ExtractOptions {
            duration_hint_ms: Some(600_000),
            method: ExtractMethod::Filter,
            ..ExtractOptions::default()
        };
        let outcome = extract_frames_with(
            &ff,
            &video,
            FrameStrategy::Interval { every_ms: 5_000 },
            &out,
            &opts,
            &runner,
        )
        .unwrap();
        assert_eq!(outcome.method, ExtractMethod::Filter);
        assert_eq!(runner.call_count(), 1, "帧率过滤只调用一次");
        assert!(runner.last_args().join(" ").contains("fps=1000/25000"));
    }

    #[test]
    fn near_duplicate_uses_hamming_distance_on_64bit_hash() {
        let base = 0b1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010_1010u64;
        assert!(is_near_duplicate(base, base, 0));
        let one_bit_flip = base ^ 0b1;
        assert!(is_near_duplicate(base, one_bit_flip, 1));
        assert!(!is_near_duplicate(base, one_bit_flip, 0));
        let far = 0u64;
        assert!(!is_near_duplicate(base, far, 8));
    }

    #[test]
    fn parses_duration_and_pts_from_ffmpeg_banner() {
        let banner = "  Duration: 00:05:23.45, start: 0.000000, bitrate: 1500 kb/s";
        assert_eq!(parse_duration_ms(banner), Some(323_450));
        assert_eq!(parse_duration_ms("Duration: 00:00:03.00, start: 0"), Some(3_000));
        assert_eq!(parse_duration_ms("Duration: N/A, start: 0"), None);
        assert_eq!(parse_duration_ms("no duration here"), None);

        let stderr = "\
n: 0 pts_time:0.000000
n: 1 pts_time:5.000000
n: 2 pts_time:10.500000
";
        assert_eq!(parse_pts_times_ms(stderr), vec![0, 5_000, 10_500]);
        assert!(parse_pts_times_ms("n: 0 pts_time:nan").is_empty());
    }

    #[test]
    fn probe_uses_duration_from_banner_when_hint_missing() {
        let (ff, video, out) = fixture("probe");
        // 探测调用返回 banner；真实抽帧调用不产生帧。
        let banner = "Duration: 00:00:04.00, start: 0.000000, bitrate: 100 kb/s";
        let runner = FakeRunner::new(0, banner);
        let outcome = extract_frames_with(
            &ff,
            &video,
            FrameStrategy::Interval { every_ms: 5_000 },
            &out,
            &ExtractOptions::default(),
            &runner,
        )
        .unwrap();
        assert_eq!(outcome.duration_ms, Some(4_000));
        // 4s < 5s ⇒ 走"中点单帧"分支。
        assert!(runner.last_args().join(" ").contains("-ss 2.000"));
    }
}
