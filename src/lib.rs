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
}

impl Default for ExtractOptions {
    fn default() -> Self {
        Self {
            threads: 2,
            max_frames: 24,
            duration_hint_ms: None,
            max_edge: 1024,
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
    let args = build_args(video, out_dir, &plan, options);
    let out = runner.run(ffmpeg_bin, &args)?;
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

    let mut frames: Vec<VideoFrame> = produced
        .into_iter()
        .enumerate()
        .map(|(i, image_path)| VideoFrame {
            image_path,
            t_ms: frame_timestamp_ms(i, &plan, &pts_ms),
        })
        .collect();
    frames.sort_by_key(|f| f.t_ms);

    Ok(ExtractOutcome {
        frames,
        duration_ms,
        truncated,
    })
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
    args.push("-fps_mode".into());
    args.push("vfr".into());
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
        calls: std::cell::RefCell<Vec<Vec<String>>>,
    }

    impl FakeRunner {
        fn new(frames: usize, stderr: &str) -> Self {
            Self {
                frames,
                stderr: stderr.to_string(),
                calls: std::cell::RefCell::new(Vec::new()),
            }
        }

        fn last_args(&self) -> Vec<String> {
            self.calls.borrow().last().cloned().unwrap_or_default()
        }
    }

    impl ProcessRunner for FakeRunner {
        fn run(&self, _program: &Path, args: &[String]) -> Result<ProcessOutput, VideoError> {
            self.calls.borrow_mut().push(args.to_vec());
            // banner 探测（只有 -i，没有输出）时不产生帧。
            let is_probe = !args.iter().any(|a| a.ends_with(".png"));
            if !is_probe {
                let out_arg = args.last().unwrap().clone();
                let dir = Path::new(&out_arg).parent().unwrap().to_path_buf();
                for i in 0..self.frames {
                    let p = dir.join(format!("frame_{:06}.png", i));
                    fs::write(&p, b"fake").unwrap();
                }
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
