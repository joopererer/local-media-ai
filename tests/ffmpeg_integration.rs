//! 27.4.2：真实 FFmpeg 集成测试（**默认忽略**，手动运行）。
//!
//! 与模型相关测试同一套路：需要本机环境提供二进制/素材，CI 与日常 `cargo test` 不执行。
//!
//! 环境变量：
//! - `FO_FFMPEG_BIN`：ffmpeg 可执行文件路径（未设置时退回 PATH 里的 `ffmpeg`）。
//! - `FO_TEST_VIDEO`：（可选）自备视频；未设置时用 ffmpeg 的 `lavfi` 现场合成 3 秒测试视频。
//!
//! 运行（PowerShell）：
//! ```powershell
//! $env:FO_FFMPEG_BIN = 'D:\ffmpeg\bin\ffmpeg.exe'
//! $env:FO_TEST_VIDEO = 'D:\test\素材 测试.mp4'   # 可选：换成你的长视频
//! cargo test -p video_frames --test ffmpeg_integration -- --ignored --nocapture --test-threads=1
//! ```
//!
//! 本文件同时输出**两种抽帧方式**（帧率过滤 / 跳转定位）在同一素材上的耗时与时刻差异，
//! 用于验证默认选择策略（时长 ≥ 60s 走跳转定位）。

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use video_frames::{
    extract_frames, plan_sampling, target_times_ms, ExtractMethod, ExtractOptions,
    FrameStrategy, FpsModePolicy,
};

/// 抽帧间隔（毫秒）。真实长视频会由均匀重采样放大到覆盖全片。
const EVERY_MS: u64 = 1_000;
const MAX_FRAMES: u32 = 24;

fn ffmpeg_bin() -> PathBuf {
    std::env::var("FO_FFMPEG_BIN")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("ffmpeg"))
}

fn temp_root(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "vf_it_{}_{}_{}",
        tag,
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::create_dir_all(&dir).unwrap();
    dir
}

/// 取得测试视频：优先 `FO_TEST_VIDEO`，否则用 lavfi 合成 3 秒 320x240@10fps 的片子。
fn test_video(target: &Path) -> bool {
    if let Ok(custom) = std::env::var("FO_TEST_VIDEO") {
        fs::copy(&custom, target).unwrap_or_else(|e| panic!("复制 FO_TEST_VIDEO 失败: {e}"));
        return true;
    }
    let out = Command::new(ffmpeg_bin())
        .args([
            "-hide_banner",
            "-nostdin",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=3:size=320x240:rate=10",
            "-pix_fmt",
            "yuv420p",
        ])
        .arg(target)
        .output()
        .expect("启动 ffmpeg（合成测试视频）失败：请检查 FO_FFMPEG_BIN");
    if !out.status.success() {
        println!(
            "合成测试视频失败：{}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
    out.status.success()
}

fn is_png(path: &Path) -> bool {
    let Ok(bytes) = fs::read(path) else {
        return false;
    };
    bytes.len() > 8 && bytes[..8] == [0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a]
}

/// 期望帧数：按**探测到的时长**算（不写死），允许 ±1（均匀重采样取整/尾帧差异）。
fn expected_frames(duration_ms: u64) -> usize {
    let plan = plan_sampling(Some(duration_ms), EVERY_MS, MAX_FRAMES);
    let n = target_times_ms(Some(duration_ms), &plan).len();
    n.max(1)
}

struct CaseResult {
    frames: usize,
    duration_ms: u64,
    elapsed_ms: u128,
    times: Vec<u64>,
    method: ExtractMethod,
}

/// 在给定目录（可含中文与空格）里跑一遍指定方式的抽帧，并做完整断言。
fn run_case(root: &Path, video_name: &str, method: ExtractMethod) -> CaseResult {
    let day_dir = root.join("素材 测试 dir"); // 中文 + 空格 + 大小写混排
    fs::create_dir_all(&day_dir).unwrap();
    let video = day_dir.join(video_name);
    assert!(test_video(&video), "准备测试视频失败（ffmpeg 是否可用？）");
    assert!(video.is_file());

    let out = day_dir.join(format!("帧 out {method:?}"));
    let opts = ExtractOptions {
        threads: 2,
        max_frames: MAX_FRAMES,
        duration_hint_ms: None, // 故意不传：验证 probe 兜底
        max_edge: 512,
        fps_mode: FpsModePolicy::Auto, // 版本未知 ⇒ 走回退逻辑
        method,
        ..ExtractOptions::default()
    };

    let started = Instant::now();
    let outcome = extract_frames(
        &ffmpeg_bin(),
        &video,
        FrameStrategy::Interval { every_ms: EVERY_MS },
        &out,
        &opts,
    )
    .expect("抽帧失败");
    let elapsed = started.elapsed();

    let duration_ms = outcome
        .duration_ms
        .unwrap_or_else(|| panic!("{method:?}: probe 未解析出时长"));
    let expected = expected_frames(duration_ms);
    let actual = outcome.frames.len();

    println!(
        "[{method:?}] 时长 {} ms ｜ 帧数 {actual}（期望 {expected} ±1）｜ 耗时 {} ms ｜ 实际方式 {:?}",
        duration_ms,
        elapsed.as_millis(),
        outcome.method
    );
    println!(
        "[{method:?}] 时间戳(ms): {:?}",
        outcome.frames.iter().map(|f| f.t_ms).collect::<Vec<_>>()
    );

    assert!(
        actual.abs_diff(expected) <= 1,
        "{method:?}: 帧数应为 min(上限, ceil(时长/间隔)) ±1 = {expected}，实际 {actual}"
    );
    assert!(actual >= 1, "{method:?}: 至少应抽到一帧");

    let mut last = None;
    for f in &outcome.frames {
        assert!(f.image_path.is_file(), "帧文件不存在: {:?}", f.image_path);
        assert!(is_png(&f.image_path), "帧不是合法 PNG: {:?}", f.image_path);
        if let Some(prev) = last {
            assert!(f.t_ms >= prev, "时间戳必须单调递增: {prev} → {}", f.t_ms);
        }
        last = Some(f.t_ms);
    }

    CaseResult {
        frames: actual,
        duration_ms,
        elapsed_ms: elapsed.as_millis(),
        times: outcome.frames.iter().map(|f| f.t_ms).collect(),
        method: outcome.method,
    }
}

#[test]
#[ignore = "需要本机 ffmpeg：设置 FO_FFMPEG_BIN 后手动运行"]
fn extracts_frames_from_ascii_path() {
    let root = temp_root("ascii");
    // 默认策略（Auto）：长视频应自动选跳转定位。
    let auto = run_case(&root.join("plain"), "clip.mp4", ExtractMethod::Auto);
    println!(
        "[ascii] 自动选择 = {:?}，{} 帧，耗时 {} ms",
        auto.method, auto.frames, auto.elapsed_ms
    );
    if auto.duration_ms >= 60_000 {
        assert_eq!(
            auto.method,
            ExtractMethod::Seek,
            "长视频应自动使用跳转定位（否则每次都要解码全片）"
        );
    }
}

#[test]
#[ignore = "需要本机 ffmpeg：设置 FO_FFMPEG_BIN 后手动运行"]
fn extracts_frames_from_chinese_and_space_path() {
    let root = temp_root("cjk");
    let dir = root.join("普通 路径");

    // 同一素材上对比两种方式：耗时与时刻一致性（验证默认选择策略的依据）。
    let filter = run_case(&dir, "样片 测试.mp4", ExtractMethod::Filter);
    let seek = run_case(&dir, "样片 测试.mp4", ExtractMethod::Seek);

    assert_eq!(
        filter.duration_ms, seek.duration_ms,
        "同一视频两种方式探到的时长必须一致"
    );
    assert!(
        (filter.frames as i64 - seek.frames as i64).abs() <= 1,
        "两种方式帧数应基本一致：filter={} seek={}",
        filter.frames,
        seek.frames
    );

    // 时刻比对：过滤方式来自实测 pts，跳转方式是请求时刻；统计最大偏差。
    let mut max_drift = 0u64;
    for (a, b) in filter.times.iter().zip(seek.times.iter()) {
        max_drift = max_drift.max(a.abs_diff(*b));
    }
    let speedup = filter.elapsed_ms as f64 / seek.elapsed_ms.max(1) as f64;
    println!(
        "[对比] filter {} ms vs seek {} ms ⇒ 耗时比 {speedup:.2}×；时刻最大偏差 {max_drift} ms",
        filter.elapsed_ms, seek.elapsed_ms
    );
    println!(
        "[对比] 选择策略：时长 ≥ {}ms 且未显式指定 ⇒ 跳转定位（每帧一个进程，约 50ms 固定开销 × 帧数）",
        video_frames::DEFAULT_SEEK_THRESHOLD_MS
    );

    assert!(
        max_drift <= video_frames::MAX_SEEK_DRIFT_MS,
        "两种方式的帧时刻偏差应在 {}ms 内，实际 {max_drift} ms",
        video_frames::MAX_SEEK_DRIFT_MS
    );

    // 耗时不设"seek 必须更快"的硬断言：跳转定位每帧起一个进程（约 50ms/次、24 帧 ⇒
    // ~1.2s 固定开销），对**易解码**的合成素材反而可能更慢（实测 180s 720p testsrc：
    // filter 1.39s vs seek 1.99s），而对真实 1080p 长片（单次全片解码 ~20s+）快一个量级。
    // 这里只做一个宽松的健全性上界，防止实现退化（例如误把解码写成全片两遍）。
    let hard_ceiling = filter.elapsed_ms * 3 + 3_000;
    assert!(
        seek.elapsed_ms <= hard_ceiling,
        "跳转定位耗时异常（seek={}ms, filter={}ms，上限 {hard_ceiling}ms）",
        seek.elapsed_ms,
        filter.elapsed_ms
    );
}
