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
//! $env:FO_TEST_VIDEO = '$HOME/videos/sample.mp4'   # 可选：换成你的长视频
//! cargo test -p video_frames --test ffmpeg_integration -- --ignored --nocapture --test-threads=1
//! ```
//!
//! 本文件同时输出**两种抽帧方式**（帧率过滤 / 跳转定位）在同一素材上的耗时与时刻差异，
//! 用于验证默认选择策略（时长 ≥ 60s 走跳转定位）。

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use sha2::{Digest, Sha256};
use video_frames::{
    extract_frames, plan_sampling, target_times_ms, ExtractMethod, ExtractOptions,
    FrameStrategy, FpsModePolicy,
};

fn sha256_of(p: &Path) -> String {
    let mut f = fs::File::open(p).expect("open png");
    let mut h = Sha256::new();
    let mut buf = [0u8; 8192];
    loop {
        let n = f.read(&mut buf).unwrap();
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    format!("{:x}", h.finalize())
}

fn path_arg(p: &Path) -> String {
    p.to_string_lossy().into_owned()
}

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

/// 期望帧数：与 ffmpeg `fps=1/EVERY_MS` 在实际时长上产出的帧数一致，且不超过 MAX_FRAMES。
/// 例：3000ms + every=1000ms ⇒ 3 帧；600000ms ⇒ 600 帧但截到 MAX_FRAMES=24。
fn expected_frames(duration_ms: u64) -> usize {
    let n = ((duration_ms as f64) / (EVERY_MS as f64)).ceil() as usize;
    n.min(MAX_FRAMES as usize).max(1)
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
        actual.abs_diff(expected) <= 2,
        "{method:?}: 帧数应在期望 {expected} ±2 范围内（-ss 偏移可能多/少一帧），实际 {actual}"
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

    let plan = plan_sampling(Some(seek.duration_ms), EVERY_MS, MAX_FRAMES);
    let targets = target_times_ms(Some(seek.duration_ms), &plan);

    // 27.4 时刻语义复核：
    // - Seek 路径：t_ms 是代码算的目标时刻（即 target_times_ms 的 mid-point），
    //   偏差仅来自 ffmpeg 落点（应 ≤ 半间隔）。
    // - Filter 路径：t_ms 来自 showinfo 的**实测** pts（扣 start），输入偏移 half_interval
    //   使首帧避开 0 秒；t_ms 相对中点目标的偏差应 ≤ 半间隔（与 seek 一致）。
    let calc_max_dev = |times: &[u64], targets: &[u64]| -> u64 {
        times.iter().zip(targets.iter())
            .map(|(t, tgt)| t.abs_diff(*tgt))
            .max()
            .unwrap_or(0)
    };

    let seek_max_dev = calc_max_dev(&seek.times, &targets);
    assert!(
        seek_max_dev <= plan.every_ms / 2,
        "跳转定位最大偏差 {seek_max_dev}ms > 半采样间隔 {}ms",
        plan.every_ms / 2
    );

    let filter_max_dev = calc_max_dev(&filter.times, &targets);
    assert!(
        filter_max_dev <= plan.every_ms / 2,
        "filter 路径最大偏差 {filter_max_dev}ms > 半采样间隔 {}ms",
        plan.every_ms / 2
    );

    // 耗时输出（不设硬断言，详见 extract_by_seek 的实测记录）。
    let between_max_dev = seek.times.iter().zip(filter.times.iter())
        .map(|(a, b)| a.abs_diff(*b))
        .max()
        .unwrap_or(0);
    println!("━━━ 时刻对比（27.4 复核后：filter 加输入偏移）━━━");
    println!("seek vs mid ± half：max {seek_max_dev}ms（应 ≤ 半间隔）");
    println!("filter vs mid ± half：max {filter_max_dev}ms（应 ≤ 半间隔）");
    println!("方式间偏差：max {between_max_dev}ms（仅参考）");
    println!("━━━ 性能对比 ━━━");
    println!("跳转定位 {}ms vs 帧率过滤 {}ms", seek.elapsed_ms, filter.elapsed_ms);

    // 跳转定位耗时不退化到 read-the-whole-file 的程度。
    let hard_ceiling = filter.elapsed_ms * 3 + 3_000;
    assert!(
        seek.elapsed_ms <= hard_ceiling,
        "跳转定位耗时异常（seek={}ms, filter={}ms，上限 {hard_ceiling}ms）",
        seek.elapsed_ms,
        filter.elapsed_ms
    );
}

/// 27.4.5 验收：抽出的帧画面里的**实测时间戳**与 t_ms 必须一致。
///
/// 实现方式：合成视频时用 `drawtext` 把当前帧的**真实 PTS** 烧录到画面上。
/// 抽帧后对每个帧按其 t_ms 用精确 `-i ... -ss t_ms -frames:v 1` seek 一帧，
/// 比较两张 PNG 的 SHA256——若 filter 的 t_ms 与画面所在真实时刻一致，
/// 这两张图必然字节相同（**绕开 OCR**，用 ffmpeg 自身 seek 做"基线"）。
///
/// 关键：filter 路径不再使用代码目标时刻，t_ms = ffmpeg showinfo 实测 pts − start。
/// 若实现退化（如旧实现按 i×every 标记 t_ms 而真实画面在中点），
/// seek 的基线与 filter 抽到的图字节比对将失败。
#[test]
#[ignore = "需要本机 ffmpeg：设置 FO_FFMPEG_BIN 后手动运行"]
fn filter_timestamp_matches_pixel_identity() {
    let root = temp_root("burn");
    let dir = root.join("时间码 验证");
    fs::create_dir_all(&dir).unwrap();
    let video = dir.join("时间码源.mp4");

    // 6 秒 / 10 fps / 320x240，drawtext 烧录真实 PTS（HH:MM:SS.mmm）。
    let synth = Command::new(ffmpeg_bin())
        .args([
            "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-f", "lavfi",
            "-i", "testsrc=duration=6:size=320x240:rate=10",
            "-vf",
            "drawtext=fontfile='C\\:/Windows/Fonts/consola.ttf'\
             :text='%{pts\\:hms}'\
             :fontcolor=white:fontsize=18:x=2:y=2\
             :box=1:boxcolor=black@0.7:boxborderw=6",
            "-pix_fmt", "yuv420p",
        ])
        .arg(&video)
        .output()
        .expect("ffmpeg 合成失败");
    assert!(synth.status.success(), "testsrc + drawtext 合成失败");

    let out_dir = dir.join("filter_out");
    fs::create_dir_all(&out_dir).unwrap();
    let opts = ExtractOptions {
        threads: 2,
        max_frames: MAX_FRAMES,
        duration_hint_ms: Some(6_000),
        max_edge: 0,
        fps_mode: FpsModePolicy::Auto,
        method: ExtractMethod::Filter,
        ..ExtractOptions::default()
    };
    let outcome = extract_frames(
        &ffmpeg_bin(), &video,
        FrameStrategy::Interval { every_ms: EVERY_MS },
        &out_dir, &opts,
    )
    .expect("filter 抽取失败");

    assert!(!outcome.frames.is_empty(), "filter 路径至少应抽到一帧");

    let mut matched = 0;
    let total = outcome.frames.len();
    for (i, f) in outcome.frames.iter().enumerate() {
        let ref_png = dir.join(format!("ref_{i:03}.png"));
        let ts = format!("{:.3}", f.t_ms as f64 / 1000.0);
        let video_str = path_arg(&video);
        let r = Command::new(ffmpeg_bin())
            .args([
                "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                "-i", &video_str,
                "-ss", &ts,
                "-frames:v", "1",
            ])
            .arg(&ref_png)
            .output()
            .expect("seek 抽取失败");
        assert!(r.status.success(), "seek 抽取失败 t_ms={}", f.t_ms);

        let a = sha256_of(&f.image_path);
        let b = sha256_of(&ref_png);
        let ok = a == b;
        if ok { matched += 1; }
        println!(
            "frame[{i}] t_ms={:>5} | filter={} | seek@-t_ms={} | {}",
            f.t_ms, &a[..12], &b[..12],
            if ok { "MATCH ✅" } else { "DIFFER ❌" }
        );
    }
    assert_eq!(
        matched, total,
        "filter t_ms 与按 t_ms 精确 seek 的画面不完全一致：{matched}/{total}"
    );
}
