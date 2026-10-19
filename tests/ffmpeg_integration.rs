//! 27.4.2：真实 FFmpeg 集成测试（**默认忽略**，手动运行）。
//!
//! 与模型相关测试同一套路：需要本机环境提供二进制/素材，CI 与日常 `cargo test` 不执行。
//!
//! 环境变量：
//! - `FO_FFMPEG_BIN`：ffmpeg 可执行文件路径（未设置时退回 PATH 里的 `ffmpeg`）。
//! - `FO_TEST_VIDEO`：（可选）自备的小视频；未设置时用 ffmpeg 的 `lavfi` 现场合成 3 秒测试视频。
//!
//! 运行（PowerShell）：
//! ```powershell
//! $env:FO_FFMPEG_BIN = 'C:\ffmpeg\bin\ffmpeg.exe'
//! cargo test -p video_frames --test ffmpeg_integration -- --ignored --nocapture --test-threads=1
//! ```
//!
//! 运行（bash）：
//! ```bash
//! FO_FFMPEG_BIN=/usr/bin/ffmpeg cargo test -p video_frames --test ffmpeg_integration -- --ignored --nocapture --test-threads=1
//! ```

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use video_frames::{extract_frames, ExtractOptions, FrameStrategy, FpsModePolicy};

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
    let status = Command::new(ffmpeg_bin())
        .args([
            "-hide_banner",
            "-nostdin",
            "-y",
            // 纯色 + 移动方块：画面有变化，便于抽帧后有可比较的帧。
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=3:size=320x240:rate=10",
            "-pix_fmt",
            "yuv420p",
        ])
        .arg(target)
        .status()
        .expect("启动 ffmpeg（合成测试视频）失败：请检查 FO_FFMPEG_BIN");
    status.success()
}

fn is_png(path: &Path) -> bool {
    let Ok(bytes) = fs::read(path) else {
        return false;
    };
    bytes.len() > 8 && bytes[..8] == [0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a]
}

/// 在给定目录（可含中文与空格）里完整跑一遍抽帧，并校验帧文件与时间戳。
fn run_case(root: &Path, video_name: &str) -> usize {
    let day_dir = root.join("素材 测试 dir"); // 中文 + 空格 + 大小写混排
    fs::create_dir_all(&day_dir).unwrap();
    let video = day_dir.join(video_name);
    assert!(test_video(&video), "合成测试视频失败（ffmpeg 是否可用？）");
    assert!(video.is_file());

    let out = day_dir.join("帧 out");
    let opts = ExtractOptions {
        threads: 2,
        max_frames: 24,
        // 3 秒视频：给 5 秒间隔会走"中点单帧"分支，这里给 1 秒拿多帧。
        duration_hint_ms: None, // 故意不传，验证 probe 兜底
        max_edge: 512,
        fps_mode: FpsModePolicy::Auto, // 版本未知 ⇒ 失败回退路径
    };
    let outcome = extract_frames(
        &ffmpeg_bin(),
        &video,
        FrameStrategy::Interval { every_ms: 1_000 },
        &out,
        &opts,
    )
    .expect("抽帧失败");

    assert!(
        outcome.duration_ms.is_some_and(|d| d >= 2_000),
        "probe 应解析出约 3 秒的时长，实际 {:?}",
        outcome.duration_ms
    );
    assert!(
        (2..=4).contains(&outcome.frames.len()),
        "3 秒 / 1 秒间隔应得到 3 帧左右，实际 {}",
        outcome.frames.len()
    );
    let mut last = None;
    for f in &outcome.frames {
        assert!(f.image_path.is_file(), "帧文件不存在: {:?}", f.image_path);
        assert!(is_png(&f.image_path), "帧不是合法 PNG: {:?}", f.image_path);
        if let Some(prev) = last {
            assert!(f.t_ms >= prev, "时间戳必须单调递增: {prev} → {}", f.t_ms);
        }
        last = Some(f.t_ms);
    }
    outcome.frames.len()
}

#[test]
#[ignore = "需要本机 ffmpeg：设置 FO_FFMPEG_BIN 后手动运行"]
fn extracts_frames_from_ascii_path() {
    let root = temp_root("ascii");
    let n = run_case(&root.join("plain"), "clip.mp4");
    println!("[ascii] frames = {n}");
    assert!(n >= 2);
}

#[test]
#[ignore = "需要本机 ffmpeg：设置 FO_FFMPEG_BIN 后手动运行"]
fn extracts_frames_from_chinese_and_space_path() {
    let root = temp_root("cjk");
    let n = run_case(&root.join("普通 路径"), "样片 测试.mp4");
    println!("[cjk+space] frames = {n}");
    assert!(n >= 2);
}
