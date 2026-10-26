//! 视频片段抽帧测试探针（支持本地路径与在线 HTTP/HTTPS/m3u8 视频流）。
//!
//! 用法：
//! ```powershell
//! cargo run -p video_frames --example stream_clip_extract -- <VIDEO_PATH_OR_URL> [START_SEC] [END_SEC] [STRATEGY]
//! ```

use std::env;
use std::path::{Path, PathBuf};
use std::time::Instant;

use video_frames::{extract_clip_frames, ClipRange, ExtractMethod, ExtractOptions, FrameStrategy};

fn find_ffmpeg() -> PathBuf {
    if let Ok(p) = env::var("FO_FFMPEG_BIN") {
        let pb = PathBuf::from(p);
        if pb.is_file() {
            return pb;
        }
    }
    let candidates = [
        r"D:\ffmpeg\bin\ffmpeg.exe",
        r"C:\ffmpeg\bin\ffmpeg.exe",
        "ffmpeg.exe",
        "ffmpeg",
    ];
    for c in candidates {
        let pb = PathBuf::from(c);
        if pb.is_file() {
            return pb;
        }
    }
    PathBuf::from("ffmpeg")
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!(
            "用法: stream_clip_extract <URL_或本地视频路径> [起始秒] [结束秒] [interval|scene]"
        );
        eprintln!("示例: cargo run -p video_frames --example stream_clip_extract -- \"D:\\sample.mp4\" 5 15 interval");
        return;
    }

    let input_path = &args[1];
    let start_sec: f64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let end_sec: f64 = args
        .get(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(start_sec + 10.0);
    let strat_str = args.get(4).map(|s| s.as_str()).unwrap_or("interval");

    let ffmpeg_bin = find_ffmpeg();
    println!("=== 视频片段区间抽帧探针 ===");
    println!("FFmpeg 路径: {}", ffmpeg_bin.display());
    println!("输入资源: {input_path}");
    println!("目标区间: {start_sec:.2}s ~ {end_sec:.2}s");

    let start_ms = (start_sec * 1000.0) as u64;
    let end_ms = (end_sec * 1000.0) as u64;
    let range = ClipRange::new(start_ms, end_ms);

    let strategy = match strat_str {
        "scene" => {
            println!("策略: 场景变化感知 (Scene, 阈值 0.4)");
            FrameStrategy::Scene { threshold: 0.4 }
        }
        _ => {
            println!("策略: 固定间隔 (Interval, 2000ms)");
            FrameStrategy::Interval { every_ms: 2000 }
        }
    };

    let temp_dir = env::temp_dir().join(format!("clip_probe_{}", std::process::id()));
    let opts = ExtractOptions {
        threads: 2,
        max_frames: 12,
        max_edge: 768,
        method: ExtractMethod::Auto,
        ..ExtractOptions::default()
    };

    let t0 = Instant::now();
    let res = extract_clip_frames(
        &ffmpeg_bin,
        Path::new(input_path),
        range,
        strategy,
        &temp_dir,
        &opts,
    );
    let elapsed = t0.elapsed();

    match res {
        Ok(outcome) => {
            println!("\n✅ 抽帧成功！总耗时: {:.2}s", elapsed.as_secs_f64());
            println!("抽帧方式: {:?}", outcome.method);
            println!("提取帧数: {}", outcome.frames.len());
            for (i, frame) in outcome.frames.iter().enumerate() {
                let size_kb = std::fs::metadata(&frame.image_path)
                    .map(|m| m.len() as f64 / 1024.0)
                    .unwrap_or(0.0);
                println!(
                    "  [帧 #{:02}] 时刻: {:02}:{:05.2} ({}ms) | 大小: {:.1} KB | 路径: {}",
                    i + 1,
                    frame.t_ms / 60000,
                    (frame.t_ms % 60000) as f64 / 1000.0,
                    frame.t_ms,
                    size_kb,
                    frame.image_path.display()
                );
            }
            println!("\n临时目录: {}", temp_dir.display());
        }
        Err(e) => {
            eprintln!("\n❌ 抽帧失败: {e}");
        }
    }
}
