//! Unified CLI entrypoint for local-media-ai sidecar.
//!
//! Provides JSON-first subcommands for Flutter and client integrations:
//! - doctor: Environment checks (FFmpeg, ORT, models)
//! - extract-clip: Keyframe extraction from video ranges (local or streaming)
//! - embed-text: Text embedding via EmbeddingGemma 2
//! - embed-frames: Batch image / frame embedding
//! - caption: VLM video clip narration and structured extraction

use std::path::{Path, PathBuf};
use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};

#[derive(Parser)]
#[command(name = "local-media-ai")]
#[command(about = "Local Multimodal AI CLI Sidecar for video clips and media management")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Environment diagnostics (FFmpeg, ORT availability)
    Doctor {
        /// Optional custom FFmpeg path
        #[arg(long)]
        ffmpeg: Option<PathBuf>,
    },

    /// Extract video keyframes within a timestamp range (supports local files and HTTP/m3u8 URLs)
    ExtractClip {
        /// Local video file path or HTTP/HTTPS/m3u8 streaming URL
        #[arg(long)]
        video: String,

        /// Start timestamp in milliseconds
        #[arg(long, default_value_t = 0)]
        start_ms: u64,

        /// End timestamp in milliseconds
        #[arg(long)]
        end_ms: u64,

        /// Extraction strategy: 'interval' or 'scene'
        #[arg(long, default_value = "interval")]
        strategy: String,

        /// Sampling interval in milliseconds for interval strategy
        #[arg(long, default_value_t = 2000)]
        every_ms: u64,

        /// Scene change detection threshold (0.0 .. 1.0) for scene strategy
        #[arg(long, default_value_t = 0.4)]
        threshold: f32,

        /// Output directory for extracted frame images
        #[arg(long)]
        out_dir: PathBuf,

        /// Maximum frames to extract
        #[arg(long, default_value_t = 12)]
        max_frames: u32,

        /// Maximum frame image edge size in pixels
        #[arg(long, default_value_t = 768)]
        max_edge: u32,

        /// Optional custom FFmpeg binary path
        #[arg(long)]
        ffmpeg: Option<PathBuf>,
    },

    /// Generate 768-d embedding for text query
    EmbedText {
        /// Query text
        #[arg(long)]
        text: String,

        /// Optional instruction prefix
        #[arg(long, default_value = "")]
        prefix: String,

        /// Model directory containing onnx weights and tokenizer.json
        #[arg(long)]
        model_dir: PathBuf,

        /// ONNX Runtime dynamic library path (onnxruntime.dll)
        #[arg(long)]
        ort_lib: Option<PathBuf>,

        /// CPU inference threads
        #[arg(long, default_value_t = 4)]
        threads: usize,
    },

    /// Generate embeddings for video frames
    EmbedFrames {
        /// JSON array of frames: [{"t_ms": 2000, "path": "C:\\temp\\f1.png"}, ...]
        #[arg(long)]
        frames_json: String,

        /// Model directory
        #[arg(long)]
        model_dir: PathBuf,

        /// ONNX Runtime dynamic library path
        #[arg(long)]
        ort_lib: Option<PathBuf>,

        /// Soft tokens budget (default 140)
        #[arg(long, default_value_t = 140)]
        soft_tokens: u32,

        /// CPU inference threads
        #[arg(long, default_value_t = 4)]
        threads: usize,
    },

    /// Generate structured caption and summary for keyframes
    Caption {
        /// Comma-separated frame image paths
        #[arg(long)]
        frames: String,

        /// Custom focus prompt or instruction
        #[arg(long)]
        instruction: Option<String>,

        /// Simulated raw response for mock/testing
        #[arg(long)]
        mock_response: Option<String>,
    },
}

#[derive(Serialize)]
struct OutputEnvelope<T: Serialize> {
    ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    data: Option<T>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

impl<T: Serialize> OutputEnvelope<T> {
    fn success(data: T) -> Self {
        Self {
            ok: true,
            data: Some(data),
            error: None,
        }
    }

    fn failure(msg: impl Into<String>) -> Self {
        Self {
            ok: false,
            data: None,
            error: Some(msg.into()),
        }
    }

    fn print(&self) {
        println!("{}", serde_json::to_string_pretty(self).unwrap());
    }
}

fn resolve_ffmpeg(custom: Option<PathBuf>) -> PathBuf {
    if let Some(p) = custom {
        if p.is_file() {
            return p;
        }
    }
    if let Ok(p) = std::env::var("FO_FFMPEG_BIN") {
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

#[derive(Serialize)]
struct DoctorResult {
    ffmpeg_found: bool,
    ffmpeg_path: String,
    ort_lib_default: String,
}

#[derive(Serialize)]
struct ExtractClipResult {
    method: String,
    frames: Vec<ExtractedFrameInfo>,
    duration_ms: Option<u64>,
}

#[derive(Serialize)]
struct ExtractedFrameInfo {
    t_ms: u64,
    path: String,
    size_bytes: u64,
}

#[derive(Serialize)]
struct EmbedTextResult {
    dimensions: usize,
    vector: Vec<f32>,
}

#[derive(Deserialize)]
struct InputFrame {
    t_ms: u64,
    path: String,
}

#[derive(Serialize)]
struct EmbedFramesResult {
    embeddings: Vec<FrameVectorInfo>,
}

#[derive(Serialize)]
struct FrameVectorInfo {
    t_ms: u64,
    vector: Vec<f32>,
    elapsed_secs: f64,
}

#[derive(Serialize)]
struct CaptionResult {
    scene: String,
    main_subject: String,
    key_actions: Vec<String>,
    summary: String,
    search_document: String,
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Doctor { ffmpeg } => {
            let ff = resolve_ffmpeg(ffmpeg);
            let found = ff.is_file();
            let data = DoctorResult {
                ffmpeg_found: found,
                ffmpeg_path: ff.display().to_string(),
                ort_lib_default: embed_core::ort_library_file_name().to_string(),
            };
            OutputEnvelope::success(data).print();
        }

        Commands::ExtractClip {
            video,
            start_ms,
            end_ms,
            strategy,
            every_ms,
            threshold,
            out_dir,
            max_frames,
            max_edge,
            ffmpeg,
        } => {
            let ff = resolve_ffmpeg(ffmpeg);
            let strat = match strategy.to_lowercase().as_str() {
                "scene" => video_frames::FrameStrategy::Scene { threshold },
                _ => video_frames::FrameStrategy::Interval { every_ms },
            };
            let range = video_frames::ClipRange::new(start_ms, end_ms);
            let opts = video_frames::ExtractOptions {
                max_frames,
                max_edge,
                clip_range: Some(range),
                ..video_frames::ExtractOptions::default()
            };

            match video_frames::extract_clip_frames(&ff, Path::new(&video), range, strat, &out_dir, &opts) {
                Ok(outcome) => {
                    let frames = outcome.frames.into_iter().map(|f| {
                        let size_bytes = std::fs::metadata(&f.image_path).map(|m| m.len()).unwrap_or(0);
                        ExtractedFrameInfo {
                            t_ms: f.t_ms,
                            path: f.image_path.display().to_string(),
                            size_bytes,
                        }
                    }).collect();
                    OutputEnvelope::success(ExtractClipResult {
                        method: format!("{:?}", outcome.method),
                        frames,
                        duration_ms: outcome.duration_ms,
                    }).print();
                }
                Err(e) => {
                    OutputEnvelope::<ExtractClipResult>::failure(e.to_string()).print();
                    std::process::exit(1);
                }
            }
        }

        Commands::EmbedText {
            text,
            prefix,
            model_dir,
            ort_lib,
            threads,
        } => {
            if let Some(p) = ort_lib {
                if let Err(e) = embed_core::init_ort(&p) {
                    OutputEnvelope::<EmbedTextResult>::failure(format!("初始化 ORT 失败: {e}")).print();
                    std::process::exit(1);
                }
            }
            match embed_core::Embedder::new(&model_dir, threads) {
                Ok(embedder) => match embedder.embed(&text, &prefix) {
                    Ok(vec) => {
                        OutputEnvelope::success(EmbedTextResult {
                            dimensions: vec.len(),
                            vector: vec,
                        }).print();
                    }
                    Err(e) => {
                        OutputEnvelope::<EmbedTextResult>::failure(e.to_string()).print();
                        std::process::exit(1);
                    }
                },
                Err(e) => {
                    OutputEnvelope::<EmbedTextResult>::failure(e.to_string()).print();
                    std::process::exit(1);
                }
            }
        }

        Commands::EmbedFrames {
            frames_json,
            model_dir,
            ort_lib,
            soft_tokens,
            threads,
        } => {
            if let Some(p) = ort_lib {
                if let Err(e) = embed_core::init_ort(&p) {
                    OutputEnvelope::<EmbedFramesResult>::failure(format!("初始化 ORT 失败: {e}")).print();
                    std::process::exit(1);
                }
            }
            let input_frames: Vec<InputFrame> = match serde_json::from_str(&frames_json) {
                Ok(f) => f,
                Err(e) => {
                    OutputEnvelope::<EmbedFramesResult>::failure(format!("解析 frames_json 失败: {e}")).print();
                    std::process::exit(1);
                }
            };

            match embed_core::Embedder::new(&model_dir, threads) {
                Ok(embedder) => {
                    let mut embeddings = Vec::with_capacity(input_frames.len());
                    for f in &input_frames {
                        match embedder.embed_image_file(Path::new(&f.path), soft_tokens, false) {
                            Ok(res) => {
                                embeddings.push(FrameVectorInfo {
                                    t_ms: f.t_ms,
                                    vector: res.vector,
                                    elapsed_secs: res.prep_secs + res.vision_secs + res.backbone_secs,
                                });
                            }
                            Err(e) => {
                                OutputEnvelope::<EmbedFramesResult>::failure(format!("帧 {} 嵌入失败: {e}", f.path)).print();
                                std::process::exit(1);
                            }
                        }
                    }
                    OutputEnvelope::success(EmbedFramesResult { embeddings }).print();
                }
                Err(e) => {
                    OutputEnvelope::<EmbedFramesResult>::failure(e.to_string()).print();
                    std::process::exit(1);
                }
            }
        }

        Commands::Caption {
            frames,
            instruction,
            mock_response,
        } => {
            let paths: Vec<&str> = frames.split(',').map(|s| s.trim()).filter(|s| !s.is_empty()).collect();
            let prompt = caption_core::PromptBuilder::build_user_prompt(paths.len(), instruction.as_deref());

            let raw = match mock_response {
                Some(m) => m,
                None => {
                    OutputEnvelope::<CaptionResult>::failure(
                        "No VLM inference backend configured for CLI. Please configure Ollama (http://localhost:11434/v1) or an OpenAI/DashScope compatible endpoint in settings."
                    ).print();
                    std::process::exit(1);
                }
            };

            let parsed = caption_core::ClipCaption::parse_robust(&raw);
            let search_document = parsed.to_search_document();
            OutputEnvelope::success(CaptionResult {
                scene: parsed.scene,
                main_subject: parsed.main_subject,
                key_actions: parsed.key_actions,
                summary: parsed.summary,
                search_document,
            }).print();
        }
    }
}
