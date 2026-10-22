//! 27.5: 嵌入路由与 llama 工作进程监管（Rust 侧，Dart 经现有 `embedding_*` 接口调用）。
//!
//! - 默认走进程内 ORT（行为不变）；实验开关切到 `llama-worker` 后走 §2 路由。
//! - 同 runtime 查询规则（方案 §3）：查询向量与资料库向量必须同 runtime；
//!   新后端覆盖率达阈值前查询继续走原后端（Dart 侧经 `embedding_current_runtime()`
//!   决定读哪个 runtime 的向量，见 settings 进度条）。
//! - 混用证伪前失败语义（方案 §6）：worker 失败不写新 runtime 向量，记
//!   `WorkerUnavailable`，稍后重试；Dart 的 `EmbeddingFailures` 落库由调用方照常处理。
//!
//! 进程模型：单 worker 子进程（stdio 帧），串行请求；闲置 120s 发 shutdown 退出；
//! 超时（文本 30s / 图片 120s）先 cancel、无响应则 kill 重启；崩溃退避重启
//! （1s/5s/30s，连续 5 次失败标记不可用直到下次 load）。

use embed_core::worker_frame::*;
use std::io::BufReader;
use std::io::BufWriter;
use std::path::PathBuf;
use std::process::Child;
use std::process::ChildStdin;
use std::process::ChildStdout;
use std::process::Command;
use std::process::Stdio;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::Duration;
use std::time::Instant;

pub mod error;
pub use error::{EmbeddingApiError, EmbeddingErrorKind};

/// 后端选择（与设置页 `embedBackend` / `FO_EMBED_BACKEND` 对应）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbedBackend {
    OrtCpu,
    LlamaWorker,
}

impl EmbedBackend {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::OrtCpu => "ort-cpu",
            Self::LlamaWorker => "llama-worker",
        }
    }
}

/// 当前实际产生向量的 runtime（Dart 查询向量库时透传，见方案 §3）。
/// 存量 ORT 行 `runtime='ort-cpu'`；worker 行由 hello 决定（§6）。
pub const RUNTIME_ORT_CPU: &str = "ort-cpu";

/// llama.cpp 构建 pin（与 27.2b 实测一致）。
pub const LLAMA_BACKEND_TAG: &str = "b11669";

pub fn llama_runtime(vulkan: bool) -> String {
    format!(
        "llama-cpp-{LLAMA_BACKEND_TAG}-{}",
        if vulkan { "vulkan" } else { "cpu" }
    )
}

#[derive(Debug, Clone)]
pub struct RouterConfig {
    pub backend: EmbedBackend,
    pub worker_exe: PathBuf,
    pub model_gguf: PathBuf,
    pub mmproj_gguf: PathBuf,
    pub idle_secs: u64,
    pub text_timeout: Duration,
    pub image_timeout: Duration,
    /// worker 日志目录（Dart 传应用 support `logs/`，与 performance.log 同目录）。
    pub log_dir: Option<PathBuf>,
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            backend: EmbedBackend::OrtCpu,
            worker_exe: PathBuf::from("llama-embed-worker"),
            model_gguf: PathBuf::new(),
            mmproj_gguf: PathBuf::new(),
            idle_secs: 120,
            text_timeout: Duration::from_secs(30),
            image_timeout: Duration::from_secs(120),
            log_dir: None,
        }
    }
}

struct Worker {
    child: Child,
    stdin: BufWriter<ChildStdin>,
    stdout: BufReader<ChildStdout>,
    runtime: String,
    last_used: Instant,
    req_seq: u64,
}

struct Inner {
    cfg: RouterConfig,
    worker: Option<Worker>,
    failures: u32,
    backoff_until: Option<Instant>,
    worker_unavailable: bool,
}

pub struct EmbedRouter {
    inner: Mutex<Inner>,
}

impl EmbedRouter {
    pub fn new(cfg: RouterConfig) -> Arc<Self> {
        Arc::new(Self {
            inner: Mutex::new(Inner {
                cfg,
                worker: None,
                failures: 0,
                backoff_until: None,
                worker_unavailable: false,
            }),
        })
    }

    pub fn set_config(&self, cfg: RouterConfig) {
        let mut inner = self.inner.lock().unwrap();
        inner.cfg = cfg;
        inner.worker_unavailable = false;
        inner.failures = 0;
        inner.backoff_until = None;
        // 后端切换时停掉旧 worker（下次请求按需重起）。
        stop_worker(&mut inner);
    }

    pub fn backend(&self) -> EmbedBackend {
        self.inner.lock().unwrap().cfg.backend
    }

    /// 当前实际产生向量的 runtime（ORT 或 hello 实测的 worker runtime）。
    pub fn current_runtime(&self) -> String {
        let inner = self.inner.lock().unwrap();
        if inner.cfg.backend == EmbedBackend::LlamaWorker {
            if let Some(w) = inner.worker.as_ref() {
                return w.runtime.clone();
            }
        }
        RUNTIME_ORT_CPU.to_string()
    }

    /// 向 worker 发送嵌入请求并取向量（含超时→cancel→kill 重启语义）。
    /// 单 worker 串行：整个往返持有 router 锁（Dart 侧本就经 `_gate` 串行加载，
    /// 推理并发由 router 内排队；FRB 线程池多线程调用在此排队）。
    pub fn call_worker(
        &self,
        req: &Request,
        timeout: Duration,
    ) -> Result<(Vec<f32>, f64), EmbeddingApiError> {
        let mut inner = self.inner.lock().unwrap();
        if !want_worker(&inner) {
            return Err(EmbeddingApiError::worker_unavailable("worker 未启用或熔断中"));
        }
        // 闲置超时：停掉旧 worker，下次重建（释放显存/内存）。
        if inner
            .worker
            .as_ref()
            .map(|w| w.last_used.elapsed() > Duration::from_secs(inner.cfg.idle_secs))
            .unwrap_or(false)
        {
            stop_worker(&mut inner);
        }
        ensure_worker(self, &mut inner)?;
        // 单次往返（不交出锁，天然串行；FRB 调用线程在此阻塞排队）。
        let out = roundtrip_once(self, &mut inner, req, timeout, Instant::now());
        // 锁交出期间（如后端切换停掉 worker）导致的"worker 已切换"：重建后重试一次。
        if matches!(out, Err(EmbedRouterRetry::Rebuilt)) {
            ensure_worker(self, &mut inner)?;
            return match roundtrip_once(self, &mut inner, req, timeout, Instant::now()) {
                Ok(v) => Ok(v),
                Err(EmbedRouterRetry::Rebuilt) => {
                    Err(EmbeddingApiError::worker_unavailable("worker 已切换"))
                }
                Err(EmbedRouterRetry::Done(e)) => Err(e),
            };
        }
        match out {
            Ok(v) => Ok(v),
            Err(EmbedRouterRetry::Done(e)) => Err(e),
            Err(EmbedRouterRetry::Rebuilt) => {
                Err(EmbeddingApiError::worker_unavailable("worker 已切换"))
            }
        }
    }

    /// 供单测：当前 worker 是否存活。
    #[cfg(test)]
    pub fn worker_alive(&self) -> bool {
        let mut inner = self.inner.lock().unwrap();
        inner
            .worker
            .as_mut()
            .map(|w| w.child.try_wait().ok().flatten().is_none())
            .unwrap_or(false)
    }
}

/// 往返结果：`Rebuilt` = 中途 worker 被外部停掉，已重建，调用方重试一次。
enum EmbedRouterRetry {
    Done(EmbeddingApiError),
    Rebuilt,
}

impl From<EmbeddingApiError> for EmbedRouterRetry {
    fn from(e: EmbeddingApiError) -> Self {
        Self::Done(e)
    }
}

/// 是否应走 worker（开关开、未熔断、退避期外）。
fn want_worker(inner: &Inner) -> bool {
    inner.cfg.backend == EmbedBackend::LlamaWorker
        && !inner.worker_unavailable
        && inner.backoff_until.is_none_or(|t| Instant::now() >= t)
}

fn stop_worker(inner: &mut Inner) {
    if let Some(mut w) = inner.worker.take() {
        let _ = write_request(&mut w.stdin, 0, &Request::Shutdown, 64 * 1024 * 1024);
        // 给 300ms 优雅退出，否则 kill。
        let start = Instant::now();
        loop {
            match w.child.try_wait() {
                Ok(Some(_)) => break,
                Ok(None) if start.elapsed() < Duration::from_millis(300) => {
                    std::thread::sleep(Duration::from_millis(10));
                }
                _ => {
                    let _ = w.child.kill();
                    let _ = w.child.wait();
                    break;
                }
            }
        }
    }
}

fn kill_worker(inner: &mut Inner) {
    if let Some(mut w) = inner.worker.take() {
        let _ = w.child.kill();
        let _ = w.child.wait();
    }
}

/// 崩溃/超时记账：退避重启，连续 5 次失败熔断到下次 load。
fn note_crash(inner: &mut Inner) {
    inner.failures += 1;
    let backoff = match inner.failures {
        1 => 1,
        2 => 5,
        _ => 30,
    };
    if inner.failures >= 5 {
        inner.worker_unavailable = true;
        inner.backoff_until = None;
    } else {
        inner.backoff_until = Some(Instant::now() + Duration::from_secs(backoff));
    }
}

/// worker 日志目录：由 Dart 经 `embedding_load_ex` 的 `log_dir` 传入
/// （`getApplicationSupportDirectory()/logs`，与 performance.log 同目录）；
/// 未传时回退 `FO_LLAMA_LOG_DIR` / `LOCALAPPDATA`。保留最近 5 个文件。
fn worker_log_file(log_dir: Option<&std::path::Path>) -> Option<std::fs::File> {
    let dir = if let Some(d) = log_dir {
        d.to_path_buf()
    } else if let Ok(d) = std::env::var("FO_LLAMA_LOG_DIR") {
        PathBuf::from(d)
    } else if let Ok(l) = std::env::var("LOCALAPPDATA") {
        PathBuf::from(l).join("file_organizer").join("logs")
    } else if let Ok(h) = std::env::var("HOME") {
        PathBuf::from(h)
            .join(".local/share/file_organizer/logs")
    } else {
        std::env::temp_dir().join("file_organizer-logs")
    };
    if std::fs::create_dir_all(&dir).is_err() {
        return None;
    }
    // 轮转：>5 个则删最旧的。
    if let Ok(entries) = std::fs::read_dir(&dir) {
        let mut logs: Vec<_> = entries
            .flatten()
            .filter(|e| {
                e.file_name()
                    .to_string_lossy()
                    .starts_with("llama-worker-")
            })
            .map(|e| e.path())
            .collect();
        if logs.len() >= 5 {
            logs.sort();
            for old in &logs[..logs.len() - 4] {
                let _ = std::fs::remove_file(old);
            }
        }
    }
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    std::fs::File::create(dir.join(format!("llama-worker-{stamp}.log"))).ok()
}

/// 确保 worker 存活并完成 hello 握手；返回其 runtime。
fn ensure_worker(router: &EmbedRouter, inner: &mut Inner) -> Result<String, EmbeddingApiError> {
    let _ = router;
    if inner.worker.is_some() {
        let alive = inner
            .worker
            .as_mut()
            .map(|w| w.child.try_wait().ok().flatten().is_none())
            .unwrap_or(false);
        if alive {
            return Ok(inner.worker.as_ref().map(|w| w.runtime.clone()).unwrap_or_default());
        }
        stop_worker(inner);
    }
    let cfg = inner.cfg.clone();
    if cfg.model_gguf.as_os_str().is_empty() || cfg.mmproj_gguf.as_os_str().is_empty() {
        return Err(EmbeddingApiError::worker_unavailable("worker 模型路径未配置"));
    }
    let mut child = Command::new(&cfg.worker_exe)
        .arg("--model")
        .arg(&cfg.model_gguf)
        .arg("--mmproj")
        .arg(&cfg.mmproj_gguf)
        // 打包布局：worker 在 <app>/，llama*.dll 在 <app>/llama/ —— 子进程 PATH 前置
        // llama 目录，cwd 置 exe 目录（Worker 用 LoadLibrary 默认搜索 PATH）。
        .current_dir(
            cfg.worker_exe
                .parent()
                .filter(|p| !p.as_os_str().is_empty())
                .unwrap_or(std::path::Path::new(".")),
        )
        .env(
            "PATH",
            {
                let mut llama_dir = cfg.worker_exe.clone();
                llama_dir.pop();
                llama_dir.push("llama");
                let old = std::env::var("PATH").unwrap_or_default();
                format!("{};{}", llama_dir.display(), old)
            },
        )
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(match worker_log_file(cfg.log_dir.as_deref()) {
            Some(f) => Stdio::from(f),
            None => Stdio::inherit(),
        })
        .spawn()
        .map_err(|e| EmbeddingApiError::worker_unavailable(format!("worker 启动失败: {e}")))?;
    let stdin = child.stdin.take().expect("piped");
    let stdout = child.stdout.take().expect("piped");
    let mut w = Worker {
        child,
        stdin: BufWriter::new(stdin),
        stdout: BufReader::new(stdout),
        runtime: String::new(),
        last_used: Instant::now(),
        req_seq: 1,
    };
    // hello 握手（10s）：确定 runtime（vulkan/cpu）。
    if let Err(e) = write_request(&mut w.stdin, 0, &Request::Hello, 1024) {
        kill_worker(inner);
        note_crash(inner);
        return Err(EmbeddingApiError::worker_unavailable(format!(
            "worker 握手失败: {e}"
        )));
    }
    let start = Instant::now();
    loop {
        if start.elapsed() > Duration::from_secs(10) {
            kill_worker(inner);
            note_crash(inner);
            return Err(EmbeddingApiError::worker_unavailable("worker 握手超时"));
        }
        {
            use std::io::BufRead;
            match w.stdout.fill_buf() {
                Ok(b) if !b.is_empty() => break,
                Ok(_) => {}
                Err(_) => {
                    note_crash(inner);
                    return Err(EmbeddingApiError::worker_unavailable("worker 握手中断"));
                }
            }
        }
        if w.child.try_wait().ok().flatten().is_some() {
            note_crash(inner);
            return Err(EmbeddingApiError::worker_crashed(None));
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    let (rid, res) = match read_response(&mut w.stdout) {
        Ok(v) => v,
        Err(e) => {
            kill_worker(inner);
            note_crash(inner);
            return Err(EmbeddingApiError::worker_unavailable(format!(
                "worker 握手读失败: {e}"
            )));
        }
    };
    if rid != 0 {
        kill_worker(inner);
        note_crash(inner);
        return Err(EmbeddingApiError::worker_unavailable("worker 握手错位"));
    }
    let hello = match res {
        Ok(_) => {
            kill_worker(inner);
            note_crash(inner);
            return Err(EmbeddingApiError::worker_unavailable("worker 握手返回了向量"));
        }
        Err(json) => json,
    };
    let hv: serde_json::Value = match serde_json::from_str(&hello) {
        Ok(v) => v,
        Err(e) => {
            kill_worker(inner);
            note_crash(inner);
            return Err(EmbeddingApiError::worker_unavailable(format!(
                "worker hello 非 JSON: {e}"
            )));
        }
    };
    let vulkan = hv.get("vulkan").and_then(|v| v.as_bool()).unwrap_or(false);
    w.runtime = llama_runtime(vulkan);
    w.last_used = Instant::now();
    let rt = w.runtime.clone();
    inner.worker = Some(w);
    inner.failures = 0;
    inner.backoff_until = None;
    Ok(rt)
}

/// 单次请求往返。`Rebuilt` 表示中途 worker 被外部停掉（调用方重建后重试一次）。
fn roundtrip_once(
    router: &EmbedRouter,
    inner: &mut Inner,
    req: &Request,
    timeout: Duration,
    start: Instant,
) -> Result<(Vec<f32>, f64), EmbedRouterRetry> {
    let _ = router;
    if inner.worker.is_none() {
        return Err(EmbedRouterRetry::Rebuilt);
    }
    let req_id = inner.worker.as_ref().map(|w| w.req_seq).unwrap_or(0);
    if let Some(w) = inner.worker.as_mut() {
        w.req_seq += 1;
    }
    {
        let stdin = &mut inner.worker.as_mut().expect("present").stdin;
        if let Err(e) = write_request(stdin, req_id, req, 64 * 1024 * 1024) {
            kill_worker(inner);
            note_crash(inner);
            return Err(EmbedRouterRetry::Done(
                EmbeddingApiError::worker_unavailable(format!("worker 写失败: {e}")),
            ));
        }
    }
    loop {
        if start.elapsed() >= timeout {
            // 超时：先 cancel，500ms 内无响应则 kill（记账由 kill 后统一处理）。
            if let Some(w) = inner.worker.as_mut() {
                let _ = write_request(&mut w.stdin, req_id, &Request::Cancel(req_id), 1024);
            }
            std::thread::sleep(Duration::from_millis(500));
            kill_worker(inner);
            note_crash(inner);
            return Err(EmbedRouterRetry::Done(EmbeddingApiError::worker_timeout(
                req_id,
            )));
        }
        {
            use std::io::BufRead;
            let stdout = &mut inner.worker.as_mut().expect("present").stdout;
            match stdout.fill_buf() {
                Ok(b) if !b.is_empty() => break,
                Ok(_) => {}
                Err(_) => {
                    kill_worker(inner);
                    note_crash(inner);
                    return Err(EmbedRouterRetry::Done(
                        EmbeddingApiError::worker_unavailable("worker 输出中断"),
                    ));
                }
            }
        }
        {
            let child = &mut inner.worker.as_mut().expect("present").child;
            if child.try_wait().ok().flatten().is_some() {
                let code = child.try_wait().ok().flatten().and_then(|s| s.code());
                kill_worker(inner);
                note_crash(inner);
                return Err(EmbedRouterRetry::Done(EmbeddingApiError::worker_crashed(code)));
            }
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    let (rid, res) = {
        let stdout = &mut inner.worker.as_mut().expect("present").stdout;
        match read_response(stdout) {
            Ok(v) => v,
            Err(e) => {
                kill_worker(inner);
                note_crash(inner);
                return Err(EmbedRouterRetry::Done(
                    EmbeddingApiError::worker_unavailable(format!("worker 读失败: {e}")),
                ));
            }
        }
    };
    if rid != req_id {
        kill_worker(inner);
        note_crash(inner);
        return Err(EmbedRouterRetry::Done(
            EmbeddingApiError::worker_unavailable(format!("worker 响应错位 {rid} != {req_id}")),
        ));
    }
    if let Some(w) = inner.worker.as_mut() {
        w.last_used = Instant::now();
    }
    match res {
        Ok((v, ms)) => {
            inner.failures = 0;
            Ok((v, ms))
        }
        Err(msg) => {
            kill_worker(inner);
            note_crash(inner);
            Err(EmbedRouterRetry::Done(
                EmbeddingApiError::worker_unavailable(format!("worker 错误: {msg}")),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn llama_cfg(exe: &str) -> RouterConfig {
        RouterConfig {
            backend: EmbedBackend::LlamaWorker,
            worker_exe: PathBuf::from(exe),
            ..Default::default()
        }
    }

    #[test]
    fn default_backend_is_ort_cpu_and_runtime_is_ort() {
        let r = EmbedRouter::new(RouterConfig::default());
        assert_eq!(r.backend(), EmbedBackend::OrtCpu);
        assert_eq!(r.current_runtime(), RUNTIME_ORT_CPU);
    }

    #[test]
    fn llama_runtime_tagging() {
        assert_eq!(llama_runtime(true), "llama-cpp-b11669-vulkan");
        assert_eq!(llama_runtime(false), "llama-cpp-b11669-cpu");
    }

    #[test]
    fn missing_model_paths_are_unavailable() {
        let r = EmbedRouter::new(llama_cfg("llama-embed-worker"));
        let err = r
            .call_worker(&Request::Text("x".into()), Duration::from_secs(1))
            .unwrap_err();
        assert!(err.detail.contains("模型路径未配置"), "{}", err.detail);
    }

    /// 崩溃路径：用系统命令当"worker"，启动后立即退出 → WorkerUnavailable（非崩溃进程泄露）。
    #[cfg(windows)]
    #[test]
    fn crashing_worker_is_reported_and_does_not_hang() {
        let mut cfg = llama_cfg("cmd.exe");
        cfg.model_gguf = PathBuf::from("x.gguf");
        cfg.mmproj_gguf = PathBuf::from("y.gguf");
        let r = EmbedRouter::new(cfg);
        let err = r
            .call_worker(&Request::Text("hi".into()), Duration::from_secs(3))
            .unwrap_err();
        assert!(
            err.detail.contains("崩溃") || err.detail.contains("中断") || err.detail.contains("握手"),
            "{}",
            err.detail
        );
        // 失败后不遗留存活 worker。
        assert!(!r.worker_alive());
    }

    /// 失败后进入退避：紧接着的请求不再反复拉起进程（避免崩溃风暴）。
    #[cfg(windows)]
    #[test]
    fn failure_enters_backoff_and_rejects_next_call() {
        let mut cfg = llama_cfg("cmd.exe");
        cfg.model_gguf = PathBuf::from("x.gguf");
        cfg.mmproj_gguf = PathBuf::from("y.gguf");
        let r = EmbedRouter::new(cfg);
        let first = r.call_worker(&Request::Text("hi".into()), Duration::from_secs(3));
        assert!(first.is_err());
        // 退避期内：直接拒绝（不再启动新进程）。
        let second = r
            .call_worker(&Request::Text("hi".into()), Duration::from_secs(1))
            .unwrap_err();
        assert!(
            second.detail.contains("未启用或熔断"),
            "second detail = {}",
            second.detail
        );
    }

    /// 连续失败达 5 次后置位 worker_unavailable（熔断到下次 load）。
    #[test]
    fn crash_bookkeeping_marks_unavailable_after_five() {
        let r = EmbedRouter::new(llama_cfg("llama-embed-worker"));
        {
            let mut inner = r.inner.lock().unwrap();
            for _ in 0..5 {
                note_crash(&mut inner);
            }
            assert!(inner.worker_unavailable);
            assert!(inner.backoff_until.is_none());
        }
    }
}
