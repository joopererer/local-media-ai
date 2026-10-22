//! 27.1A：嵌入一致性工具（capture / check）。
//!
//! 用法：
//!   embed_consistency capture --suite <json> --image_dir <dir>
//!       --ort_lib <dll> --model_dir <dir> [--threads 4] [--soft_tokens 140]
//!       --out <golden.json>
//!   embed_consistency check --suite <json> --image_dir <dir>
//!       --ort_lib <dll> --model_dir <dir> [--threads 4] [--soft_tokens 140]
//!       --golden <golden.json> [--min_cos 0.9999]
//!
//! capture：对 suite 内输入生成向量，输出 GeneratorId + 输入清单（含 sha256）+ 向量。
//! check：重算并与 golden 比余弦，报告 min/p50/mean；输入漂移或余弦不足即非零退出。
//!
//! 用途：① 运行库/模型升级前的回归；② 27.0 搬迁的行为不变证明；③ 27.2 后端对比。

use embed_core::{init_ort, Embedder};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

const MODEL_ID: &str = "embeddinggemma2-st140";
const RUN_SECS_NOTE: &str = "same-backend expect exact; cross-backend min_cos";

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

fn sha256_bytes(b: &[u8]) -> String {
    hex::encode(Sha256::digest(b))
}

fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let (mut dot, mut na, mut nb) = (0.0, 0.0, 0.0);
    for i in 0..a.len().min(b.len()) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    dot / (na.sqrt() * nb.sqrt()).max(f64::MIN_POSITIVE)
}

#[derive(Clone)]
struct Input {
    id: String,
    kind: String,
    text: String,
    file: String,
    sha: String,
}

fn load_suite(path: &str, image_dir: &Path) -> (String, Vec<Input>) {
    let doc: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).expect("suite 读取失败"))
            .expect("suite JSON 解析失败");
    let prefix = doc["prefix"].as_str().unwrap_or("").to_string();
    let mut inputs = Vec::new();
    for t in doc["texts"].as_array().cloned().unwrap_or_default() {
        let text = t["text"].as_str().expect("texts[].text").to_string();
        let id = t["id"].as_str().unwrap_or(&text).to_string();
        inputs.push(Input {
            sha: sha256_bytes(text.as_bytes()),
            id,
            kind: "text".into(),
            text,
            file: String::new(),
        });
    }
    for m in doc["images"].as_array().cloned().unwrap_or_default() {
        let file = m["file"].as_str().expect("images[].file").to_string();
        let id = m["id"].as_str().unwrap_or(&file).to_string();
        let bytes = std::fs::read(image_dir.join(&file))
            .unwrap_or_else(|e| panic!("图片读取失败 {file}: {e}"));
        inputs.push(Input {
            sha: sha256_bytes(&bytes),
            id,
            kind: "image".into(),
            text: String::new(),
            file,
        });
    }
    (prefix, inputs)
}

fn embed_image(embedder: &Embedder, image_dir: &Path, file: &str, soft_tokens: u32) -> Vec<f64> {
    let r = embedder
        .embed_image_file(&image_dir.join(file), soft_tokens, false)
        .unwrap_or_else(|e| panic!("图片嵌入失败 {file}: {e}"));
    r.vector.iter().map(|&x| x as f64).collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let need = |n: &str| arg(&args, n).unwrap_or_else(|| panic!("缺少参数 {n}"));
    let sub = args.get(1).cloned().unwrap_or_default();
    if sub != "capture" && sub != "check" {
        panic!("用法：embed_consistency (capture|check) --suite ... --out/--golden ...");
    }
    let threads: usize = arg(&args, "--threads").map_or(4, |s| s.parse().expect("--threads"));
    let soft_tokens: u32 =
        arg(&args, "--soft_tokens").map_or(140, |s| s.parse().expect("--soft_tokens"));
    let image_dir = PathBuf::from(arg(&args, "--image_dir").unwrap_or_default());

    init_ort(&PathBuf::from(need("--ort_lib")))?;
    let t0 = Instant::now();
    let embedder = Embedder::new(&PathBuf::from(need("--model_dir")), threads)?;
    let load_s = t0.elapsed().as_secs_f64();

    let (prefix, inputs) = load_suite(&need("--suite"), &image_dir);

    let mut items = Vec::new();
    let t1 = Instant::now();
    for inp in &inputs {
        let (n_tokens, vec) = if inp.kind == "text" {
            let n = embedder.token_ids(&inp.text, &prefix)?.len();
            let v: Vec<f32> = embedder.embed(&inp.text, &prefix)?;
            (n, v.iter().map(|&x| x as f64).collect::<Vec<_>>())
        } else {
            let v = embed_image(&embedder, &image_dir, &inp.file, soft_tokens);
            (0, v)
        };
        items.push(serde_json::json!({
            "id": inp.id, "kind": inp.kind,
            "sha256": inp.sha, "n_tokens": n_tokens, "vector": vec,
        }));
    }
    let avg_s = t1.elapsed().as_secs_f64() / inputs.len().max(1) as f64;

    let generator = serde_json::json!({
        "model": MODEL_ID,
        "runtime": "ort-cpu",
        "runtime_version": env!("CARGO_PKG_VERSION"),
        "precision": "int8",
        "soft_tokens": soft_tokens,
        "ort_threads": threads,
        "note": RUN_SECS_NOTE,
    });

    if sub == "capture" {
        let out = serde_json::json!({
            "suite": need("--suite"), "prefix": prefix,
            "generator": generator,
            "load_seconds": load_s, "avg_seconds_per_item": avg_s,
            "items": items,
        });
        std::fs::write(need("--out"), serde_json::to_string(&out)?)?;
        println!(
            "captured items={} load={load_s:.2}s avg={avg_s:.3}s/item",
            items.len()
        );
        return Ok(());
    }

    // check
    let golden: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(need("--golden"))?)?;
    let min_cos: f64 = arg(&args, "--min_cos").map_or(0.9999, |s| s.parse().expect("--min_cos"));
    // 同后端回归的硬门：逐元素最大绝对差。余弦对同后端扰动过于宽松
    // （对归一化 768 维向量扰动 0.02 仅使余弦下降约 4e-4），必须辅以本阈值。
    let max_abs_gate: Option<f64> = arg(&args, "--max_abs").map(|s| s.parse().expect("--max_abs"));
    let by_id: HashMap<_, _> = golden["items"]
        .as_array()
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .map(|it| (it["id"].as_str().unwrap_or("").to_string(), it))
        .collect();

    let mut cosines = Vec::new();
    let mut max_abs: f64 = 0.0;
    let mut failures: Vec<String> = Vec::new();
    for (item, inp) in items.iter().zip(inputs.iter()) {
        let id = item["id"].as_str().unwrap_or("");
        let cur: Vec<f64> = serde_json::from_value(item["vector"].clone())?;
        let Some(g) = by_id.get(id) else {
            failures.push(format!("{id}: golden 缺失"));
            continue;
        };
        if g["sha256"].as_str().unwrap_or("") != inp.sha {
            failures.push(format!("{id}: 输入漂移（sha256 不一致，重跑 capture）"));
            continue;
        }
        let old: Vec<f64> = serde_json::from_value(g["vector"].clone())?;
        if old.len() != cur.len() {
            failures.push(format!("{id}: 维度变化 {} vs {}", old.len(), cur.len()));
            continue;
        }
        let c = cosine(&old, &cur);
        cosines.push(c);
        for (x, y) in old.iter().zip(cur.iter()) {
            max_abs = max_abs.max((x - y).abs());
        }
        if c < min_cos {
            failures.push(format!("{id}: cos={c:.6} < {min_cos}"));
        }
    }
    if let Some(gate) = max_abs_gate {
        if max_abs > gate {
            failures.push(format!("max_abs_diff={max_abs:.3e} > {gate:.0e}"));
        }
    }
    cosines.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let (min, p50, mean) = if cosines.is_empty() {
        (0.0, 0.0, 0.0)
    } else {
        (
            cosines[0],
            cosines[cosines.len() / 2],
            cosines.iter().sum::<f64>() / cosines.len() as f64,
        )
    };
    println!(
        "check n={} min_cos={min_cos} min={min:.6} p50={p50:.6} mean={mean:.6} max_abs_diff={max_abs:.3e}",
        cosines.len()
    );
    if failures.is_empty() {
        println!("PASS");
        Ok(())
    } else {
        for f in &failures {
            eprintln!("FAIL {f}");
        }
        std::process::exit(2);
    }
}
