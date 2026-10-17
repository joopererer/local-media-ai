// 27.1B 检索评测 runner：文字搜图 + 以图搜图，公开集与私密集分开报告。
//
// 跑一次评测（ORT-CPU），把查询向量对全库图片向量做余弦排序，
// 按 golden 期望统计 Recall@K 与首位命中，结果入库 eval/history/（gitignore），
// 基线快照可入库（--snapshot 另存一份带日期的结果）。
//
// 用法（在 native/ 下）：
//   retrieval_eval --ort_lib <dll> --model_dir <dir>
//     --pub_dir <eval/public/images> --queries eval/queries.jsonl --expected eval/expected.jsonl
//     --sim eval/similar.jsonl [--priv_dir D:/test/images --priv_manifest eval/private/manifest.jsonl]
//     --threads 4 --soft_tokens 140 --top_k 10 [--history_dir eval/history]
//
// 输出 JSON + 人读摘要（文字搜图 / 以图搜图 × 公开 / 私密 四块）。

use embed_core::{init_ort, Embedder};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

fn cos(a: &[f32], b: &[f32]) -> f32 {
    let mut d = 0f64;
    let (mut na, mut nb) = (0f64, 0f64);
    for (x, y) in a.iter().zip(b.iter()) {
        d += f64::from(*x) * f64::from(*y);
        na += f64::from(*x) * f64::from(*x);
        nb += f64::from(*y) * f64::from(*y);
    }
    (d / (na.sqrt() * nb.sqrt())) as f32
}

#[derive(serde::Deserialize)]
struct Query {
    id: String,
    query_zh: String,
    query_en: String,
    #[allow(dead_code)]
    #[serde(default)]
    expected: Vec<String>,
    #[allow(dead_code)]
    #[serde(default)]
    images: Vec<String>,
}

#[derive(serde::Deserialize)]
struct Expect {
    query_id: String,
    lang: String,
    hit: Vec<String>,
}

#[derive(serde::Deserialize)]
struct SimGroup {
    id: String,
    kind: String,
    files: Vec<String>,
    excluded: Option<Vec<String>>,
    /// 难负样本组：不计 recall，单独报告误命中率（组员互进 top-K 的比例）。
    #[serde(default)]
    hard_negative: bool,
}

fn read_jsonl<T: serde::de::DeserializeOwned>(path: &str) -> Vec<T> {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("读 {path} 失败: {e}"));
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).expect("jsonl 解析失败"))
        .collect()
}

fn rank_of(order: &[String], want: &str) -> Option<usize> {
    order.iter().position(|f| f == want).map(|i| i + 1)
}

/// 文字搜图：在库 `lib`（文件名→向量）中检索查询，统计期望命中的 Recall@K 与首位命中。
/// 返回 (per_query 明细, recall@K, 首位命中率)。
fn eval_text_queries(
    embedder: &Embedder,
    queries: &[Query],
    expects: &HashMap<String, Expect>,
    lib: &HashMap<String, Vec<f32>>,
    top_k: usize,
) -> (Vec<serde_json::Value>, f64, f64) {
    let order: Vec<&String> = {
        let mut v: Vec<&String> = lib.keys().collect();
        v.sort();
        v
    };
    let mut rows = Vec::new();
    let (mut hit_sum, mut hit_n) = (0usize, 0usize);
    let mut first_hit = 0usize;
    for q in queries {
        let Some(exp) = expects.get(&q.id) else {
            continue;
        };
        for (lang, text) in [("zh", q.query_zh.as_str()), ("en", q.query_en.as_str())] {
            if exp.lang != "both" && exp.lang != lang {
                continue;
            }
            if text.trim().is_empty() {
                continue;
            }
            let qv = embedder.embed(text, "").expect("查询嵌入失败");
            let mut scored: Vec<(&String, f32)> =
                order.iter().map(|f| (*f, cos(&qv, &lib[*f]))).collect();
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let ranked: Vec<String> = scored.iter().map(|(f, _)| (*f).clone()).collect();
            let want: HashSet<&str> = exp.hit.iter().map(String::as_str).collect();
            let top: HashSet<&str> =
                ranked.iter().take(top_k).map(String::as_str).collect();
            let hits = want.intersection(&top).count();
            hit_sum += hits;
            hit_n += want.len();
            let first_rank = exp.hit.iter().filter_map(|w| rank_of(&ranked, w)).min();
            if first_rank == Some(1) {
                first_hit += 1;
            }
            rows.push(serde_json::json!({
                "query": q.id, "lang": lang, "nhit": exp.hit.len(),
                "recall": hits, "first_rank": first_rank,
                "top": &ranked[..ranked.len().min(top_k)],
            }));
        }
    }
    let recall = if hit_n > 0 { hit_sum as f64 / hit_n as f64 } else { 1.0 };
    let first = if !rows.is_empty() { first_hit as f64 / rows.len() as f64 } else { 1.0 };
    (rows, recall, first)
}

/// 以图搜图：组内每张作查询，在库中检索其余组员，统计"组员进入 top-K"的 Recall@K。
/// 难负样本组（hard_negative，期望组员互不相关）：不计 recall，
/// 返回误命中率 = 组员出现在彼此 top-K 的查询占比（越低越好），明细照常记录。
fn eval_image_queries(
    lib: &HashMap<String, Vec<f32>>,
    groups: &[SimGroup],
    top_k: usize,
    exclude_from_lib: &HashSet<String>,
) -> (Vec<serde_json::Value>, f64, Option<f64>) {
    let (mut hit_sum, mut hit_n) = (0usize, 0usize);
    let (mut fp_q, mut fp_n) = (0usize, 0usize);
    let mut rows = Vec::new();
    for g in groups {
        let members: Vec<&String> =
            g.files.iter().filter(|f| lib.contains_key(*f)).collect();
        if members.len() < 2 {
            continue;
        }
        for q in &members {
            let qv = &lib[*q];
            let mut scored: Vec<(&String, f32)> = lib
                .iter()
                .filter(|(f, _)| *f != *q && !exclude_from_lib.contains(*f))
                .map(|(f, v)| (f, cos(qv, v)))
                .collect();
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let ranked: Vec<&String> = scored.iter().map(|(f, _)| *f).collect();
            let want: HashSet<&String> =
                members.iter().filter(|m| m.as_str() != q.as_str()).copied().collect();
            let top: HashSet<&String> =
                ranked.iter().take(top_k).cloned().collect();
            let hits = want.intersection(&top).count();
            if g.hard_negative {
                // 难负样本：组员本不应互相关；记录"误命中"（有组员进入 top-K 的查询占比）。
                fp_n += 1;
                if hits > 0 {
                    fp_q += 1;
                }
            } else {
                hit_sum += hits;
                hit_n += want.len();
            }
            let first_rank = want
                .iter()
                .filter_map(|w| ranked.iter().position(|f| f == w).map(|i| i + 1))
                .min();
            rows.push(serde_json::json!({
                "group": g.id, "query": q, "kind": g.kind,
                "hard_negative": g.hard_negative,
                "nmembers": want.len(), "recall": hits, "first_rank": first_rank,
                "top": ranked.iter().take(top_k).collect::<Vec<_>>(),
            }));
        }
    }
    let recall = if hit_n > 0 { hit_sum as f64 / hit_n as f64 } else { 1.0 };
    let fp_rate = if fp_n > 0 { Some(fp_q as f64 / fp_n as f64) } else { None };
    (rows, recall, fp_rate)
}

fn embed_dir(
    embedder: &Embedder,
    dir: &std::path::Path,
    files: &[String],
    soft: u32,
    tag: &str,
) -> HashMap<String, Vec<f32>> {
    let mut lib = HashMap::new();
    for (i, f) in files.iter().enumerate() {
        let out = embedder.embed_image_file(&dir.join(f), soft, false);
        match out {
            Ok(e) => {
                lib.insert(f.clone(), e.vector);
            }
            Err(e) => eprintln!("{tag} 跳过 {f}: {e}"),
        }
        if (i + 1) % 20 == 0 {
            eprintln!("{tag} {}/{}", i + 1, files.len());
        }
    }
    lib
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let need = |n: &str| arg(&args, n).unwrap_or_else(|| panic!("缺少参数 {n}"));
    let threads: usize = arg(&args, "--threads").map_or(4, |s| s.parse().unwrap());
    let soft: u32 = arg(&args, "--soft_tokens").map_or(140, |s| s.parse().unwrap());
    let top_k: usize = arg(&args, "--top_k").map_or(10, |s| s.parse().unwrap());
    let history_dir = need("--history_dir");
    let snapshot = arg(&args, "--snapshot").unwrap_or_default();

    init_ort(&PathBuf::from(need("--ort_lib")))?;
    let embedder = Embedder::new(&PathBuf::from(need("--model_dir")), threads)?;

    // --- 库：公开集（入库图片全量）+ 私密集（manifest 清单）
    let pub_dir = PathBuf::from(need("--pub_dir"));
    let mut pub_files: Vec<String> = std::fs::read_dir(&pub_dir)
        .expect("公开目录不可读")
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|f| f.ends_with(".jpg") || f.ends_with(".png"))
        .collect();
    pub_files.sort();
    let pub_lib = embed_dir(&embedder, &pub_dir, &pub_files, soft, "公开");

    let mut priv_lib = HashMap::new();
    if let Some(p) = arg(&args, "--priv_manifest") {
        let text = std::fs::read_to_string(&p).expect("私密 manifest 不可读");
        let priv_manifest: Vec<String> = text
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| {
                let v: serde_json::Value = serde_json::from_str(l).unwrap();
                v["file"].as_str().unwrap().to_string()
            })
            .collect();
        let priv_dir = PathBuf::from(need("--priv_dir"));
        priv_lib = embed_dir(&embedder, &priv_dir, &priv_manifest, soft, "私密");
    }

    // --- 查询与期望（公开集 golden）
    let queries: Vec<Query> = read_jsonl(&need("--queries"));
    let expects: HashMap<String, Expect> =
        read_jsonl::<Expect>(&need("--expected")).into_iter().map(|e| (e.query_id.clone(), e)).collect();
    let sim: Vec<SimGroup> = read_jsonl(&need("--sim"));

    // 公开以图搜图排除"已排除"图片（不适合评测的内容）
    let mut pub_excluded = HashSet::new();
    for g in &sim {
        if let Some(ex) = &g.excluded {
            for f in ex {
                pub_excluded.insert(f.clone());
            }
        }
    }

    let priv_sim: Vec<SimGroup> = arg(&args, "--priv_sim").map_or(Vec::new(), |p| read_jsonl(&p));
    let priv_queries: Vec<Query> =
        arg(&args, "--priv_queries").map_or(Vec::new(), |p| read_jsonl(&p));
    // 私密文字搜图期望：每条私密查询命中其 images 清单（人审标注）。
    let priv_expects: HashMap<String, Expect> = priv_queries
        .iter()
        .map(|q| {
            (
                q.id.clone(),
                Expect {
                    query_id: q.id.clone(),
                    lang: "both".to_string(),
                    hit: q.images.clone(),
                },
            )
        })
        .collect();
    let priv_excluded = HashSet::new();

    let (pub_text_rows, pub_text_recall, pub_text_first) =
        eval_text_queries(&embedder, &queries, &expects, &pub_lib, top_k);
    let (pub_img_rows, pub_img_recall, _) =
        eval_image_queries(&pub_lib, &sim, top_k, &pub_excluded);

    let priv_block = if !priv_lib.is_empty() {
        // 私密文字搜图：私密查询在私密库中检索（同库口径）；私密查询无结果则跳过该块。
        let (t_rows, t_recall, t_first) =
            eval_text_queries(&embedder, &priv_queries, &priv_expects, &priv_lib, top_k);
        // 私密以图搜图组是难负样本：不计 recall，只报误命中率（组员互进 top-K 的查询占比）。
        let (i_rows, i_recall, i_fp) =
            eval_image_queries(&priv_lib, &priv_sim, top_k, &priv_excluded);
        Some(serde_json::json!({
            "n_images": priv_lib.len(),
            "text2image": {"recall": t_recall, "first_hit_rate": t_first, "queries": t_rows},
            "image2image": {"recall": i_recall, "false_hit_rate": i_fp, "groups": i_rows},
        }))
    } else {
        None
    };

    let report = serde_json::json!({
        "backend": "ort-cpu", "threads": threads, "soft_tokens": soft, "top_k": top_k,
        "public": {
            "n_images": pub_lib.len(),
            "text2image": {"recall": pub_text_recall, "first_hit_rate": pub_text_first, "queries": pub_text_rows},
            "image2image": {"recall": pub_img_recall, "groups": pub_img_rows},
        },
        "private": priv_block,
    });

    std::fs::create_dir_all(&history_dir)?;
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let hist_path = format!("{history_dir}/retrieval-{stamp}.json");
    std::fs::write(&hist_path, serde_json::to_string(&report)?)?;
    if !snapshot.is_empty() {
        std::fs::write(&snapshot, serde_json::to_string_pretty(&report)?)?;
        eprintln!("快照 -> {snapshot}");
    }

    // 人读摘要：文字搜图 / 以图搜图 × 公开 / 私密
    println!("== 文字搜图 recall@{top_k} / 首位命中率 ==");
    println!(
        "   公开（{} 图）: recall={:.4} first={:.4} ({} 条查询×中英)",
        pub_lib.len(),
        pub_text_recall,
        pub_text_first,
        pub_text_rows.len()
    );
    if let Some(p) = &priv_block {
        println!(
            "   私密（{} 图）: recall={:.4} first={:.4}（同库口径：私密查询在私密库中检索）",
            p["n_images"],
            p["text2image"]["recall"].as_f64().unwrap(),
            p["text2image"]["first_hit_rate"].as_f64().unwrap()
        );
    }
    println!("== 以图搜图 recall@{top_k} ==");
    println!("   公开: recall={:.4} ({} 组内查询)", pub_img_recall, pub_img_rows.len());
    if let Some(p) = &priv_block {
        // 私密组是难负样本：报误命中率（组员互进 top-K 的查询占比，越低越好），不计 recall。
        let fp = p["image2image"]["false_hit_rate"].as_f64();
        println!(
            "   私密（难负样本）: 误命中率={} ({} 组内查询)",
            fp.map_or("n/a".to_string(), |v| format!("{v:.4}")),
            p["image2image"].as_object().unwrap()["groups"].as_array().unwrap().len()
        );
    }
    for r in &pub_text_rows {
        let rec = r["recall"].as_u64().unwrap();
        let nhit = r["nhit"].as_u64().unwrap();
        if rec < nhit {
            println!(
                "   !! {}/{}: 期望 {nhit} 命中 {rec}，首位 rank={} top={:?}",
                r["query"].as_str().unwrap(),
                r["lang"].as_str().unwrap(),
                r["first_rank"],
                r["top"]
            );
        }
    }
    println!("-> {hist_path}");
    Ok(())
}
