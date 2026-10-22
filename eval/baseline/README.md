# Public evaluation baseline

The baseline below was produced with the **public evaluation set only**
(36 CC0 / public-domain images, 32 bilingual queries, 24 image-query groups) and
the **ONNX Runtime CPU backend**.

## How it was produced

```bash
cargo run --release -p embed_core --example retrieval_eval -- \
  --ort_lib <ONNX_RUNTIME_LIB> \
  --model_dir <EMBEDDINGGEMMA2_MODEL_DIR> \
  --pub_dir eval/public/images \
  --queries eval/queries.jsonl \
  --expected eval/expected.jsonl \
  --sim eval/similar.jsonl \
  --threads 4 --soft_tokens 140 --top_k 10 \
  --history_dir eval/history \
  --snapshot eval/baseline/public-baseline-ort-cpu.json
```

| Parameter | Value |
|---|---|
| Backend | `ort-cpu` (ONNX Runtime, dynamic library) |
| Model | EmbeddingGemma 2, quantized ONNX export (`embeddinggemma2-st140`) |
| Images | 36 (all CC0 or public domain) |
| Queries | 32 bilingual (zh + en) text queries, 24 image-query groups |
| `threads` / `soft_tokens` / `top_k` | 4 / 140 / 10 |

## Result

| Metric | Value |
|---|---|
| Text → image, recall@10 | **0.9861** |
| Text → image, first-hit rate | **0.9722** |
| Image → image, recall@10 | **0.9091** |

The raw report is `eval/baseline/public-baseline-ort-cpu.json`.

### Known miss

`q07/zh` expects three hits but only two are inside the top 10; the first hit is
at rank 1. This is a property of the model + the public set, not a regression
baseline — it is recorded here so later runs can be compared against it.

## Comparing a later run

Re-run the command above with a different `--snapshot` path and diff the
`text2image.recall`, `text2image.first_hit_rate` and `image2image.recall`
fields. A drop means the vector semantics changed — which, per the versioning
rules, requires a **MAJOR** release and the consistency check in
[`../../docs/EVALUATION.md`](../../docs/EVALUATION.md).

## Consistency golden

`eval/baseline/selfcheck-public.ort-cpu.json` holds the vectors of the public
self-check suite (25 items: 20 texts + 5 images) for the ORT-CPU backend, with
the sha256 of every input.

```bash
cargo run --release -p embed_core --example embed_consistency -- check \
  --suite eval/suites/selfcheck.json --image_dir eval/public/images \
  --ort_lib <ONNX_RUNTIME_LIB> --model_dir <EMBEDDINGGEMMA2_MODEL_DIR> \
  --golden eval/baseline/selfcheck-public.ort-cpu.json \
  --min_cos 0.9999 --max_abs 1e-6
```

Recorded result at the time of writing: `n=25 min_cos=1.000000
max_abs_diff=5.551e-17` → **PASS** (same backend, element-wise equal).
