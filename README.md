# local-media-ai

Reusable, app-agnostic **local multimodal AI** building blocks in Rust:

- **Embedding** of text and images with EmbeddingGemma 2 (ONNX Runtime, CPU),
- **OCR** of images (PaddleOCR-style PP-OCRv5 ONNX models),
- **Video keyframe extraction** through an external `ffmpeg`,
- **A backend router** that supervises a `llama.cpp` worker process (stdio frames).

Everything here is independent of any application: the crates take file paths or
text and return plain values. No database, no UI, no app-specific concepts.

> This repository was extracted from the File Organizer project
> (https://github.com/joopererer/file_organizer, Apache-2.0). The git history is
> preserved, so the provenance of every change stays visible. See `NOTICE`.

## License

Apache-2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
Third-party components (ONNX Runtime, llama.cpp, EmbeddingGemma 2, PaddleOCR, …)
are listed in [`docs/THIRD_PARTY_LICENSES.md`](docs/THIRD_PARTY_LICENSES.md).

**No model weights and no runtime binaries are stored in this repository.**
The scripts under [`scripts/`](scripts) download them on demand and verify their
sha256 checksums.

## Crates

| Crate | What it does |
|---|---|
| [`embed_core`](crates/embed_core) | EmbeddingGemma 2 inference (text + image → 768-d L2-normalised vector), image preprocessing, and the stdio frame protocol shared with the worker |
| [`embed_router`](crates/embed_router) | Backend selection (in-process ORT vs. llama.cpp worker) and worker process supervision: handshake, timeouts, crash backoff, circuit breaker, log rotation |
| [`ocr_core`](crates/ocr_core) | Image → text + line boxes, engine-replaceable (PP-OCRv5 ONNX) |
| [`video_frames`](crates/video_frames) | Video keyframe extraction via an external `ffmpeg` |
| [`llama_worker`](crates/llama_worker) | The `llama-embed-worker` binary: a llama.cpp embedding server speaking the stdio frame protocol |

## Using it from another project

Pin to a tag (Cargo matches tags exactly — they never move):

```toml
[dependencies]
embed_core   = { git = "https://github.com/joopererer/local-media-ai.git", tag = "v0.1.0" }
embed_router = { git = "https://github.com/joopererer/local-media-ai.git", tag = "v0.1.0" }
ocr_core     = { git = "https://github.com/joopererer/local-media-ai.git", tag = "v0.1.0" }
video_frames = { git = "https://github.com/joopererer/local-media-ai.git", tag = "v0.1.0" }
```

Build the worker binary from a checkout of the same tag:

```bash
git clone --branch v0.1.0 --depth 1 https://github.com/joopererer/local-media-ai.git
cargo build --release --manifest-path local-media-ai/Cargo.toml -p llama_worker
# -> local-media-ai/target/release/llama-embed-worker
```

## Getting the models and runtimes

```powershell
# ONNX Runtime (dynamic library; loaded from an absolute path at runtime)
./scripts/fetch_onnxruntime.ps1 -Platform windows-x64 -OutDir ./runtime/ort

# EmbeddingGemma 2 (ONNX export, quantized)
./scripts/fetch_embedding_models.ps1 -OutDir ./runtime/models

# llama.cpp runtime + import libraries (only needed for the llama.cpp backend)
./scripts/fetch_llama_runtime.ps1 -OutDir ./runtime/llama
```

Each script verifies the sha256 of everything it downloads. The pinned
checksums live next to the scripts in `scripts/checksums/`.

## Building and testing

```bash
cargo build --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

Tests that need a real model or runtime are `#[ignore]`d and driven by
environment variables (`FO_ORT_LIB`, `FO_MODEL_DIR`, `FO_TEST_IMAGE`); see
[`docs/EVALUATION.md`](docs/EVALUATION.md).

## Versioning

Semantic versioning. **Any change to the vector semantics** — dimensionality,
normalisation, preprocessing, pooling, prompt prefix — is a **MAJOR** change,
because previously stored vectors are not comparable with new ones. Upgrading
llama.cpp or the model must pass the consistency check described in
[`docs/EVALUATION.md`](docs/EVALUATION.md) (same backend: element-wise equal
within `1e-6`; cross backend: cosine ≥ 0.99).

### Released tags are immutable

A pushed `v*` tag is never moved and never deleted. Downstream projects pin to a
tag and `Cargo.lock` records the exact commit, so moving a tag changes what
everyone builds without changing the version number. Fixes always ship as the
next revision (`v0.1.1`, `v0.1.2`, …). Do not depend on a branch.

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the full rule and the release
checklist.

## Repository layout

```
crates/          the Rust crates
docs/            third-party licences, evaluation notes
eval/            public evaluation set (CC0 / public-domain images only)
scripts/         model & runtime download scripts with sha256 verification
```

## License headers

Contributions keep the existing Apache-2.0 header or add one:

```rust
// Copyright 2026 Jiawei LI
// Licensed under the Apache License, Version 2.0.
```

## Support

If this project saves you time, you can support its development:

- [GitHub Sponsors](https://github.com/sponsors/joopererer)
- [Buy Me a Coffee](https://www.buymeacoffee.com/joopererer)

Sponsorship is optional. The crates stay Apache-2.0 and the model/runtime
scripts stay free to use either way.
