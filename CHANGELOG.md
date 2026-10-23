# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Vector compatibility

Changes marked **vector-breaking** change the meaning of the produced vectors
(dimension, normalisation, preprocessing, pooling, prompt prefix). They are
always released as a **MAJOR** version, because vectors stored by earlier
versions are not comparable with the new ones.

Upgrading llama.cpp or the EmbeddingGemma 2 export additionally requires the
consistency check to pass — see [`docs/EVALUATION.md`](docs/EVALUATION.md).

> **Immutability note.** The `v0.1.1` and `v0.1.2` tags existed only briefly
> before the first public push and were deleted when `v0.1.0` was moved to the
> CI-green revision. This was a one-time event during the initial publication,
> before any downstream consumer pinned to those tags. As of `v0.1.0`, released
> tags are immutable — see [`CONTRIBUTING.md`](CONTRIBUTING.md).

## [0.1.0] - 2026-10-23

### Added
- Initial public release: `embed_core`, `embed_router`, `ocr_core`,
  `video_frames` and the `llama_worker` crate (binary `llama-embed-worker`),
  extracted from File Organizer with the git history preserved.

### Changed
- The worker binary is named `llama-embed-worker` (previously `fo-llama-worker`);
  the crate name stays `llama_worker`.
- `embed_router` is a standalone crate; it used to live inside the application
  binding layer.
- `embed_core::worker_frame::MAX_FRAME` is public, so downstream code can share
  the frame size limit instead of hard-coding it.

### Fixed
- CI: the `retrieval_eval` example trips the `chunks_exact` lint of newer clippy
  releases; it now uses `as_chunks::<4>()`.
- CI: the documentation job asked rustup for a `rustdoc` component, which is not
  separately installable on stable, and then failed on intra-doc links that
  looked like arrays (`[0,1]`, `[x0,y0,x1,y1]`).

