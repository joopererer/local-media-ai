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

## [Unreleased]

### Added
- Initial public release: `embed_core`, `embed_router`, `ocr_core`,
  `video_frames` and the `llama_worker` crate (binary `llama-embed-worker`),
  extracted from File Organizer with the git history preserved.

### Changed
- The worker binary is named `llama-embed-worker` (previously `fo-llama-worker`);
  the crate name stays `llama_worker`.
- `embed_router` is a standalone crate; it used to live inside the application
  binding layer.
