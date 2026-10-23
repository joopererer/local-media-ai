# Contributing

Thanks for taking a look. This repository is small on purpose: five reusable
crates, no application code, no database, no UI.

## Before you start

- Open an issue for anything larger than a typo or a one-line fix, so we can
  agree on the approach before you spend time on it.
- The crates must stay independent of any application. If a change needs to know
  about a database, a settings store or a UI concept, it does not belong here.

## Checks to run

Every pull request must pass the same gates as CI:

```bash
cargo fmt --all -- --check
cargo build --workspace --all-targets
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps
```

Tests that need a real model or runtime are `#[ignore]`d and driven by
environment variables (`FO_ORT_LIB`, `FO_MODEL_DIR`, `FO_TEST_IMAGE`,
`FO_LLAMA_LIB_DIR`, `FO_FFMPEG_BIN`); see
[`docs/EVALUATION.md`](docs/EVALUATION.md). A worker built without
`FO_LLAMA_LIB_DIR` is a skeleton: the frame protocol works, the embedding call
returns an error.

Contributions keep the existing Apache-2.0 header or add one:

```rust
// Copyright 2026 Jiawei LI
// Licensed under the Apache License, Version 2.0.
```

## Versioning

Semantic versioning. **Any change to the vector semantics** — dimensionality,
normalisation, preprocessing, pooling, prompt prefix — is a **MAJOR** change,
because previously stored vectors are not comparable with new ones. Upgrading
llama.cpp or the model must pass the consistency check in
[`docs/EVALUATION.md`](docs/EVALUATION.md) (same backend: element-wise equal
within `1e-6`; cross backend: cosine ≥ 0.99).

## Released tags are immutable

Once a `v*` tag has been pushed, **it is never moved and never deleted**. Not for
a broken build, not for a documentation typo, not for a "small" fix, and not for
a rewritten history. Downstream projects pin to a tag (`tag = "v0.1.0"` in
`Cargo.toml`) and `Cargo.lock` records the exact commit; a moved tag silently
changes what everyone else builds while the version number stays the same.

Fix forward instead:

1. Land the fix on `main`.
2. Publish the next revision (`v0.1.1`, `v0.1.2`, …) with `CHANGELOG.md` updated.
3. Tell downstream consumers to bump the pin in a normal dependency update.

If a tag is wrong in a way that cannot be fixed forward — for example a leaked
secret or leaked personal data in the published history — stop and open an issue.
Deleting the tag and rewriting the history is a last resort that requires the
maintainer's explicit decision, and the old tag must be documented as withdrawn
in `CHANGELOG.md`. It is never a routine operation.

## Release checklist

1. `CHANGELOG.md` has a dated section for the new version.
2. All checks above pass locally, and CI is green on `main`.
3. `git tag -a vX.Y.Z -m "..."` on the commit CI already passed, then
   `git push origin vX.Y.Z`.
4. Confirm the CI run for the tag is green, and that the `LICENSE` is still
   detected as Apache-2.0 in the repository sidebar.
