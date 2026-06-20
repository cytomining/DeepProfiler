# DeepProfiler Roadmap

DeepProfiler is being redesigned from a training-and-profiling framework into a **lean, pip-installable feature extractor** for microscopy images.
This document describes where the project is heading and what will be removed along the way.

## Vision

DeepProfiler's new focus is a single, well-defined job:

> **Take microscopy images (with optional cell masks) → extract deep learning features → output cytotable-compatible Parquet files.**

The package will be a thin layer over deep learning models, with seamless integration with models hosted on [HuggingFace](https://huggingface.co/), and outputs that follow [cytotable](https://github.com/cytomining/CytoTable) standards for downstream compatibility with [pycytominer](https://github.com/cytomining/pycytominer) and the broader cytomining ecosystem.

---

## Milestones

### ✅ v0.3.x — Software gardening
- Migrate to `pyproject.toml` and `uv`
- Publish to PyPI
- Add CI/CD via GitHub Actions
- Adopt cytomining community standards (CONTRIBUTING, CODE_OF_CONDUCT, CITATION.cff)
- Deprecation notices for functionality to be removed

### ✅ v0.5.1 — Maintenance release (current)
- Remove model training, plugin system, and CometML integration
- Simplify codebase to a single supported use case: feature extraction with the [Cell Painting CNN v1](https://doi.org/10.5281/zenodo.7114558) checkpoint
- Migrate to `uv`, dynamic versioning, mypy, ruff
- CI/CD with PyPI trusted publishing
- Integration test suite: downloads the Zenodo checkpoint and validates the full profiling pipeline end-to-end (`uv run pytest -m integration`)
- **Constraint:** requires Python 3.10–3.11 and TensorFlow 2.10–2.15 (Keras 2); see [README](README.md)

### 🔜 v0.6.x — PyTorch rewrite (featurizer)
- Full rewrite in PyTorch (drop TensorFlow)
- Clean public API: `DeepProfiler.from_pretrained(...)` → `.profile(...)`
- Native HuggingFace model support (`transformers`, `timm`)
- Cell masking support: per-cell crops extracted from masks
- Cytotable-compatible Parquet output (`Metadata_*` + feature columns)
- Optimized inference: `torch.compile`, mixed precision (`torch.autocast`), multi-worker DataLoader
- Python 3.12+ support

### 🔜 v0.7.x — Input flexibility
- Multiple input sources: numpy arrays, file paths, metadata CSVs, Parquet manifests
- Streaming large datasets without loading into memory

### 🔜 v1.0.0 — Stable release
- Stable public API
- Full documentation
- Pre-trained cytomining model weights hosted on HuggingFace

---

## What is being removed

| Functionality | Status | Removed in |
|---|---|---|
| Model training (`train`, `traintf2` commands) | 🚫 Removed | v0.5.1 |
| Single-cell export for training (`export-sc` command) | 🚫 Removed | v0.5.1 |
| Plugin system (models, crop generators, metrics) | 🚫 Removed | v0.5.1 |
| CometML experiment tracking | 🚫 Removed | v0.5.1 |
| TensorFlow backend | ⚠️ Deprecated | v0.6.x |

---

## Licensing transition

DeepProfiler was originally created at the Broad Institute in 2018 and is distributed under the BSD 3-Clause License.
Ongoing development and stewardship has moved to the Cytomining community.

The project is distributed under a single BSD 3-Clause License with two copyright holders, reflecting its origins and ongoing stewardship:

```
Copyright (c) 2018, Broad Institute, Inc.
Copyright (c) 2026, Cytomining
```

---

## Agentic AI

This project is developed with the help of agentic AI tools and is being designed to be usable by AI agents in automated profiling workflows.
See [AGENTS.md](AGENTS.md) for the full plan — covering both agents that help build this codebase and agents that use DeepProfiler as a tool.

---

## Feedback

If you depend on any functionality listed as deprecated, please [open an issue](https://github.com/cytomining/DeepProfiler/issues) to let us know.
We want to understand active use cases before removing anything.
