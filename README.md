![DeepProfiler](figures/logo/banner.png)
-----------------
[![Python 3.10–3.11](https://img.shields.io/badge/python-3.10%20|%203.11-blue)](https://www.python.org/downloads/)
[![CI](https://github.com/cytomining/DeepProfiler/actions/workflows/integration-test.yml/badge.svg)](https://github.com/cytomining/DeepProfiler/actions/workflows/integration-test.yml)
[![codecov](https://codecov.io/gh/cytomining/DeepProfiler/branch/main/graph/badge.svg)](https://codecov.io/gh/cytomining/DeepProfiler)
[![Cell Painting CNN-1 DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7114558.svg)](https://doi.org/10.5281/zenodo.7114558)
[![Example data DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7515132.svg)](https://doi.org/10.5281/zenodo.7515132)

> [!IMPORTANT]
> **v0.5.1 is a focused maintenance release.** Model training, the plugin system, and CometML
> integration have been removed. The only supported use case is **feature extraction using the
> [Cell Painting CNN v1 checkpoint](https://doi.org/10.5281/zenodo.7114558)** (EfficientNet B0).
> This release requires **Python 3.10–3.11** and **TensorFlow 2.10–2.15**. If you need training
> or used the plugin system, please use the [`v0.3.0` tag](https://github.com/cytomining/DeepProfiler/tree/v0.3.0)
> and [open an issue](https://github.com/cytomining/DeepProfiler/issues) to let us know your use case.
> See [ROADMAP.md](ROADMAP.md) for the full plan.

# Image-based profiling using deep learning

DeepProfiler is a set of tools to use deep learning for analyzing imaging data in high-throughput biological experiments.
Please, see our [DeepProfiler Handbook](https://cytomining.github.io/DeepProfiler-handbook/) for more details about how 
to use it and [DeepProfilerExperiments repository](https://github.com/broadinstitute/DeepProfilerExperiments) 
for the examples of configuration files and downstream analysis.

Checkout our Nature Communications [paper](https://www.nature.com/articles/s41467-024-45999-1).

# Cell Painting CNN

<p align="center">
<img src="figures/cell_painting_cnn.png" width="650"/>
</p>

[_**Cell Painting CNN weights are available on Zenodo.**_](https://doi.org/10.5281/zenodo.7114558)

We used DeepProfiler to train a feature extraction model for single cells in Cell Painting experiments. 
The model brings state-of-the-art profiling performance for downstream analysis tasks. This model is an EfficientNet 
trained to process the 5 channels of the Cell Painting assay and produce single-cell morphology embeddings, which can 
be aggregated to profile treatments in large-scale experiments. Features obtained with the Cell Painting CNN are more 
robust and improve performance.

<p align="center">
<img src="figures/cell_painting_cnn_perf.png" width="350"/>
</p>

# Quick Guide

## System requirements

- Python 3.10 or 3.11 (Python 3.12+ is not yet supported — see [ROADMAP.md](ROADMAP.md))
- TensorFlow 2.10–2.15 (TF 2.16+ ships with Keras 3 which is not yet compatible)
- Linux (Ubuntu 20.04+) recommended; macOS arm64 (Apple Silicon) is not supported for this release due to TensorFlow version constraints
- For GPU acceleration, a CUDA-compatible GPU is recommended

## Install

```
pip install deepprofiler
```

Or run directly without any environment setup using [uvx](https://docs.astral.sh/uv/guides/tools/) — it handles installation automatically in an isolated environment:

```
uvx deepprofiler --root=/path/to/project --config=config.json profile
```

For contributing, see [CONTRIBUTING.md](CONTRIBUTING.md).

## Download example data

This repository contains example data structured as a DeepProfiler project.
Unpack it with:
```
tar -xzf example_data.tar.gz
```

## Profiling with the Cell Painting CNN-1

The only supported use case in v0.5.1 is feature extraction using the [Cell Painting CNN v1](https://doi.org/10.5281/zenodo.7114558) checkpoint — an EfficientNet B0 trained on 5-channel Cell Painting images (DNA, ER, RNA, AGP, Mito).

**How inference works:**

1. DeepProfiler reads a metadata CSV listing your images and a locations CSV with per-image cell coordinates (e.g. from CellProfiler nucleus segmentation).
2. For each image, it crops a fixed-size patch around each cell centroid.
3. The crops are passed through the EfficientNet B0 backbone; the `GlobalAveragePooling2D` layer (`pool5`) produces a 1280-dimensional embedding per cell.
4. Embeddings are written to `.npz` files (one per image) containing a `features` array of shape `(num_cells, 1280)` alongside metadata and crop coordinates.

These per-cell `.npz` files can be aggregated with [pycytominer](https://github.com/cytomining/pycytominer) for downstream analysis.

**Setup:**

Initialize your project directory structure:
```
deepprofiler --root=/path/to/project --config=config.json setup
```

Place your images, metadata CSV, and cell locations in the created directories
(see the [handbook](https://cytomining.github.io/DeepProfiler-handbook/docs/02-structure.html) for layout details).
[Download an example configuration file](https://github.com/broadinstitute/DeepProfilerExperiments/blob/master/resources/config/cell_painting_cnn_profiling_example.json)
and put it in `project/inputs/config/`.

Copy the model weights (`Cell_Painting_CNN_v1.hdf5`,
[available on Zenodo](https://doi.org/10.5281/zenodo.7114558)) into `project/outputs/cell_painting/checkpoint/`.

Run feature extraction:
```
deepprofiler --root=/path/to/project --config=cell_painting_cnn.json --exp=cell_painting --gpu=0 profile
```

Extracted features are written to `project/outputs/cell_painting/features/`.

## Verifying your installation

After installing, you can verify that the Cell Painting CNN checkpoint loads and produces features by running the integration test suite.
This downloads the checkpoint from Zenodo (~80 MB) and runs a full end-to-end profiling pipeline on synthetic data:

```
uv run pytest -m integration -v
```

The integration tests check three things:
1. The Zenodo checkpoint loads into the EfficientNet B0 architecture without error.
2. The loaded model produces non-trivial feature vectors for random input crops.
3. The full `Profile` pipeline (checkpoint load → crop generation → feature extraction) writes a valid `.npz` output file.

Integration tests are excluded from the default test run (`uv run pytest`) to avoid network access in CI.

## Training your own models

> **🚫 Removed in v0.5.1:** Model training (`train`, `traintf2`, `export-sc` commands) has been removed. If you need training, use the [`v0.3.0` tag](https://github.com/cytomining/DeepProfiler/tree/v0.3.0). A PyTorch-based training pipeline is planned for v0.6.x.

## Plugin system

> **🚫 Removed in v0.5.1:** The plugin system for models, crop generators, and metrics has been removed.

## CometML experiment tracking

> **🚫 Removed in v0.5.1:** CometML integration has been removed.

**Happy profiling!**
