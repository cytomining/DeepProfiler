"""Integration tests for the Cell Painting CNN v1 checkpoint.

The Cell Painting CNN v1 is a pre-trained EfficientNet B0 model that extracts
1280-dimensional feature embeddings from 5-channel Cell Painting microscopy
images (DNA, ER, RNA, AGP, Mito).
It was trained on 490 compound classes and its weights are hosted on Zenodo
(DOI: 10.5281/zenodo.7114558).
This file tests that the checkpoint can be downloaded, loaded, and used to
profile real and synthetic images end-to-end.

Test structure
--------------
The checkpoint is downloaded once per test session via the ``checkpoint_path``
fixture (``scope="module"``), then copied into each test's temporary project
directory.
Tests are split into two groups:

**Synthetic data tests** (``zenodo_*`` fixtures):
    Use randomly generated 128×128 uint16 images with a single cell centred in
    the frame.
    These tests are fast and verify structural correctness — that the checkpoint
    loads without error, that the feature extractor produces the expected output
    shape ``(n_cells, 1280)``, and that the full ``Profile`` pipeline writes a
    valid ``.npz`` file.
    They do not verify feature values, because random inputs produce
    non-deterministic outputs.

**Real data tests** (``cpg0000_*`` fixtures):
    Use a single site of actual Cell Painting images from the cpg0000-jump-pilot
    dataset (plate BR00116991, well A01, site 1), committed to
    ``tests/data/cpg0000/``.
    These images are 1080×1080 uint16 TIFFs acquired on a PerkinElmer Phenix
    instrument.
    Cell locations (109 nuclei) come from CellProfiler's ``Nuclei.csv``
    (``AreaShape_Center_X/Y``), converted to the DeepProfiler
    ``Nuclei_Location_Center_X/Y`` format at fixture setup time.
    These tests verify the full pipeline on biologically real data: exact cell
    count, spot-checked feature values for specific cells, and global feature
    statistics (mean and std).
    If weights, preprocessing, or the crop pipeline regress, the spot-checked
    values will drift and the test will fail.

Channel mapping for cpg0000 source_4 (from Index.idx.xml):
    ch5 = DNA  (HOECHST 33342)
    ch4 = ER   (Alexa 488)
    ch3 = RNA  (488 long)
    ch1 = AGP  (Alexa 647)
    ch2 = Mito (Alexa 568)

Marked with @pytest.mark.integration — skipped by default.
Run with: uv run pytest -m integration --override-ini="addopts="
"""

import os
import shutil
import urllib.request

import numpy as np
import pandas as pd
import pytest
import skimage.io
import tensorflow as tf

import deepprofiler.dataset.image_dataset
import deepprofiler.dataset.metadata
import deepprofiler.dataset.target
import deepprofiler.profiling

tf.compat.v1.disable_v2_behavior()

ZENODO_URL = "https://zenodo.org/record/7114558/files/Cell_Painting_CNN_v1.hdf5"
CHECKPOINT_FILENAME = "Cell_Painting_CNN_v1.hdf5"

CHANNELS = ["DNA", "ER", "RNA", "AGP", "Mito"]
BOX_SIZE = 128

# cpg0000 — plate BR00116991, well A01 (r01c01), site 1 (f01)
_CPG0000_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "cpg0000", "BR00116991", "A01-1"
)
CPG0000_PLATE = "BR00116991"
CPG0000_WELL = "A01"
CPG0000_SITE = 1
CPG0000_IMAGE_WIDTH = 1080
CPG0000_IMAGE_HEIGHT = 1080
CPG0000_CHANNEL_FILES = {
    "DNA":  "r01c01f01p01-ch5sk1fk1fl1.tiff",
    "ER":   "r01c01f01p01-ch4sk1fk1fl1.tiff",
    "RNA":  "r01c01f01p01-ch3sk1fk1fl1.tiff",
    "AGP":  "r01c01f01p01-ch1sk1fk1fl1.tiff",
    "Mito": "r01c01f01p01-ch2sk1fk1fl1.tiff",
}


# ---------------------------------------------------------------------------
# Shared fixture — checkpoint downloaded once for the entire module
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def checkpoint_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("checkpoint") / CHECKPOINT_FILENAME
    print("Downloading Cell Painting CNN v1 checkpoint from Zenodo...")
    urllib.request.urlretrieve(ZENODO_URL, path)
    return str(path)


# ---------------------------------------------------------------------------
# Synthetic-data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def zenodo_config(tmp_path_factory, checkpoint_path):
    root = tmp_path_factory.mktemp("project")
    checkpoints_dir = root / "checkpoints"
    checkpoints_dir.mkdir()
    features_dir = root / "features"
    features_dir.mkdir()
    images_dir = root / "images"
    images_dir.mkdir()
    metadata_dir = root / "metadata"
    metadata_dir.mkdir()
    locations_dir = root / "locations"
    locations_dir.mkdir()

    shutil.copy(checkpoint_path, checkpoints_dir / CHECKPOINT_FILENAME)

    return {
        "dataset": {
            "images": {
                "channels": CHANNELS,
                "width": BOX_SIZE,
                "height": BOX_SIZE,
                "bits": 16,
                "file_format": "tif",
            },
            "locations": {
                "mode": "single_cells",
                "box_size": BOX_SIZE,
                "mask_objects": False,
            },
            "metadata": {"label_field": "Class", "control_id": "0"},
        },
        "train": {
            "model": {
                "name": "efficientnet",
                "crop_generator": "crop_generator",
                "params": {"conv_blocks": 0, "batch_size": 2, "learning_rate": 0.0001, "label_smoothing": 0.0},
            },
            "sampling": {"workers": 1, "cache_size": 1, "factor": 1.0, "alpha": 0.2},
            "validation": {"batch_size": 2},
            "partition": {
                "targets": ["Class"],
                "split_field": "Split",
                "training": [0],
                "validation": [1],
            },
        },
        "profile": {
            "feature_layer": "pool5",
            "checkpoint": CHECKPOINT_FILENAME,
            "batch_size": 2,
        },
        "num_classes": 490,
        "paths": {
            "checkpoints": str(checkpoints_dir),
            "features": str(features_dir),
            "images": str(images_dir),
            "metadata": str(metadata_dir),
            "locations": str(locations_dir),
            "root_dir": str(root),
            "index": str(metadata_dir / "index.csv"),
        },
    }


@pytest.fixture(scope="module")
def zenodo_dataset(zenodo_config):
    """Synthetic 5-channel dataset for end-to-end profiling with the Zenodo checkpoint."""
    plate, well, site = 1, 1, 1
    channel_files = {ch: f"img_{ch}.tif" for ch in CHANNELS}

    images_dir = zenodo_config["paths"]["images"]
    for ch_file in channel_files.values():
        img = np.random.randint(0, 65535, (BOX_SIZE, BOX_SIZE), dtype=np.uint16)
        skimage.io.imsave(os.path.join(images_dir, ch_file), img)

    index_path = zenodo_config["paths"]["index"]
    df = pd.DataFrame({
        "Metadata_Plate": [plate],
        "Metadata_Well": [well],
        "Metadata_Site": [site],
        **{ch: [channel_files[ch]] for ch in CHANNELS},
        "Class": ["0"],
        "Split": [0],
    })
    df.to_csv(index_path, index=False)

    loc_dir = os.path.join(zenodo_config["paths"]["locations"], str(plate))
    os.makedirs(loc_dir, exist_ok=True)
    locs = pd.DataFrame({
        "Nuclei_Location_Center_X": [BOX_SIZE // 2],
        "Nuclei_Location_Center_Y": [BOX_SIZE // 2],
    })
    locs.to_csv(os.path.join(loc_dir, f"{well}-{site}-Nuclei.csv"), index=False)

    meta = deepprofiler.dataset.metadata.Metadata(index_path)
    def keygen(r):
        return "{}/{}-{}".format(r["Metadata_Plate"], r["Metadata_Well"], r["Metadata_Site"])
    dset = deepprofiler.dataset.image_dataset.ImageDataset(
        meta, "Class", CHANNELS, zenodo_config["paths"]["images"], keygen, zenodo_config
    )
    target = deepprofiler.dataset.target.MetadataColumnTarget("Class", meta.data["Class"].unique())
    dset.add_target(target)
    return dset


# ---------------------------------------------------------------------------
# cpg0000 real-data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cpg0000_config(tmp_path_factory, checkpoint_path):
    root = tmp_path_factory.mktemp("cpg0000_project")
    checkpoints_dir = root / "checkpoints"
    checkpoints_dir.mkdir()
    features_dir = root / "features"
    features_dir.mkdir()
    metadata_dir = root / "metadata"
    metadata_dir.mkdir()
    locations_dir = root / "locations"
    locations_dir.mkdir()

    shutil.copy(checkpoint_path, checkpoints_dir / CHECKPOINT_FILENAME)

    return {
        "dataset": {
            "images": {
                "channels": CHANNELS,
                "width": CPG0000_IMAGE_WIDTH,
                "height": CPG0000_IMAGE_HEIGHT,
                "bits": 16,
                "file_format": "tiff",
            },
            "locations": {
                "mode": "single_cells",
                "box_size": BOX_SIZE,
                "mask_objects": False,
            },
            "metadata": {"label_field": "Class", "control_id": "0"},
        },
        "train": {
            "model": {
                "name": "efficientnet",
                "crop_generator": "crop_generator",
                "params": {"conv_blocks": 0, "batch_size": 2, "learning_rate": 0.0001, "label_smoothing": 0.0},
            },
            "sampling": {"workers": 1, "cache_size": 1, "factor": 1.0, "alpha": 0.2},
            "validation": {"batch_size": 2},
            "partition": {
                "targets": ["Class"],
                "split_field": "Split",
                "training": [0],
                "validation": [1],
            },
        },
        "profile": {
            "feature_layer": "pool5",
            "checkpoint": CHECKPOINT_FILENAME,
            "batch_size": 2,
        },
        "num_classes": 490,
        "paths": {
            "checkpoints": str(checkpoints_dir),
            "features": str(features_dir),
            "images": _CPG0000_DIR,
            "metadata": str(metadata_dir),
            "locations": str(locations_dir),
            "root_dir": str(root),
            "index": str(metadata_dir / "index.csv"),
        },
    }


@pytest.fixture(scope="module")
def cpg0000_dataset(cpg0000_config):
    """ImageDataset built from committed cpg0000 test images and CellProfiler locations."""
    index_path = cpg0000_config["paths"]["index"]
    df = pd.DataFrame({
        "Metadata_Plate": [CPG0000_PLATE],
        "Metadata_Well": [CPG0000_WELL],
        "Metadata_Site": [CPG0000_SITE],
        **{ch: [CPG0000_CHANNEL_FILES[ch]] for ch in CHANNELS},
        "Class": ["DMSO"],
        "Split": [0],
    })
    df.to_csv(index_path, index=False)

    nuclei = pd.read_csv(os.path.join(_CPG0000_DIR, "Nuclei.csv"))
    locs = pd.DataFrame({
        "Nuclei_Location_Center_X": nuclei["AreaShape_Center_X"],
        "Nuclei_Location_Center_Y": nuclei["AreaShape_Center_Y"],
    })
    loc_dir = os.path.join(cpg0000_config["paths"]["locations"], CPG0000_PLATE)
    os.makedirs(loc_dir, exist_ok=True)
    locs.to_csv(os.path.join(loc_dir, f"{CPG0000_WELL}-{CPG0000_SITE}-Nuclei.csv"), index=False)

    meta = deepprofiler.dataset.metadata.Metadata(index_path)
    def keygen(r):
        return "{}/{}-{}".format(r["Metadata_Plate"], r["Metadata_Well"], r["Metadata_Site"])
    dset = deepprofiler.dataset.image_dataset.ImageDataset(
        meta, "Class", CHANNELS, cpg0000_config["paths"]["images"], keygen, cpg0000_config
    )
    target = deepprofiler.dataset.target.MetadataColumnTarget("Class", meta.data["Class"].unique())
    dset.add_target(target)
    return dset


# ---------------------------------------------------------------------------
# Tests — synthetic data
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_checkpoint_loads(zenodo_config):
    """Verify the Zenodo checkpoint can be loaded into the EfficientNet B0 architecture."""
    model = deepprofiler.profiling.build_model(zenodo_config)
    checkpoint = os.path.join(zenodo_config["paths"]["checkpoints"], CHECKPOINT_FILENAME)
    model.load_weights(checkpoint, by_name=True)
    assert model is not None


@pytest.mark.integration
def test_checkpoint_produces_features(zenodo_config):
    """Verify the loaded checkpoint produces non-trivial feature vectors."""
    model = deepprofiler.profiling.build_model(zenodo_config)
    checkpoint = os.path.join(zenodo_config["paths"]["checkpoints"], CHECKPOINT_FILENAME)
    model.load_weights(checkpoint, by_name=True)

    feat_extractor = tf.keras.Model(
        model.inputs,
        model.get_layer(zenodo_config["profile"]["feature_layer"]).output
    )
    crops = np.random.rand(2, BOX_SIZE, BOX_SIZE, len(CHANNELS)).astype(np.float32)
    features = feat_extractor.predict(crops)

    assert features.shape == (2, 1280)
    assert not np.all(features == 0)


@pytest.mark.integration
def test_profile_pipeline_produces_npz(zenodo_config, zenodo_dataset):
    """Run the full Profile pipeline on synthetic data and verify the .npz output."""
    with tf.compat.v1.Session().as_default():
        prof = deepprofiler.profiling.Profile(zenodo_config, zenodo_dataset)
        prof.configure()

        meta = zenodo_dataset.meta.data.iloc[0]
        image_array = np.random.randint(0, 65535, (BOX_SIZE, BOX_SIZE, len(CHANNELS)), dtype=np.uint16)
        prof.extract_features(None, image_array, meta)

    output_file = os.path.join(
        zenodo_config["paths"]["features"],
        str(meta["Metadata_Plate"]),
        str(meta["Metadata_Well"]),
        f"{meta['Metadata_Site']}.npz",
    )
    assert os.path.isfile(output_file), f"Expected output not found: {output_file}"
    result = np.load(output_file, allow_pickle=True)
    assert "features" in result
    assert result["features"].ndim == 2
    assert result["features"].shape[0] >= 1


# ---------------------------------------------------------------------------
# Tests — real cpg0000 data
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_cpg0000_profile_pipeline_produces_npz(cpg0000_config, cpg0000_dataset):
    """Run the full profiling pipeline on a real cpg0000 site and verify the output.

    Loads real Cell Painting images from tests/data/cpg0000/ and runs them
    through the Cell Painting CNN v1 checkpoint. Verifies that the pipeline
    produces a valid .npz with the expected number of cells and known feature
    values — something synthetic data cannot guarantee.

    Args:
        cpg0000_config: Experiment config pointing at committed test images.
        cpg0000_dataset: ImageDataset built from the committed test data.
    """
    meta = cpg0000_dataset.meta.data.iloc[0]

    channel_arrays = [
        skimage.io.imread(os.path.join(_CPG0000_DIR, CPG0000_CHANNEL_FILES[ch]))
        for ch in CHANNELS
    ]
    image_array = np.stack(channel_arrays, axis=-1)
    assert image_array.shape == (CPG0000_IMAGE_HEIGHT, CPG0000_IMAGE_WIDTH, len(CHANNELS))

    with tf.compat.v1.Session().as_default():
        prof = deepprofiler.profiling.Profile(cpg0000_config, cpg0000_dataset)
        prof.configure()
        prof.extract_features(None, image_array, meta)

    output_file = os.path.join(
        cpg0000_config["paths"]["features"],
        CPG0000_PLATE,
        CPG0000_WELL,
        f"{CPG0000_SITE}.npz",
    )
    assert os.path.isfile(output_file), f"Expected output not found: {output_file}"
    result = np.load(output_file, allow_pickle=True)

    assert "features" in result

    feats = result["features"]
    assert feats.shape == (109, 1280)

    # Spot-check a few feature values to catch regressions in weights or preprocessing
    assert feats[0, :5] == pytest.approx(
        [-0.03095404, -0.00573806, 0.11657361, -0.11692496, 2.05769897], abs=1e-4
    )
    assert feats[1, :5] == pytest.approx(
        [-0.06177977, -0.05242480, 0.55334610, -0.10083556, -0.14802243], abs=1e-4
    )

    # Global statistics should be stable across runs
    assert np.mean(feats) == pytest.approx(0.195951, abs=1e-4)
    assert np.std(feats) == pytest.approx(0.635050, abs=1e-4)
