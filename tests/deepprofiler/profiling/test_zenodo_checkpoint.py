"""Zenodo model download and usage integration test.

Downloads and runs the Cell Painting CNN v1 checkpoint from Zenodo.
Marked with @pytest.mark.integration — skipped by default.
Run with: uv run pytest -m integration
"""
import os
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


@pytest.fixture(scope="module")
def checkpoint_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("checkpoint") / CHECKPOINT_FILENAME
    print("Downloading Cell Painting CNN v1 checkpoint from Zenodo...")
    urllib.request.urlretrieve(ZENODO_URL, path)
    return str(path)


CHANNELS = ["DNA", "ER", "RNA", "AGP", "Mito"]
BOX_SIZE = 128


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

    import shutil
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

    # Write synthetic 5-channel images (128×128, uint16)
    images_dir = zenodo_config["paths"]["images"]
    for ch_file in channel_files.values():
        img = np.random.randint(0, 65535, (BOX_SIZE, BOX_SIZE), dtype=np.uint16)
        skimage.io.imsave(os.path.join(images_dir, ch_file), img)

    # Write metadata CSV
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

    # Write locations CSV  (one cell at image centre)
    loc_dir = os.path.join(zenodo_config["paths"]["locations"], str(plate))
    os.makedirs(loc_dir, exist_ok=True)
    loc_path = os.path.join(loc_dir, f"{well}-{site}-Nuclei.csv")
    locs = pd.DataFrame({
        "Nuclei_Location_Center_X": [BOX_SIZE // 2],
        "Nuclei_Location_Center_Y": [BOX_SIZE // 2],
    })
    locs.to_csv(loc_path, index=False)

    # Build ImageDataset
    meta = deepprofiler.dataset.metadata.Metadata(index_path)
    def keygen(r):
        return "{}/{}-{}".format(r["Metadata_Plate"], r["Metadata_Well"], r["Metadata_Site"])
    dset = deepprofiler.dataset.image_dataset.ImageDataset(
        meta, "Class", CHANNELS, zenodo_config["paths"]["images"], keygen, zenodo_config
    )
    target = deepprofiler.dataset.target.MetadataColumnTarget("Class", meta.data["Class"].unique())
    dset.add_target(target)
    return dset


@pytest.mark.integration
def test_checkpoint_loads(zenodo_config):
    """Verify the Zenodo checkpoint can be loaded into the EfficientNet B0 architecture."""
    model = deepprofiler.profiling.build_model(zenodo_config)
    checkpoint = os.path.join(
        zenodo_config["paths"]["checkpoints"], CHECKPOINT_FILENAME
    )
    model.load_weights(checkpoint, by_name=True)
    assert model is not None


@pytest.mark.integration
def test_checkpoint_produces_features(zenodo_config):
    """Verify the loaded checkpoint produces non-trivial feature vectors."""
    model = deepprofiler.profiling.build_model(zenodo_config)
    checkpoint = os.path.join(
        zenodo_config["paths"]["checkpoints"], CHECKPOINT_FILENAME
    )
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
    """Run the full Profile pipeline end-to-end and verify the .npz output."""
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
