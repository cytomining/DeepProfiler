from comet_ml import Experiment
import importlib
import os
import pytest
import keras
import numpy as np
import random
import pandas as pd

import deepprofiler.imaging.cropping
import deepprofiler.dataset.image_dataset
import deepprofiler.dataset.metadata
import deepprofiler.dataset.target
import plugins.models.gan


def __rand_array():
    return np.array(random.sample(range(100), 12))


@pytest.fixture(scope='function')
def out_dir(tmpdir):
    return os.path.abspath(tmpdir.mkdir('test'))


@pytest.fixture(scope='function')
def config(out_dir):
    return {
        "model": {
            "name": "gan",
            "crop_generator": "autoencoder_crop_generator",
            "feature_dim": 128,
            "latent_dim": 128,
            "conv_blocks": 3,
            "params": {
                "epochs": 3,
                "steps": 10,
                "learning_rate": 0.0002,
                "batch_size": 16
            },
        },
        "sampling": {
            "images": 12,
            "box_size": 16,
            "locations": 10,
            "locations_field": 'R'
        },
        "image_set": {
            "channels": ['R', 'G', 'B'],
            "mask_objects": False,
            "width": 128,
            "height": 128,
            "path": out_dir
        },
        "training": {
            "learning_rate": 0.001,
            "output": out_dir,
            "epochs": 2,
            "steps": 12,
            "minibatch": 2,
            "visible_gpus": "0"
        },
        "queueing": {
            "loading_workers": 2,
            "queue_size": 2
        },
        "validation": {
            "minibatch": 2,
            "output": out_dir,
            "api_key":'[REDACTED]',
            "project_name":'pytests',
            "frame":"train",
            "sample_first_crops": True,
            "top_k": 2
        }
    }


@pytest.fixture(scope='function')
def metadata(out_dir):
    filename = os.path.join(out_dir, 'metadata.csv')
    df = pd.DataFrame({
        'Metadata_Plate': __rand_array(),
        'Metadata_Well': __rand_array(),
        'Metadata_Site': __rand_array(),
        'R': [str(x) + '.png' for x in __rand_array()],
        'G': [str(x) + '.png' for x in __rand_array()],
        'B': [str(x) + '.png' for x in __rand_array()],
        'Class': ['0', '1', '2', '3', '0', '1', '2', '3', '0', '1', '2', '3'],
        'Sampling': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        'Split': [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    }, dtype=int)
    df.to_csv(filename, index=False)
    meta = deepprofiler.dataset.metadata.Metadata(filename)
    train_rule = lambda data: data['Split'].astype(int) == 0
    val_rule = lambda data: data['Split'].astype(int) == 1
    meta.splitMetadata(train_rule, val_rule)
    return meta


@pytest.fixture(scope='function')
def dataset(metadata, out_dir):
    keygen = lambda r: "{}/{}-{}".format(r["Metadata_Plate"], r["Metadata_Well"], r["Metadata_Site"])
    dset = deepprofiler.dataset.image_dataset.ImageDataset(metadata, 'Sampling', ['R', 'G', 'B'], out_dir, keygen)
    target = deepprofiler.dataset.target.MetadataColumnTarget('Class', metadata.data['Class'].unique())
    dset.add_target(target)
    return dset


@pytest.fixture(scope='function')
def generator():
    return deepprofiler.imaging.cropping.CropGenerator


@pytest.fixture(scope='function')
def val_generator():
    return deepprofiler.imaging.cropping.SingleImageCropGenerator


def test_gan(config, generator, val_generator):
    gan = plugins.models.gan.GAN(config, generator, val_generator)
    assert gan.config == config
    assert gan.crop_generator == generator
    assert gan.val_crop_generator == val_generator
    assert gan.img_cols == config["sampling"]["box_size"]
    assert gan.img_rows == config["sampling"]["box_size"]
    assert gan.channels == len(config["image_set"]["channels"])
    assert gan.img_shape == (
        config["sampling"]["box_size"],
        config["sampling"]["box_size"],
        len(config["image_set"]["channels"])
    )
    assert gan.latent_dim == config["model"]["latent_dim"]
    assert isinstance(gan.generator, keras.Model)
    assert isinstance(gan.discriminator, keras.Model)
    assert isinstance(gan.combined, keras.Model)
    assert gan.generator in gan.combined.layers
    assert gan.discriminator in gan.combined.layers
    assert not gan.discriminator.trainable


def test_init(config, dataset, generator, val_generator):
    dpmodel = plugins.models.gan.ModelClass(config, dataset, generator, val_generator)
    gan = plugins.models.gan.GAN(config, generator, val_generator)
    assert dpmodel.gan.__eq__(gan)
