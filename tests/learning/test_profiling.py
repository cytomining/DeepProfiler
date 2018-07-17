import importlib
import random
import deepprofiler.learning.profiling
import deepprofiler.dataset.image_dataset
import deepprofiler.dataset.metadata
import deepprofiler.dataset.target
import pytest
import tensorflow as tf
import json
import os
import shutil
import pandas as pd
import numpy as np
import skimage.io


def __rand_array():
    return np.array(random.sample(range(100), 12))


@pytest.fixture(scope='function')
def out_dir(tmpdir):
    return os.path.abspath(tmpdir.mkdir("test_profiling"))


@pytest.fixture(scope='function')
def config(out_dir):
    return {
        "model": {
            "name": "cnn",
            "crop_generator": "crop_generator",
            "feature_dim": 128,
            "conv_blocks": 3,
            "params": {
                "epochs": 3,
                "steps": 10,
                "learning_rate": 0.0001,
                "batch_size": 16
            }
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
        },
        "profiling": {
            "feature_layer": "features",
            "output_dir": out_dir,
            "checkpoint": None,
            "gpu": "0"
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
def target():
    return deepprofiler.dataset.target.MetadataColumnTarget("Class", ["0", "1", "2", "3"])


@pytest.fixture(scope='function')
def dataset(metadata, target, out_dir):
    keygen = lambda r: "{}/{}-{}".format(r["Metadata_Plate"], r["Metadata_Well"], r["Metadata_Site"])
    dset = deepprofiler.dataset.image_dataset.ImageDataset(metadata, 'Sampling', ['R', 'G', 'B'], out_dir, keygen)
    dset.add_target(target)
    return dset


@pytest.fixture(scope='function')
def locations(out_dir, metadata, config):
    for i in range(len(metadata.data.index)):
        meta = metadata.data.iloc[i]
        path = os.path.join(out_dir, meta['Metadata_Plate'], 'locations')
        os.makedirs(path, exist_ok=True)
        path = os.path.abspath(os.path.join(path, '{}-{}-{}.csv'.format(meta['Metadata_Well'],
                                                  meta['Metadata_Site'],
                                                  config['sampling']['locations_field'])))
        locs = pd.DataFrame({
            'R_Location_Center_X': np.random.randint(0, 128, (config['sampling']['locations'])),
            'R_Location_Center_Y': np.random.randint(0, 128, (config['sampling']['locations']))
        })
        locs.to_csv(path, index=False)


@pytest.fixture(scope='function')
def data(metadata, out_dir):
    images = np.random.randint(0, 256, (128, 128, 36), dtype=np.uint8)
    for i in range(0, 36, 3):
        skimage.io.imsave(os.path.join(out_dir, metadata.data['R'][i // 3]), images[:, :, i])
        skimage.io.imsave(os.path.join(out_dir, metadata.data['G'][i // 3]), images[:, :, i + 1])
        skimage.io.imsave(os.path.join(out_dir, metadata.data['B'][i // 3]), images[:, :, i + 2])


@pytest.fixture(scope='function')
def profile(config, dataset):
    return deepprofiler.learning.profiling.Profile(config, dataset)


def test_crop_transform():
    test_crop_ph = tf.placeholder(tf.float32, shape=(16,16,16,10))
    test_image_size = 8
    test_rgb_data = deepprofiler.learning.profiling.crop_transform(test_crop_ph, test_image_size)
    assert test_rgb_data.shape == (160,8,8,3)


def test_init(config, dataset):
    prof = deepprofiler.learning.profiling.Profile(config, dataset)
    test_num_channels = len(config["image_set"]["channels"])
    assert prof.config == config
    assert prof.dset == dataset
    assert prof.num_channels == test_num_channels
    assert prof.crop_generator == importlib.import_module(
        "plugins.crop_generators.{}".format(config['model']['crop_generator'])).GeneratorClass
    assert isinstance(prof.profile_crop_generator, importlib.import_module(
            "plugins.crop_generators.{}".format(config['model']['crop_generator'])).SingleImageGeneratorClass)
    assert isinstance(prof.dpmodel, importlib.import_module("plugins.models.{}".format(config['model']['name'])).ModelClass)

    # tf.assert_equal(profile.raw_crops, tf.placeholder(tf.float32, shape=(None, 92, 92, 3)))


def test_check(profile, metadata):
    assert profile.check(metadata.data)


# TODO: make a pre-trained model publically available online and add tests for extract_features, configure_resnet, and profile using that model
