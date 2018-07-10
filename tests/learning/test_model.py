import importlib
import os
import random

import keras
import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
import skimage.io

import deepprofiler.dataset.target
import deepprofiler.dataset.metadata
import deepprofiler.dataset.image_dataset
import deepprofiler.imaging.cropping
from deepprofiler.learning.model import DeepProfilerModel


def __rand_array():
    return np.array(random.sample(range(100), 12))


@pytest.fixture(scope='function')
def out_dir(tmpdir):
    return os.path.abspath(tmpdir.mkdir('test'))


@pytest.fixture(scope='function')
def config(out_dir):
    return {
        "model": {
            "name": "resnet18",
            "crop_generator": "crop_generator"
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
        "validation": {
            "minibatch": 2,
            "output": out_dir
        },
        "queueing": {
            "loading_workers": 2,
            "queue_size": 2
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
        'Sampling': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        'Split': [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        'Target': [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
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
    target = deepprofiler.dataset.target.MetadataColumnTarget('Target', metadata.data['Target'].unique())
    dset.add_target(target)
    return dset


@pytest.fixture(scope='function')
def data(metadata, out_dir):
    images = np.random.randint(0, 256, (128, 128, 36), dtype=np.uint8)
    for i in range(0, 36, 3):
        skimage.io.imsave(os.path.join(out_dir, metadata.data['R'][i // 3]), images[:, :, i])
        skimage.io.imsave(os.path.join(out_dir, metadata.data['G'][i // 3]), images[:, :, i + 1])
        skimage.io.imsave(os.path.join(out_dir, metadata.data['B'][i // 3]), images[:, :, i + 2])


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
def crop_generator(config, dataset):
    return deepprofiler.imaging.cropping.CropGenerator(config, dataset)


@pytest.fixture(scope='function')
def model(config, dataset):
    module = importlib.import_module("plugins.models.{}".format(config['model']['name']))
    importlib.invalidate_caches()
    mdl = module.define_model(config, dataset)
    return mdl


@pytest.fixture(scope='function')
def deep_profiler_model(model, config, dataset, crop_generator):
    return DeepProfilerModel(model, config, dataset, crop_generator)


def test_init(model, config, dataset, crop_generator):
    dpmodel = DeepProfilerModel(model, config, dataset, crop_generator)
    assert dpmodel.model == model
    assert dpmodel.config == config
    assert dpmodel.dset == dataset
    assert dpmodel.crop_generator == crop_generator
    assert dpmodel.random_seed is None


def test_seed(deep_profiler_model):
    seed = random.randint(0, 256)
    deep_profiler_model.seed(seed)
    assert deep_profiler_model.random_seed == seed


def test_train(deep_profiler_model, out_dir, data, locations):
    epoch = 1
    deep_profiler_model.train(epoch)
    assert os.path.exists(os.path.join(out_dir, "checkpoint_0001.hdf5"))
    assert os.path.exists(os.path.join(out_dir, "checkpoint_0002.hdf5"))
    assert os.path.exists(os.path.join(out_dir, "log.csv"))
    epoch = 3
    deep_profiler_model.config['training']['epochs'] = 4
    deep_profiler_model.train(epoch)
    assert os.path.exists(os.path.join(out_dir, "checkpoint_0003.hdf5"))
    assert os.path.exists(os.path.join(out_dir, "checkpoint_0004.hdf5"))
    assert os.path.exists(os.path.join(out_dir, "log.csv"))
