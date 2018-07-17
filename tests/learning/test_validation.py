import os
import random

import numpy as np
import pandas as pd
import pytest
import skimage.io
import tensorflow as tf

import deepprofiler.dataset.image_dataset
import deepprofiler.dataset.metadata
import deepprofiler.dataset.target
import deepprofiler.imaging.cropping
import deepprofiler.learning.training
import deepprofiler.learning.validation


def __rand_array():
    return np.array(random.sample(range(100), 12))


@pytest.fixture(scope='function')
def out_dir(tmpdir):
    return os.path.abspath(tmpdir.mkdir("test_validation"))


@pytest.fixture(scope='function')
def config(out_dir):
    return {
        "model": {
            "name": "cnn",
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
            "epochs": 0,
            "steps": 12,
            "minibatch": 2
        },
        "validation": {
            "minibatch": 2,
            "save_features": True,
            "sample_first_crops": False,
            "frame": "val",
            "top_k": 2
        },
        "queueing": {
            "loading_workers": 2,
            "queue_size": 2,
            "min_size": 0
        },
        "profiling": {
            "feature_layer": "pool5"  # TODO: make this work with any model
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
def session():
    configuration = tf.ConfigProto()
    configuration.gpu_options.visible_device_list = "0"
    session = tf.Session(config = configuration)
    return session


@pytest.fixture(scope='function')
def crop_generator(config, dataset, session):
    crop_generator = deepprofiler.imaging.cropping.SingleImageCropGenerator(config, dataset)
    crop_generator.start(session)
    return crop_generator


@pytest.fixture(scope='function')
def validation(config, dataset, crop_generator, session):
    return deepprofiler.learning.validation.Validation(config, dataset, crop_generator, session)


def test_init(config, dataset, crop_generator, session, validation):
    validation = validation
    config["queueing"]["min_size"] = 0
    assert validation.config == config
    assert validation.dset == dataset
    assert validation.crop_generator == crop_generator
    assert validation.session == session
    assert validation.batch_inputs == []
    assert validation.batch_outputs == []


def test_process_batches():  # tested in test_validate
    pass


def test_validate(config, dataset, crop_generator, session, out_dir, data, locations, target):
    test_images, test_labels = deepprofiler.learning.validation.validate(config, dataset, crop_generator, session)
    assert test_labels.shape == (60,4)
    assert test_images.shape == (60,16,16,3)
    test_labels_amax = np.amax(test_labels, axis=1)
    test_labels_amax_sum = 0
    for term in test_labels_amax:
        test_labels_amax_sum += term
    assert test_labels_amax_sum == 60
