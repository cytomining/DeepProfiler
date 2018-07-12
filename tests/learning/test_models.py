from comet_ml import Experiment
import keras
import numpy as np
import pytest
import tensorflow as tf
import os

import deepprofiler.dataset.target
import deepprofiler.learning.models


@pytest.fixture(scope='function')
def targets():
    return [
        deepprofiler.dataset.target.MetadataColumnTarget("0", np.random.uniform(0, 1, 10))
    ]

@pytest.fixture(scope='function')
def out_dir(tmpdir):
    return os.path.abspath(tmpdir.mkdir("test_models"))

@pytest.fixture(scope='function')
def config(out_dir):
    return {
        "model": {
            "type": "convnet"
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
            "minibatch": 2
        },
        "queueing": {
            "loading_workers": 2,
            "queue_size": 2
        },
        "validation": {
            "api_key":'rDrWV4m8ITk0PGyDDKWjEgS2q',
            "project_name":'pytests',
            "minibatch":2,
            "frame":"train",
            "sample_first_crops": True,
            "top_k": 1
        }
    }

def test_make_regularizer():
    transforms = [np.random.uniform(0, 1, (100, 100)) for i in range(10)]
    reg_lambda = np.random.uniform(0, 10)
    loss = deepprofiler.learning.models.make_regularizer(transforms, reg_lambda)
    expected = 0
    for i in range(len(transforms)):
        for j in range(i+1, len(transforms)):
            expected += reg_lambda * tf.reduce_sum(tf.abs(tf.matmul(transforms[i], transforms[j], transpose_a=True, transpose_b=False)))
    tf.assert_equal(loss, expected)


def test_create_keras_resnet(config, targets):
    input_shape = (100, 100, 3)
    lr = 0.001
    embed_dims = 256
    reg_lambda = np.random.uniform(0, 10)
    model = deepprofiler.learning.models.create_keras_resnet(input_shape, targets, config, lr, embed_dims, reg_lambda)
    assert model.input_shape == (None,) + input_shape
    assert model.output_shape == (None, 10)


def test_create_recurrent_keras_resnet(targets):  # deprecated
    pass
    # input_shape = (4, 100, 100, 3)
    # lr = 0.001
    # embed_dims = 256
    # reg_lambda = np.random.uniform(0, 10)
    # model = deepprofiler.learning.models.create_recurrent_keras_resnet(input_shape, targets, lr, embed_dims, reg_lambda)
    # assert model.input_shape == (None,) + input_shape
    # assert model.output_shape == (None, 10)


def test_create_keras_vgg(config, targets):
    input_shape = (100, 100, 3)
    lr = 0.001
    embed_dims = 256
    reg_lambda = np.random.uniform(0, 10)
    model = deepprofiler.learning.models.create_keras_vgg(input_shape, targets, config, lr, embed_dims, reg_lambda)
    assert model.input_shape == (None,) + input_shape
    assert model.output_shape == (None, 10)
