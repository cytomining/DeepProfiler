import keras
import numpy as np
import pytest
import tensorflow as tf

import deepprofiler.dataset.target
import deepprofiler.learning.model


@pytest.fixture(scope='function')
def targets():
    return [
        deepprofiler.dataset.target.MetadataColumnTarget("0", np.random.uniform(0, 1, 10))
    ]


def test_make_regularizer():
    transforms = [np.random.uniform(0, 1, (100, 100)) for i in range(10)]
    reg_lambda = np.random.uniform(0, 10)
    loss = deepprofiler.learning.model.make_regularizer(transforms, reg_lambda)
    expected = 0
    for i in range(len(transforms)):
        for j in range(i+1, len(transforms)):
            expected += reg_lambda * tf.reduce_sum(tf.abs(tf.matmul(transforms[i], transforms[j], transpose_a=True, transpose_b=False)))
    tf.assert_equal(loss, expected)


def test_create_keras_resnet(targets):
    input_shape = (100, 100, 3)
    lr = 0.001
    embed_dims = 256
    reg_lambda = np.random.uniform(0, 10)
    model = deepprofiler.learning.model.create_keras_resnet(input_shape, targets, lr, embed_dims, reg_lambda)
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


def test_create_keras_vgg(targets):
    input_shape = (100, 100, 3)
    lr = 0.001
    embed_dims = 256
    reg_lambda = np.random.uniform(0, 10)
    model = deepprofiler.learning.model.create_keras_vgg(input_shape, targets, lr, embed_dims, reg_lambda)
    assert model.input_shape == (None,) + input_shape
    assert model.output_shape == (None, 10)
