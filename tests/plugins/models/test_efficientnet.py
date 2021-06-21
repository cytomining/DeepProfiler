import pytest
import tensorflow as tf

import deepprofiler.imaging.cropping
import deepprofiler.dataset.image_dataset
import deepprofiler.dataset.metadata
import deepprofiler.dataset.target
import plugins.models.efficientnet

tf.compat.v1.disable_v2_behavior()


@pytest.fixture(scope="function")
def generator():
    return deepprofiler.imaging.cropping.CropGenerator


@pytest.fixture(scope="function")
def val_generator():
    return deepprofiler.imaging.cropping.SingleImageCropGenerator


def test_init(config, dataset, generator, val_generator, is_training=True):
    with tf.compat.v1.Session().as_default():
        config["train"]["model"]["name"] = "efficientnet"
        config["train"]["model"]["params"]["conv_blocks"] = 0
        config["dataset"]["locations"]["box_size"] = 32
        config["train"]["model"]["params"]["pooling"] = 'avg'
        dpmodel = plugins.models.efficientnet.ModelClass(config, dataset, generator, val_generator, is_training)
        model, optimizer, loss = plugins.models.efficientnet.ModelClass.define_model(dpmodel, config, dataset)
        assert dpmodel.feature_model.__eq__(model)
        assert dpmodel.optimizer.__eq__(optimizer)
        assert dpmodel.loss.__eq__(loss)
        assert isinstance(model, tf.compat.v1.keras.Model)
        assert isinstance(optimizer, str) or isinstance(optimizer, tf.compat.v1.keras.optimizers.Optimizer)
        assert isinstance(loss, str) or callable(loss)
