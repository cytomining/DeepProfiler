import pytest
import tensorflow as tf

import deepprofiler.imaging.cropping
import deepprofiler.dataset.image_dataset
import deepprofiler.dataset.metadata
import deepprofiler.dataset.target
import plugins.models.densenet

tf.compat.v1.disable_v2_behavior()


@pytest.fixture(scope="function")
def generator():
    return deepprofiler.imaging.cropping.CropGenerator


@pytest.fixture(scope="function")
def val_generator():
    return deepprofiler.imaging.cropping.SingleImageCropGenerator


def test_init(config, dataset, generator, val_generator, is_training=False):
    config["train"]["model"]["name"] = "densenet"
    config["train"]["model"]["params"]["conv_blocks"] = 121
    config["dataset"]["locations"]["box_size"] = 32
    dpmodel = plugins.models.densenet.ModelClass(config, dataset, generator, val_generator, is_training=True)
    model, optimizer, loss = plugins.models.densenet.ModelClass.define_model(dpmodel, config, dataset)
    assert dpmodel.feature_model.__eq__(model)
    assert dpmodel.optimizer.__eq__(optimizer)
    assert dpmodel.loss.__eq__(loss)
    assert isinstance(model, tf.compat.v1.keras.Model)
    assert isinstance(optimizer, str) or isinstance(optimizer, tf.compat.v1.keras.optimizers.Optimizer)
    assert isinstance(loss, str) or callable(loss)
