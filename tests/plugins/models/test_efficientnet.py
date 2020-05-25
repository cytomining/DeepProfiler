import pytest
import keras

import deepprofiler.imaging.cropping
import deepprofiler.dataset.image_dataset
import deepprofiler.dataset.metadata
import deepprofiler.dataset.target
import plugins.models.efficientnet


@pytest.fixture(scope="function")
def generator():
    return deepprofiler.imaging.cropping.CropGenerator


@pytest.fixture(scope="function")
def val_generator():
    return deepprofiler.imaging.cropping.SingleImageCropGenerator


def test_init(config, dataset, generator, val_generator):
    config["train"]["model"]["name"] = "efficientnet"
    config["train"]["model"]["params"]["conv_blocks"] = 0
    config["dataset"]["locations"]["box_size"] = 32
    config["train"]["model"]["params"]["pooling"] = 'avg'
    dpmodel = plugins.models.efficientnet.ModelClass(config, dataset, generator, val_generator)
    model, optimizer, loss = plugins.models.efficientnet.ModelClass.define_model(dpmodel, config, dataset)
    assert dpmodel.feature_model.__eq__(model)
    assert dpmodel.optimizer.__eq__(optimizer)
    assert dpmodel.loss.__eq__(loss)
    assert isinstance(model, keras.Model)
    assert isinstance(optimizer, str) or isinstance(optimizer, keras.optimizers.Optimizer)
    assert isinstance(loss, str) or callable(loss)
