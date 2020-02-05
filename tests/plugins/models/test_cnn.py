import pytest
import keras

import deepprofiler.imaging.cropping
import deepprofiler.dataset.image_dataset
import deepprofiler.dataset.metadata
import deepprofiler.dataset.target
import plugins.models.cnn


@pytest.fixture(scope="function")
def generator():
    return deepprofiler.imaging.cropping.CropGenerator


@pytest.fixture(scope="function")
def val_generator():
    return deepprofiler.imaging.cropping.SingleImageCropGenerator


def test_define_model(config, dataset):
    model, optimizer, loss = plugins.models.cnn.define_model(config, dataset)
    assert isinstance(model, keras.Model)
    assert isinstance(optimizer, str) or isinstance(optimizer, keras.optimizers.Optimizer)
    assert isinstance(loss, str) or callable(loss)


def test_init(config, dataset, generator, val_generator):
    dpmodel = plugins.models.cnn.ModelClass(config, dataset, generator, val_generator)
    model, optimizer, loss = plugins.models.cnn.define_model(config, dataset)
    assert dpmodel.feature_model.__eq__(model)
    assert dpmodel.optimizer.__eq__(optimizer)
    assert dpmodel.loss.__eq__(loss)
