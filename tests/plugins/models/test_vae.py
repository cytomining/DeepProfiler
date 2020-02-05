import pytest
import keras

import deepprofiler.imaging.cropping
import deepprofiler.dataset.image_dataset
import deepprofiler.dataset.metadata
import deepprofiler.dataset.target
import plugins.models.vae


@pytest.fixture(scope="function")
def generator():
    return deepprofiler.imaging.cropping.CropGenerator


@pytest.fixture(scope="function")
def val_generator():
    return deepprofiler.imaging.cropping.SingleImageCropGenerator


def test_define_model(config, dataset):
    vae, encoder, decoder, optimizer, loss = plugins.models.vae.define_model(config, dataset)
    assert isinstance(vae, keras.Model)
    assert isinstance(encoder, keras.Model)
    assert isinstance(decoder, keras.Model)
    assert isinstance(optimizer, str) or isinstance(optimizer, keras.optimizers.Optimizer)
    assert isinstance(loss, str) or callable(loss)


def test_init(config, dataset, generator, val_generator):
    dpmodel = plugins.models.vae.ModelClass(config, dataset, generator, val_generator)
    vae, encoder, decoder, optimizer, loss = plugins.models.vae.define_model(config, dataset)
    assert dpmodel.feature_model.__eq__(vae)
    assert dpmodel.encoder.__eq__(encoder)
    assert dpmodel.generator.__eq__(decoder)
    assert dpmodel.optimizer.__eq__(optimizer)
    assert dpmodel.loss.__eq__(loss)
