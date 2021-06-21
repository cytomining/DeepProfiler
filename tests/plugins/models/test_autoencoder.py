import pytest
import tensorflow as tf

import deepprofiler.imaging.cropping
import deepprofiler.dataset.image_dataset
import deepprofiler.dataset.metadata
import deepprofiler.dataset.target
import plugins.models.autoencoder


tf.compat.v1.disable_v2_behavior()


@pytest.fixture(scope="function")
def generator():
    return deepprofiler.imaging.cropping.CropGenerator


@pytest.fixture(scope="function")
def val_generator():
    return deepprofiler.imaging.cropping.SingleImageCropGenerator


def test_define_model(config, dataset):
    autoencoder, encoder, decoder, optimizer, loss = plugins.models.autoencoder.define_model(config, dataset)
    assert isinstance(autoencoder, tf.compat.v1.keras.Model)
    assert isinstance(encoder, tf.compat.v1.keras.Model)
    assert isinstance(decoder, tf.compat.v1.keras.Model)
    assert isinstance(optimizer, str) or isinstance(optimizer, tf.compat.v1.keras.optimizers.Optimizer)
    assert isinstance(loss, str) or callable(loss)


def test_init(config, dataset, generator, val_generator):
    dpmodel = plugins.models.autoencoder.ModelClass(config, dataset, generator, val_generator, is_training=True)
    autoencoder, encoder, decoder, optimizer, loss = plugins.models.autoencoder.define_model(config, dataset)
    assert dpmodel.feature_model.__eq__(autoencoder)
    assert dpmodel.encoder.__eq__(encoder)
    assert dpmodel.decoder.__eq__(decoder)
    assert dpmodel.optimizer.__eq__(optimizer)
    assert dpmodel.loss.__eq__(loss)
