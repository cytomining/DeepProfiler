import os
import pytest
import keras

import deepprofiler.imaging.cropping
import deepprofiler.dataset.image_dataset
import deepprofiler.dataset.metadata
import deepprofiler.dataset.target
import plugins.models.gan


@pytest.fixture(scope="function")
def generator():
    return deepprofiler.imaging.cropping.CropGenerator


@pytest.fixture(scope="function")
def val_generator():
    return deepprofiler.imaging.cropping.SingleImageCropGenerator


@pytest.fixture(scope="function")
def model(config, dataset, generator, val_generator):
    def create():
        return plugins.models.gan.ModelClass(config, dataset, generator, val_generator)
    return create


def test_gan(config, generator, val_generator):
    gan = plugins.models.gan.GAN(config, generator, val_generator)
    assert gan.config == config
    assert gan.crop_generator == generator
    assert gan.val_crop_generator == val_generator
    assert gan.img_cols == config["train"]["sampling"]["box_size"]
    assert gan.img_rows == config["train"]["sampling"]["box_size"]
    assert gan.channels == len(config["dataset"]["images"]["channels"])
    assert gan.img_shape == (
        config["train"]["sampling"]["box_size"],
        config["train"]["sampling"]["box_size"],
        len(config["dataset"]["images"]["channels"])
    )
    assert gan.latent_dim == config["train"]["model"]["params"]["latent_dim"]
    assert isinstance(gan.generator, keras.Model)
    assert isinstance(gan.discriminator, keras.Model)
    assert isinstance(gan.discriminator_fixed, keras.Model)
    assert isinstance(gan.combined, keras.Model)
    assert gan.generator in gan.combined.layers
    assert gan.discriminator not in gan.combined.layers
    assert gan.discriminator_fixed in gan.combined.layers
    assert gan.generator.trainable
    assert gan.discriminator.trainable
    assert not gan.discriminator_fixed.trainable


def test_init(config, dataset, generator, val_generator):
    dpmodel = plugins.models.gan.ModelClass(config, dataset, generator, val_generator)
    gan = plugins.models.gan.GAN(config, generator, val_generator)
    assert dpmodel.gan.__eq__(gan)
    assert isinstance(dpmodel.feature_model, keras.Model)


def test_train(model, out_dir, data, locations, config, make_struct):
    model1 = model()
    model1.train()
    assert os.path.exists(os.path.join(config["paths"]["checkpoints"], "discriminator_epoch_0001.hdf5"))
    assert os.path.exists(os.path.join(config["paths"]["checkpoints"], "generator_epoch_0001.hdf5"))
    assert os.path.exists(os.path.join(config["paths"]["checkpoints"], "discriminator_epoch_0002.hdf5"))
    assert os.path.exists(os.path.join(config["paths"]["checkpoints"], "generator_epoch_0002.hdf5"))
    model2 = model()
    epoch = 3
    model2.config["train"]["model"]["epochs"] = 4
    model2.train(epoch)
    assert os.path.exists(os.path.join(config["paths"]["checkpoints"], "discriminator_epoch_0003.hdf5"))
    assert os.path.exists(os.path.join(config["paths"]["checkpoints"], "generator_epoch_0003.hdf5"))
    assert os.path.exists(os.path.join(config["paths"]["checkpoints"], "discriminator_epoch_0004.hdf5"))
    assert os.path.exists(os.path.join(config["paths"]["checkpoints"], "generator_epoch_0004.hdf5"))
