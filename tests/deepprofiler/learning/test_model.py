import importlib
import os
import random

import numpy as np
import pytest

from deepprofiler.learning.model import DeepProfilerModel


def __rand_array():
    return np.array(random.sample(range(100), 12))


@pytest.fixture(scope="function")
def crop_generator(config):
    module = importlib.import_module("plugins.crop_generators.{}".format(config["train"]["model"]["crop_generator"]))
    importlib.invalidate_caches()
    generator = module.GeneratorClass
    return generator


@pytest.fixture(scope="function")
def val_crop_generator(config):
    module = importlib.import_module("plugins.crop_generators.{}".format(config["train"]["model"]["crop_generator"]))
    importlib.invalidate_caches()
    generator = module.SingleImageGeneratorClass
    return generator


@pytest.fixture(scope="function")
def model(config, dataset, crop_generator, val_crop_generator):
    def create():
        module = importlib.import_module("plugins.models.{}".format(config["train"]["model"]["name"]))
        importlib.invalidate_caches()
        dpmodel = module.ModelClass(config, dataset, crop_generator, val_crop_generator, is_training=True)
        return dpmodel
    return create


def test_init(config, dataset, crop_generator, val_crop_generator):
    dpmodel = DeepProfilerModel(config, dataset, crop_generator, val_crop_generator, is_training=True)
    assert dpmodel.feature_model is None
    assert dpmodel.config == config
    assert dpmodel.dset == dataset
    assert isinstance(dpmodel.train_crop_generator, crop_generator)
    assert isinstance(dpmodel.val_crop_generator, val_crop_generator)
    assert dpmodel.random_seed is None


def test_seed(model):
    model1 = model()
    seed = random.randint(0, 256)
    model1.seed(seed)
    assert model1.random_seed == seed


def test_train(model, out_dir, data, make_struct, config):
    model1 = model()
    model1.train()
    assert os.path.exists(os.path.join(config["paths"]["checkpoints"], "checkpoint_0001.hdf5"))
    assert os.path.exists(os.path.join(config["paths"]["checkpoints"], "checkpoint_0002.hdf5"))
    assert os.path.exists(os.path.join(config["paths"]["logs"], "log.csv"))
    model2 = model()
    epoch = 3
    model2.config["train"]["model"]["epochs"] = 4
    model2.train(epoch)
    assert os.path.exists(os.path.join(config["paths"]["checkpoints"], "checkpoint_0003.hdf5"))
    assert os.path.exists(os.path.join(config["paths"]["checkpoints"], "checkpoint_0004.hdf5"))
    assert os.path.exists(os.path.join(config["paths"]["logs"], "log.csv"))
