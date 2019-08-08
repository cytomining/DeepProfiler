import importlib
import os
import random
import json

import numpy as np
import pandas as pd
import pytest
import skimage.io

import deepprofiler.dataset.target
import deepprofiler.dataset.image_dataset
import deepprofiler.imaging.cropping
from deepprofiler.learning.model import DeepProfilerModel


def test_init(config, model_dataset, crop_generator_plugin, val_crop_generator_plugin):
    dpmodel = DeepProfilerModel(config, model_dataset, crop_generator_plugin, val_crop_generator_plugin)
    assert dpmodel.feature_model is None
    assert dpmodel.config == config
    assert dpmodel.dset == model_dataset
    assert isinstance(dpmodel.train_crop_generator, crop_generator_plugin)
    assert isinstance(dpmodel.val_crop_generator, val_crop_generator_plugin)
    assert dpmodel.random_seed is None


def test_seed(model):
    model1 = model()
    seed = random.randint(0, 256)
    model1.seed(seed)
    assert model1.random_seed == seed


def test_train(model, out_dir, make_struct, config):
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
