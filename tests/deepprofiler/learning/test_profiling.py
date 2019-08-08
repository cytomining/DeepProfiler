from comet_ml import Experiment

import importlib
import random

import keras
import deepprofiler.learning.profiling
import deepprofiler.dataset.image_dataset
import deepprofiler.dataset.metadata
import deepprofiler.dataset.target
import pytest
import tensorflow as tf
import json
import os
import shutil
import pandas as pd
import numpy as np
import skimage.io


def test_init(config, model_dataset):
    prof = deepprofiler.learning.profiling.Profile(config, model_dataset)
    test_num_channels = len(config["dataset"]["images"]["channels"])
    assert prof.config == config
    assert prof.dset == model_dataset
    assert prof.num_channels == test_num_channels
    assert prof.crop_generator == importlib.import_module(
        "plugins.crop_generators.{}".format(config["train"]["model"]["crop_generator"])).GeneratorClass
    assert isinstance(prof.profile_crop_generator, importlib.import_module(
            "plugins.crop_generators.{}".format(config["train"]["model"]["crop_generator"])).SingleImageGeneratorClass)
    assert isinstance(prof.dpmodel, importlib.import_module("plugins.models.{}".format(config["train"]["model"]["name"])).ModelClass)


def test_configure(profile, checkpoint):
    profile.configure()
    assert isinstance(profile.feat_extractor, keras.Model)
    assert isinstance(profile.sess, tf.Session)


def test_check(profile, imaging_metadata):
    assert profile.check(imaging_metadata.data)  # TODO: test false positive


def test_extract_features(profile, imaging_metadata, locations, checkpoint):
    meta = imaging_metadata.data.iloc[0]
    image = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
    profile.configure()
    profile.extract_features(None, image, meta)
    output_file = profile.config["paths"]["features"] + "/{}_{}_{}.npz"\
        .format(meta["Metadata_Plate"], meta["Metadata_Well"], meta["Metadata_Site"])
    assert os.path.isfile(output_file)


def test_profile(config, model_dataset, model_data, locations, checkpoint):
    deepprofiler.learning.profiling.profile(config, model_dataset)
    for index, row in model_dataset.meta.data.iterrows():
        output_file = config["paths"]["features"] + "/{}_{}_{}.npz" \
            .format(row["Metadata_Plate"], row["Metadata_Well"], row["Metadata_Site"])
        assert os.path.isfile(output_file)
