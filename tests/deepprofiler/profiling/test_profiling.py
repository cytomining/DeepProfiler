import importlib
import os

import numpy as np
import pytest
import tensorflow as tf

import deepprofiler.dataset.image_dataset
import deepprofiler.profiling

tf.compat.v1.disable_v2_behavior()


@pytest.fixture(scope="function")
def checkpoint(config, dataset):
    prof = deepprofiler.profiling.Profile(config, dataset)
    prof.feature_model.compile("sgd", "categorical_crossentropy")
    filename = os.path.join(config["paths"]["checkpoints"], config["profile"]["checkpoint"])
    with tf.compat.v1.Session().as_default():
        prof.feature_model.save_weights(filename)
    return filename


@pytest.fixture(scope="function")
def profile(config, dataset):
    return deepprofiler.profiling.Profile(config, dataset)


def test_init(config, dataset, locations):
    metadata = deepprofiler.dataset.image_dataset.read_dataset(config)
    prof = deepprofiler.profiling.Profile(config, metadata)
    assert prof.config == config
    assert prof.dset.config == dataset.config
    assert prof.num_channels == len(config["dataset"]["images"]["channels"])
    assert isinstance(prof.profile_crop_generator, importlib.import_module(
        "deepprofiler.crop_generators.{}".format(config["train"]["model"]["crop_generator"])
    ).SingleImageGeneratorClass)
    assert prof.feature_model is not None


def test_configure(profile, checkpoint):
    with tf.compat.v1.Session().as_default():
        profile.configure()
        assert isinstance(profile.feat_extractor, tf.keras.Model)


def test_check(profile, metadata):
    assert profile.check(metadata.data)


def test_extract_features(profile, metadata, locations, checkpoint):
    with tf.compat.v1.Session().as_default():
        meta = metadata.data.iloc[0]
        image = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        profile.configure()
        profile.extract_features(None, image, meta)
        output_file = profile.config["paths"]["features"] + "/{}/{}/{}.npz"\
            .format(meta["Metadata_Plate"], meta["Metadata_Well"], meta["Metadata_Site"])
        assert os.path.isfile(output_file)


def test_profile(config, dataset, data, locations, checkpoint):
    with tf.compat.v1.Session().as_default():
        deepprofiler.profiling.profile(config, dataset)
        for index, row in dataset.meta.data.iterrows():
            output_file = config["paths"]["features"] + "/{}/{}/{}.npz"\
                .format(row["Metadata_Plate"], row["Metadata_Well"], row["Metadata_Site"])
            assert os.path.isfile(output_file)
