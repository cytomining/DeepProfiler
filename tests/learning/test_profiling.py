import deepprofiler.learning.profiling
import deepprofiler.dataset.image_dataset
import pytest
import tensorflow as tf
import json
import os
import shutil
import pandas as pd

def test_crop_transform():
    test_crop_ph = tf.placeholder(tf.float32, shape=(16,16,16,10))
    test_image_size = 8
    test_rgb_data = deepprofiler.learning.profiling.crop_transform(test_crop_ph, test_image_size)
    assert test_rgb_data.shape == (160,8,8,3)

@pytest.fixture(scope='function')
def profile():
    test_config = json.load(open("tests/files/config/test_config.json"))
    test_dataset = deepprofiler.dataset.image_dataset.read_dataset(test_config)
    return deepprofiler.learning.profiling.Profile(test_config, test_dataset)

def test_init(profile):
    test_config = json.load(open("tests/files/config/test_config.json"))
    test_dataset = deepprofiler.dataset.image_dataset.read_dataset(test_config)
    test_num_channels = 3
    assert profile.config == test_config
    assert profile.dset.__eq__(test_dataset)
    assert profile.num_channels == test_num_channels
    tf.assert_equal(profile.raw_crops, tf.placeholder(tf.float32, shape=(None, 92, 92, 3)))

def test_check(profile):
    test_config = json.load(open("tests/files/config/test_config.json"))
    test_meta = pd.read_csv(test_config["image_set"]["metadata"])
    assert profile.check(test_meta)
    shutil.rmtree(test_config["profiling"]["output_dir"])

#make a pre-trained model publically available online and add tests for extract_features, configure_resnet, and profile using that model