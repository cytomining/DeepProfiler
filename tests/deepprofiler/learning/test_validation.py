import numpy as np
import pytest
import tensorflow as tf

import deepprofiler.dataset.image_dataset
import deepprofiler.dataset.metadata
import deepprofiler.dataset.target
import deepprofiler.imaging.cropping
import deepprofiler.learning.training
import deepprofiler.learning.validation

tf.compat.v1.disable_v2_behavior()


@pytest.fixture(scope="function")
def session():
    configuration = tf.compat.v1.ConfigProto(device_count = {'GPU': 0})
    configuration.gpu_options.visible_device_list = "0"
    session = tf.compat.v1.Session(config=configuration)
    return session


@pytest.fixture(scope="function")
def crop_generator(config, dataset, session):
    crop_generator = deepprofiler.imaging.cropping.SingleImageCropGenerator(config, dataset)
    crop_generator.start(session)
    return crop_generator


@pytest.fixture(scope="function")
def validation(config, dataset, crop_generator, session):
    return deepprofiler.learning.validation.Validation(config, dataset, crop_generator, session)


def test_init(config, dataset, crop_generator, session, validation):
    validation = validation
    assert validation.config == config
    assert validation.dset == dataset
    assert validation.crop_generator == crop_generator
    assert validation.session == session
    assert validation.batch_inputs == []
    assert validation.batch_outputs == []


def test_load_validation_data(config, dataset, crop_generator, session, out_dir, data, locations):
    test_images, test_labels = deepprofiler.learning.validation.load_validation_data(config, dataset, crop_generator, session)
    assert test_labels.shape == (12,4)
    assert test_images.shape == (12,16,16,3)
    test_labels_amax = np.amax(test_labels, axis=1)
    test_labels_amax_sum = 0
    for term in test_labels_amax:
        test_labels_amax_sum += term
    assert test_labels_amax_sum == 12
