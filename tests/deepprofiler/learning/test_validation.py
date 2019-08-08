import os
import random
import json

import numpy as np
import pandas as pd
import pytest
import skimage.io
import tensorflow as tf

import deepprofiler.dataset.image_dataset
import deepprofiler.dataset.metadata
import deepprofiler.dataset.target
import deepprofiler.imaging.cropping
import deepprofiler.learning.training
import deepprofiler.learning.validation


def test_init(config, dataset, crop_generator, session, validation):
    validation = validation
    assert validation.config == config
    assert validation.dset == dataset
    assert validation.crop_generator == crop_generator
    assert validation.session == session
    assert validation.batch_inputs == []
    assert validation.batch_outputs == []


def test_process_batches():  # tested in test_validate
    pass


def test_validate(config, dataset, crop_generator, session, out_dir, data, locations):
    test_images, test_labels = deepprofiler.learning.validation.validate(config, dataset, crop_generator, session)
    assert test_labels.shape == (12,4)
    assert test_images.shape == (12,16,16,3)
    test_labels_amax = np.amax(test_labels, axis=1)
    test_labels_amax_sum = 0
    for term in test_labels_amax:
        test_labels_amax_sum += term
    assert test_labels_amax_sum == 12
