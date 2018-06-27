import numpy as np
import pytest
import tensorflow as tf

import deepprofiler.imaging.augmentations


def test_augment():
    crop = np.random.uniform(0, 1, (128, 128, 3))
    augmented = deepprofiler.imaging.augmentations.augment(crop)
    assert augmented.shape == crop.shape


def test_augment_multiple():
    crops = np.random.uniform(0, 1, (10, 128, 128, 3))
    augmented = deepprofiler.imaging.augmentations.augment_multiple(crops)
    assert augmented.shape == crops.shape
