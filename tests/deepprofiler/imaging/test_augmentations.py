import numpy as np
import tensorflow as tf

import deepprofiler.imaging.augmentations


def test_augment():
    crop = tf.constant(
        np.random.uniform(0, 1, (128, 128, 3)).astype(np.float32)
    )
    augmented = deepprofiler.imaging.augmentations.augment(crop).numpy()
    assert augmented.shape == crop.shape


def test_augment_multiple():
    crops = tf.constant(
        np.random.uniform(0, 1, (10, 128, 128, 3)).astype(np.float32)
    )
    augmented = deepprofiler.imaging.augmentations.augment_multiple(crops).numpy()
    assert augmented.shape == crops.shape
