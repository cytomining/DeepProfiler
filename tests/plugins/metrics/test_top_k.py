import pytest
from keras.metrics import top_k_categorical_accuracy
from deepprofiler.learning.metric import Metric
import plugins.metrics.top_k
import inspect
import os
import numpy as np
import tensorflow as tf

def is_method(obj, name):
    return hasattr(obj, name) and inspect.ismethod(getattr(obj, name))

@pytest.fixture(scope='function')
def out_dir(tmpdir):
    return os.path.abspath(tmpdir.mkdir("test_profiling"))

@pytest.fixture(scope='function')
def config(out_dir):
    return {
        "model": {
            "name": "cnn",
            "crop_generator": "crop_generator",
            "feature_dim": 128,
            "conv_blocks": 3,
            "params": {
                "epochs": 3,
                "steps": 10,
                "learning_rate": 0.0001,
                "batch_size": 16
            }
        },
        "sampling": {
            "images": 12,
            "box_size": 16,
            "locations": 10,
            "locations_field": 'R'
        },
        "image_set": {
            "channels": ['R', 'G', 'B'],
            "mask_objects": False,
            "width": 128,
            "height": 128,
            "path": out_dir
        },
        "training": {
            "learning_rate": 0.001,
            "output": out_dir,
            "epochs": 2,
            "steps": 12,
            "minibatch": 2,
            "visible_gpus": "0"
        },
        "queueing": {
            "loading_workers": 2,
            "queue_size": 2
        },
        "validation": {
            "minibatch": 2,
            "output": out_dir,
            "api_key":'[REDACTED]',
            "project_name":'pytests',
            "frame":"train",
            "sample_first_crops": True,
            "top_k": 1
        },
        "profiling": {
            "feature_layer": "features",
            "output_dir": out_dir,
            "checkpoint": None,
            "gpu": "0"
        }
    }

def test_create_metric(config):
    name = "Dog"
    metric = plugins.metrics.top_k.MetricClass(config, name)
    expected_name = "top_" + str(config['validation']['top_k'])
    assert is_method(metric, "create_metric")
    assert metric.f.__name__ == expected_name


def test_metric(config):
    sess = tf.InteractiveSession()
    name = "Dog"
    metric = plugins.metrics.top_k.MetricClass(config, name)
    y_true = np.array([[0,1],[1,0]])
    y_pred = np.array([[0,1],[0,1]])
    expected_output = 0.5
    output = metric.metric(y_true, y_pred).eval()
    print(output)
    assert output == expected_output
    assert is_method(metric, "metric")
