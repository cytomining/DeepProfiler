from abc import ABC, abstractmethod
import pytest
import deepprofiler.learning.metric
import os
import inspect

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
            "top_k": 2
        },
        "profiling": {
            "feature_layer": "features",
            "output_dir": out_dir,
            "checkpoint": None,
            "gpu": "0"
        }
    }

class MetricClass(deepprofiler.learning.metric.Metric):
     def metric(self, y_true, y_pred):
         pass

def test_init(config):
    test_name = "Dog"
    test_metric = MetricClass(config, test_name)
    assert is_method(test_metric, "__init__")
    assert test_metric.config == config
    assert test_metric.name == test_name

def test_create_metric():
    test_name = "Dog"
    test_metric = MetricClass(config, test_name)
    assert is_method(test_metric, "create_metric")
    assert test_metric.f.__name__ == test_name

def test_metric(config):
    test_name = "Dog"
    test_metric = MetricClass(config, test_name)
    assert is_method(test_metric, "metric")