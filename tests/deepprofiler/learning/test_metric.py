import deepprofiler.learning.metric
import inspect


def is_method(obj, name):
    return hasattr(obj, name) and inspect.ismethod(getattr(obj, name))


class MetricClass(deepprofiler.learning.metric.Metric):
    def metric(self, y_true, y_pred):
        pass


def test_init(config, out_dir, make_struct):
    test_name = "Dog"
    test_metric = MetricClass(config, test_name)
    assert is_method(test_metric, "__init__")
    assert test_metric.config == config
    assert test_metric.name == test_name


def test_create_metric(config, out_dir, make_struct):
    test_name = "Dog"
    test_metric = MetricClass(config, test_name)
    assert is_method(test_metric, "create_metric")
    assert test_metric.f.__name__ == test_name


def test_metric(config, out_dir, make_struct):
    test_name = "Dog"
    test_metric = MetricClass(config, test_name)
    assert is_method(test_metric, "metric")