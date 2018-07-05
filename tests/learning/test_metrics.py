import re

import numpy as np
import pytest
import tensorflow as tf

import deepprofiler.learning.metrics


@pytest.fixture(scope='function')
def metrics():
    return deepprofiler.learning.metrics.Metrics(k=2, name="test")


def test_init():
    k = 2
    name = "test"
    metrics = deepprofiler.learning.metrics.Metrics(k, name)
    assert metrics.correct == 0.0
    assert metrics.in_top_k == 0.0
    assert metrics.counts == 0.0
    assert metrics.with_k == k
    assert metrics.name == name
    assert metrics.cmatrix is None


def test_update(metrics):
    old_correct = metrics.correct
    old_in_top_k = metrics.in_top_k
    old_counts = metrics.counts
    old_cmatrix = metrics.cmatrix
    values = np.random.uniform(0, 1, (3,)).tolist()
    values[2] = np.random.uniform(0, 1, (2, 2))
    counts = np.random.uniform(0, 1)
    metrics.update(values, counts)
    assert metrics.correct == old_correct + values[0]
    assert metrics.in_top_k == old_in_top_k + values[1]
    assert metrics.counts == old_counts + counts
    if old_cmatrix is None:
        np.testing.assert_array_equal(metrics.cmatrix, values[2])
    else:
        np.testing.assert_array_equal(metrics.cmatrix, old_cmatrix + values[2])


def test_result_string(metrics):
    values = np.random.uniform(0, 1, (3,)).tolist()
    values[2] = np.random.uniform(0, 1, (2, 2))
    counts = np.random.uniform(0, 1)
    metrics.update(values, counts)
    result = metrics.result_string()
    expected = "{}=[Acc: {:0.4f} Top-2: {:0.4f} Samples: {:0.0f}]".format(metrics.name,
                                                                          metrics.correct / metrics.counts,
                                                                          metrics.in_top_k / metrics.counts,
                                                                          metrics.counts)
    assert result == expected


def test_configure_ops(metrics):
    classes = 10
    label_dict = {k: np.random.uniform(0, 1) for k in range(classes)}
    metrics.configure_ops(label_dict)
    assert metrics.label_dict == label_dict
    assert metrics.true_labels.get_shape().as_list() == [None]
    assert metrics.predictions.get_shape().as_list() == [None, classes]
    assert metrics.correct_op.get_shape().as_list() == []
    assert metrics.in_top_k_op.get_shape().as_list() == []
    assert metrics.confussion_matrix.get_shape().as_list() == [classes, classes]


def test_get_ops(metrics):
    classes = 10
    label_dict = {k: np.random.uniform(0, 1) for k in range(classes)}
    metrics.configure_ops(label_dict)
    ops = metrics.get_ops()
    assert ops[0] == metrics.correct_op
    assert ops[1] == metrics.in_top_k_op
    assert ops[2] == metrics.confussion_matrix


def test_set_inputs(metrics):
    classes = 10
    label_dict = {k: np.random.uniform(0, 1) for k in range(classes)}
    metrics.configure_ops(label_dict)
    batch_size = 32
    labels = np.random.uniform(0, 1, (batch_size,))
    predictions = np.random.uniform(0, 1, (batch_size, classes))
    feed_dict = metrics.set_inputs(labels, predictions)
    np.testing.assert_array_equal(feed_dict[metrics.true_labels], labels)
    np.testing.assert_array_equal(feed_dict[metrics.predictions], predictions)
