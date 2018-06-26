import deepprofiler.dataset.target
import pytest

from random import shuffle


@pytest.fixture(scope='function')
def target():
    field_name = 'test'
    values = list(range(10))
    shuffle(values)
    return deepprofiler.dataset.target.MetadataColumnTarget(field_name, values)


def test_init(target):
    field_name = 'test'
    values = list(range(10))
    shuffle(values)
    assert target.field_name == field_name
    assert len(target.index) == len(values)
    assert list(target.index) == sorted(values)


def test_get_values(target):
    record = {'test': 0}
    values = list(range(10))
    assert target.get_values(record) == values[0]


def test_shape(target):
    values = list(range(10))
    assert len(target.shape) == 2
    assert target.shape[1] == len(values)
