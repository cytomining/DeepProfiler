import dataset.illumination_correction
import numpy
import numpy.testing
import numpy.random
import pytest

@pytest.fixture(scope="function")
def corrector():
    stats = {"mean_image": numpy.ones((16,16,3))}
    channels = ["DNA", "ER", "Mito"]
    target_dim = (24,24,3)
    return dataset.illumination_correction.IlluminationCorrection(
        stats,
        channels,
        target_dim
    )

def test_init(corrector):
    stats = {"mean_image": numpy.ones((16,16,3))}
    channels = ["DNA", "ER", "Mito"]
    target_dim = (24,24,3)

    numpy.testing.assert_array_equal(corrector.stats["mean_image"], stats["mean_image"])
    assert corrector.channels == channels
    assert corrector.target_dim[0] == target_dim[0]
    assert corrector.target_dim[1] == target_dim[1]


def test_channel_function(corrector):
    numpy.random.seed(8)
    mean_image = numpy.random.uniform(low=0.0, high=255.0, size=(16,16))
    result = corrector.channel_function(mean_image, 3)
    # TODO: check contents of result
    assert result.shape == (24,24)


def test_compute_all(corrector):
    corrector.compute_all(3)
    # TODO: check the contents of illum_corr_func
    assert corrector.illum_corr_func.shape == (24,24,3)

def test_apply(corrector):
    image = numpy.random.randint(256, size=(24, 24, 3), dtype=numpy.uint16)
    corrector.compute_all(3)
    corrected = corrector.apply(image)
    # TODO: check the contents of corrected
    assert corrected.shape == (24,24,3)