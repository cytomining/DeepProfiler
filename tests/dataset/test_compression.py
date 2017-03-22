import dataset.compression
import numpy
import numpy.testing
import numpy.random
import pytest
import glob
import os.path
import scipy.misc


@pytest.fixture(scope="function")
def out_dir(tmpdir):
    return os.path.abspath(tmpdir.mkdir("images"))

@pytest.fixture(scope="function")
def compress(out_dir):
    stats = {"original_size": [16, 16]}
    channels = ["DNA", "ER", "Mito"]
    return dataset.compression.Compress(stats, channels, out_dir)


def test_init(compress, out_dir):
    stats = {"original_size": [16, 16]}
    channels = ["DNA", "ER", "Mito"]
    control_distribution = numpy.zeros((3, 2 ** 8), dtype=numpy.float64)

    assert compress.stats == stats
    assert compress.channels == channels
    assert compress.out_dir == out_dir
    assert compress.count == 0
    assert compress.expected == 1
    assert not compress.metadata_control_filter("x")
    numpy.testing.assert_array_equal(compress.controls_distribution, control_distribution)
    assert compress.source_format == "tiff"
    assert compress.target_format == "png"
    assert compress.output_shape == [16, 16]


def test_process_image(compress, out_dir):
    numpy.random.seed(8)
    image = numpy.random.randint(256, size=(16, 16, 3), dtype=numpy.uint16)

    meta = {
        "DNA": "/User/jcaciedo/LUAD/dna.tiff",
        "ER": "/User/jcaciedo/LUAD/er.tiff",
        "Mito": "/User/jcaciedo/LUAD/mito.tiff"
    }
    compress.stats["illum_correction_function"] = numpy.ones((16,16,3))
    compress.stats["upper_percentiles"] = [255, 255, 255]
    compress.stats["lower_percentiles"] = [0, 0, 0]

    compress.process_image(0, image, meta)

    filenames = glob.glob(os.path.join(out_dir,"*"))
    real_filenames = [os.path.join(out_dir, x) for x in ["dna.png", "er.png", "mito.png"]]
    filenames.sort()

    assert real_filenames == filenames

    for i in range(3):
        data = scipy.misc.imread(filenames[i])
        numpy.testing.assert_array_equal(image[:,:,i], data)
