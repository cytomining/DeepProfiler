import tempfile

import numpy
import numpy.random
import skimage.io

import deepprofiler.dataset.pixels


def test_openImage():
    numpy.random.seed(11)

    tmp_path = tempfile.mkdtemp()

    skimage.io.imsave(tmp_path + "/rand_img_1.png", numpy.random.randint(256, size=(16, 16), dtype=numpy.uint16))
    skimage.io.imsave(tmp_path + "/rand_img_2.png", numpy.random.randint(256, size=(16, 16), dtype=numpy.uint16))
    skimage.io.imsave(tmp_path + "/rand_img_3.png", numpy.random.randint(256, size=(16, 16), dtype=numpy.uint16))
    skimage.io.imsave(tmp_path + "/rand_outlines.png", numpy.random.randint(256, size=(16, 16), dtype=numpy.uint16))

    test_paths = [tmp_path + "/rand_img_1.png", tmp_path + "/rand_img_2.png", tmp_path + "/rand_img_3.png"]
    test_outlines = tmp_path + "/rand_outlines.png"

    new_img = deepprofiler.dataset.pixels.openImage(test_paths, None)
    assert new_img.shape == (16, 16, 3)

    new_img = deepprofiler.dataset.pixels.openImage(test_paths, test_outlines)
    assert new_img.shape == (16, 16, 4)
