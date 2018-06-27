import json
import os
import random

import deepprofiler.dataset.image_dataset
import deepprofiler.dataset.metadata
import deepprofiler.dataset.target
import numpy as np
import pandas as pd
import pytest
import skimage.io


def __rand_array():
    return np.array(random.sample(range(100), 12))


@pytest.fixture(scope='function')
def out_dir(tmpdir):
    return os.path.abspath(tmpdir.mkdir("test"))


@pytest.fixture(scope='function')
def metadata(out_dir):
    filename = os.path.join(out_dir, 'metadata.csv')
    df = pd.DataFrame({
        'Metadata_Plate': __rand_array(),
        'Metadata_Well': __rand_array(),
        'Metadata_Site': __rand_array(),
        'R': [str(x) + '.png' for x in __rand_array()],
        'G': [str(x) + '.png' for x in __rand_array()],
        'B': [str(x) + '.png' for x in __rand_array()],
        'Sampling': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        'Split': [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    }, dtype=int)
    df.to_csv(filename, index=False)
    meta = deepprofiler.dataset.metadata.Metadata(filename)
    train_rule = lambda data: data['Split'].astype(int) == 0
    val_rule = lambda data: data['Split'].astype(int) == 1
    meta.splitMetadata(train_rule, val_rule)
    return meta


@pytest.fixture(scope='function')
def dataset(metadata, out_dir):
    keygen = lambda r: "{}/{}-{}".format(r["Metadata_Plate"], r["Metadata_Well"], r["Metadata_Site"])
    return deepprofiler.dataset.image_dataset.ImageDataset(metadata, 'Sampling', ['R', 'G', 'B'], out_dir, keygen)


def test_init(metadata, out_dir):
    sampling_field = 'Sampling'
    channels = ['R', 'G', 'B']
    keygen = lambda r: "{}/{}-{}".format(r["Metadata_Plate"], r["Metadata_Well"], r["Metadata_Site"])
    dset = deepprofiler.dataset.image_dataset.ImageDataset(metadata, sampling_field, channels, out_dir, keygen)
    assert dset.meta == metadata
    assert dset.sampling_field == sampling_field
    np.testing.assert_array_equal(dset.sampling_values, metadata.data[sampling_field].unique())
    assert dset.channels == channels
    assert dset.root == out_dir
    assert dset.keyGen == keygen


def test_get_image_paths(dataset):
    for idx, row in dataset.meta.data.iterrows():
        key, image, outlines = dataset.getImagePaths(row)
        testKey = dataset.keyGen(row)
        testImage = [dataset.root + '/' + row[ch] for ch in dataset.channels]
        testOutlines = dataset.outlines
        assert key == testKey
        assert image == testImage
        assert outlines == testOutlines


def test_sample_images(dataset):
    n = 3
    keys, images, targets, outlines = dataset.sampleImages(dataset.sampling_values, n)
    print(keys, images, targets, outlines)
    assert len(keys) == 2 * n
    assert len(images) == 2 * n
    assert len(targets) == 2 * n
    assert len(outlines) == 2 * n


def test_get_train_batch(dataset, out_dir):
    images = np.random.randint(0, 256, (128, 128, 36), dtype=np.uint8)
    for i in range(0, 36, 3):
        skimage.io.imsave(os.path.join(out_dir, dataset.meta.data['R'][i // 3]), images[:, :, i])
        skimage.io.imsave(os.path.join(out_dir, dataset.meta.data['G'][i // 3]), images[:, :, i + 1])
        skimage.io.imsave(os.path.join(out_dir, dataset.meta.data['B'][i // 3]), images[:, :, i + 2])
    batch_size = 3
    batch = dataset.getTrainBatch(batch_size)
    assert len(batch) == batch_size
    for image in batch['images']:
        assert image.shape == (128, 128, 3)
        for i in range(3):
            assert image[:, :, i] in np.rollaxis(images, -1)


def test_scan(dataset, out_dir):
    images = np.random.randint(0, 256, (128, 128, 36), dtype=np.uint8)
    for i in range(0, 36, 3):
        skimage.io.imsave(os.path.join(out_dir, dataset.meta.data['R'][i // 3]), images[:, :, i])
        skimage.io.imsave(os.path.join(out_dir, dataset.meta.data['G'][i // 3]), images[:, :, i + 1])
        skimage.io.imsave(os.path.join(out_dir, dataset.meta.data['B'][i // 3]), images[:, :, i + 2])
    # f = lambda index, image, meta: print(index, image, meta)
    data = {'index': [], 'image': [], 'meta': []}

    def func(index, image, meta):
        data['index'].append(index)
        data['image'].append(image)
        data['meta'].append(meta)

    dataset.scan(func, frame='all')
    for index in data['index']:
        assert index in range(12)
    for image in data['image']:
        assert image.shape == (128, 128, 3)
        for i in range(3):
            assert image[:, :, i] in np.rollaxis(images, -1)
    for meta in data['meta']:
        assert (dataset.meta.data == meta).all(1).any()


def test_number_of_records(dataset):
    assert dataset.number_of_records('all') == len(dataset.meta.data)
    assert dataset.number_of_records('val') == len(dataset.meta.val)
    assert dataset.number_of_records('train') == len(dataset.meta.train)
    assert dataset.number_of_records('other') == 0


def test_add_target(dataset):
    target = deepprofiler.dataset.target.MetadataColumnTarget('Target', random.sample(range(100), 12))
    dataset.add_target(target)
    assert target in dataset.targets


def test_read_dataset():
    with open('deepprofiler/examples/config/learning.json', 'r') as f:
        config = json.load(f)
    dset = deepprofiler.dataset.image_dataset.read_dataset(config)
    pd.testing.assert_frame_equal(dset.meta.data, deepprofiler.dataset.metadata.Metadata(config["image_set"]["index"], dtype=None).data)
    assert dset.channels == config["image_set"]["channels"]
    assert dset.root == config["image_set"]["path"]
    assert dset.sampling_field == config["sampling"]["field"]
    np.testing.assert_array_equal(dset.sampling_values, dset.meta.data[dset.sampling_field].unique())
