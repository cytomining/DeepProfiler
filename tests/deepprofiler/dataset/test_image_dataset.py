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


def test_init(imaging_metadata, out_dir, imaging_dataset, config, make_struct):
    sampling_field = config["train"]["sampling"]["field"]
    channels = config["dataset"]["images"]["channels"]
    keygen = lambda r: "{}/{}-{}".format(r["Metadata_Plate"], r["Metadata_Well"], r["Metadata_Site"])
    dset = deepprofiler.dataset.image_dataset.ImageDataset(imaging_metadata, sampling_field, channels, out_dir, keygen)
    assert dset.meta == imaging_metadata
    assert dset.sampling_field == sampling_field
    np.testing.assert_array_equal(dset.sampling_values, imaging_metadata.data["Sampling"].unique())
    assert dset.channels == channels
    assert dset.root == out_dir
    assert dset.keyGen == keygen


def test_get_image_paths(imaging_metadata, out_dir, imaging_dataset, config, make_struct):
    for idx, row in imaging_dataset.meta.data.iterrows():
        key, image, outlines = imaging_dataset.getImagePaths(row)
        testKey = imaging_dataset.keyGen(row)
        testImage = [imaging_dataset.root + "/" + row[ch] for ch in imaging_dataset.channels]
        testOutlines = imaging_dataset.outlines
        assert key == testKey
        assert image == testImage
        assert outlines == testOutlines


def test_sample_images(imaging_metadata, out_dir, imaging_dataset, config, make_struct):
    n = 3
    keys, images, targets, outlines = imaging_dataset.sampleImages(imaging_dataset.sampling_values, n)
    print(keys, images, targets, outlines)
    assert len(keys) == 2 * n
    assert len(images) == 2 * n
    assert len(targets) == 2 * n
    assert len(outlines) == 2 * n


def test_get_train_batch(imaging_metadata, out_dir, imaging_dataset, config, make_struct):
    images = np.random.randint(0, 256, (128, 128, 36), dtype=np.uint8)
    for i in range(0, 36, 3):
        skimage.io.imsave(os.path.join(out_dir, imaging_dataset.meta.data["R"][i // 3]), images[:, :, i])
        skimage.io.imsave(os.path.join(out_dir, imaging_dataset.meta.data["G"][i // 3]), images[:, :, i + 1])
        skimage.io.imsave(os.path.join(out_dir, imaging_dataset.meta.data["B"][i // 3]), images[:, :, i + 2])
    batch_size = 3
    batch = imaging_dataset.getTrainBatch(batch_size)
    assert len(batch) == batch_size
    for image in batch["images"]:
        assert image.shape == (128, 128, 3)
        for i in range(3):
            assert image[:, :, i] in np.rollaxis(images, -1)


def test_scan(imaging_metadata, out_dir, imaging_dataset, config, make_struct):
    images = np.random.randint(0, 256, (128, 128, 36), dtype=np.uint8)
    for i in range(0, 36, 3):
        skimage.io.imsave(os.path.join(out_dir, imaging_dataset.meta.data["R"][i // 3]), images[:, :, i])
        skimage.io.imsave(os.path.join(out_dir, imaging_dataset.meta.data["G"][i // 3]), images[:, :, i + 1])
        skimage.io.imsave(os.path.join(out_dir, imaging_dataset.meta.data["B"][i // 3]), images[:, :, i + 2])
    data = {"index": [], "image": [], "meta": []}

    def func(index, image, meta):
        data["index"].append(index)
        data["image"].append(image)
        data["meta"].append(meta)

    imaging_dataset.scan(func, frame="all")
    for index in data["index"]:
        assert index in range(12)
    for image in data["image"]:
        assert image.shape == (128, 128, 3)
        for i in range(3):
            assert image[:, :, i] in np.rollaxis(images, -1)
    for meta in data["meta"]:
        assert (imaging_dataset.meta.data == meta).all(1).any()


def test_number_of_records(imaging_metadata, out_dir, imaging_dataset, config, make_struct):
    assert imaging_dataset.number_of_records("all") == len(imaging_dataset.meta.data)
    assert imaging_dataset.number_of_records("val") == len(imaging_dataset.meta.val)
    assert imaging_dataset.number_of_records("train") == len(imaging_dataset.meta.train)
    assert imaging_dataset.number_of_records("other") == 0


def test_add_target(imaging_metadata, out_dir, imaging_dataset, config, make_struct):
    target = deepprofiler.dataset.target.MetadataColumnTarget("Target", random.sample(range(100), 12))
    imaging_dataset.add_target(target)
    assert target in imaging_dataset.targets


def test_read_dataset(imaging_metadata, out_dir, imaging_dataset, config, make_struct):
    dset = deepprofiler.dataset.image_dataset.read_dataset(config)
    pd.testing.assert_frame_equal(dset.meta.data, deepprofiler.dataset.metadata.Metadata(config["paths"]["index"], dtype=None).data)
    assert dset.channels == config["dataset"]["images"]["channels"]
    assert dset.root == config["paths"]["images"]
    assert dset.sampling_field == config["train"]["sampling"]["field"]
    np.testing.assert_array_equal(dset.sampling_values, dset.meta.data[dset.sampling_field].unique())
