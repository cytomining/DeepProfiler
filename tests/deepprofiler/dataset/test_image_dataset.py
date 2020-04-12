import json
import os
import random
import threading

import deepprofiler.dataset.image_dataset
import deepprofiler.dataset.metadata
import deepprofiler.dataset.target
import numpy as np
import pandas as pd
import skimage.io


def test_init(metadata, out_dir, dataset, config, make_struct):
    sampling_field = config["dataset"]["metadata"]["label_field"]
    channels = config["dataset"]["images"]["channels"]
    keygen = lambda r: "{}/{}-{}".format(r["Metadata_Plate"], r["Metadata_Well"], r["Metadata_Site"])
    dset = deepprofiler.dataset.image_dataset.ImageDataset(metadata, sampling_field, channels, config["paths"]["root_dir"], keygen, config)
    assert dset.meta == metadata
    assert dset.sampling_field == sampling_field
    np.testing.assert_array_equal(dset.sampling_values, metadata.data["Class"].unique())
    assert dset.channels == channels
    assert dset.root == out_dir
    assert dset.keyGen == keygen


def test_get_image_paths(metadata, out_dir, dataset, config, make_struct):
    for idx, row in dataset.meta.data.iterrows():
        key, image, outlines = dataset.get_image_paths(row)
        testKey = dataset.keyGen(row)
        testImage = [dataset.root + "/" + row[ch] for ch in dataset.channels]
        testOutlines = dataset.outlines
        assert key == testKey
        assert image == testImage
        assert outlines == testOutlines


def test_get_train_batch(metadata, out_dir, dataset, config, make_struct):
    images = np.random.randint(0, 256, (128, 128, 36), dtype=np.uint8)
    for i in range(0, 36, 3):
        skimage.io.imsave(os.path.join(out_dir, dataset.meta.data["R"][i // 3]), images[:, :, i])
        skimage.io.imsave(os.path.join(out_dir, dataset.meta.data["G"][i // 3]), images[:, :, i + 1])
        skimage.io.imsave(os.path.join(out_dir, dataset.meta.data["B"][i // 3]), images[:, :, i + 2])

    lock = threading.Lock()
    batch = dataset.get_train_batch(lock)
    assert len(batch["keys"]) == int(config["train"]["model"]["params"]["batch_size"] / config["train"]["sampling"]["workers"]) 


def test_scan(metadata, out_dir, dataset, config, make_struct):
    images = np.random.randint(0, 256, (128, 128, 36), dtype=np.uint8)
    for i in range(0, 36, 3):
        skimage.io.imsave(os.path.join(out_dir, dataset.meta.data["R"][i // 3]), images[:, :, i])
        skimage.io.imsave(os.path.join(out_dir, dataset.meta.data["G"][i // 3]), images[:, :, i + 1])
        skimage.io.imsave(os.path.join(out_dir, dataset.meta.data["B"][i // 3]), images[:, :, i + 2])
    data = {"index": [], "image": [], "meta": []}

    def func(index, image, meta):
        data["index"].append(index)
        data["image"].append(image)
        data["meta"].append(meta)

    dataset.scan(func, frame="all")
    for index in data["index"]:
        assert index in range(12)
    for image in data["image"]:
        assert image.shape == (128, 128, 3)
        for i in range(3):
            assert image[:, :, i] in np.rollaxis(images, -1)
    for meta in data["meta"]:
        assert (dataset.meta.data == meta).all(1).any()


def test_number_of_records(metadata, out_dir, dataset, config, make_struct):
    assert dataset.number_of_records("all") == len(dataset.meta.data)
    assert dataset.number_of_records("val") == len(dataset.meta.val)
    assert dataset.number_of_records("train") == len(dataset.meta.train)
    assert dataset.number_of_records("other") == 0


def test_add_target(metadata, out_dir, dataset, config, make_struct):
    target = deepprofiler.dataset.target.MetadataColumnTarget("Target", random.sample(range(100), 12))
    dataset.add_target(target)
    assert target in dataset.targets


def test_read_dataset(metadata, out_dir, dataset, config, make_struct):
    dset = deepprofiler.dataset.image_dataset.read_dataset(config)
    pd.testing.assert_frame_equal(dset.meta.data, deepprofiler.dataset.metadata.Metadata(config["paths"]["index"], dtype=None).data)
    assert dset.channels == config["dataset"]["images"]["channels"]
    assert dset.root == config["paths"]["images"]
    assert dset.sampling_field == config["dataset"]["metadata"]["label_field"]
    np.testing.assert_array_equal(dset.sampling_values, dset.meta.data[dset.sampling_field].unique())
