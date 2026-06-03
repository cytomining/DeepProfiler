import os

import numpy as np
import pandas as pd
import pytest
import skimage.io
import tensorflow as tf

import deepprofiler.dataset.image_dataset
import deepprofiler.dataset.metadata
import deepprofiler.dataset.target
import deepprofiler.imaging.cropping


@pytest.fixture(scope="function")
def crop_generator(config, dataset):
    return deepprofiler.imaging.cropping.CropGenerator(config, dataset)


@pytest.fixture(scope="function")
def single_image_crop_generator(config, dataset):
    return deepprofiler.imaging.cropping.SingleImageCropGenerator(config, dataset)


@pytest.fixture(scope="function")
def prepared_crop_generator(crop_generator, out_dir):
    images = np.random.randint(0, 256, (128, 128, 36), dtype=np.uint8)
    for i in range(0, 36, 3):
        skimage.io.imsave(os.path.join(out_dir, crop_generator.dset.meta.data["R"][i // 3]), images[:, :, i])
        skimage.io.imsave(os.path.join(out_dir, crop_generator.dset.meta.data["G"][i // 3]), images[:, :, i + 1])
        skimage.io.imsave(os.path.join(out_dir, crop_generator.dset.meta.data["B"][i // 3]), images[:, :, i + 2])
    crop_generator.build_input_graph()
    return crop_generator


def test_crop_graph():
    num_crops = 100
    channels = 3
    box_size = 16
    images = tf.constant(np.random.uniform(0, 1, (10, 128, 128, channels + 1)).astype(np.float32))
    boxes = tf.constant(np.random.uniform(0, 1, (num_crops, 4)).astype(np.float32))
    box_ind = tf.constant(np.random.randint(0, 10, (num_crops,)).astype(np.int32))
    mask_ind = tf.constant(np.zeros((num_crops,), dtype=np.int32))
    op = deepprofiler.imaging.cropping.crop_graph(images, boxes, box_ind, mask_ind, box_size, mask_boxes=True)
    assert tuple(op.shape) == (num_crops, box_size, box_size, channels)
    op = deepprofiler.imaging.cropping.crop_graph(images, boxes, box_ind, None, box_size, mask_boxes=False)
    assert tuple(op.shape) == (num_crops, box_size, box_size, channels + 1)


def test_crop_generator_init(config, dataset):
    generator = deepprofiler.imaging.cropping.CropGenerator(config, dataset)
    assert generator.config == config
    assert generator.dset == dataset


def test_crop_generator_build_input_graph(crop_generator):
    crop_generator.build_input_graph()
    iv = crop_generator.input_variables
    channels = len(crop_generator.config["dataset"]["images"]["channels"])
    box_size = crop_generator.config["dataset"]["locations"]["box_size"]
    assert iv["shapes"]["batch"] == (-1,
                                     crop_generator.config["dataset"]["images"]["height"],
                                     crop_generator.config["dataset"]["images"]["width"],
                                     channels)
    assert iv["num_targets"] == len(crop_generator.dset.targets)
    assert iv["box_size"] == box_size
    assert iv["shapes"]["crops"][0] == (box_size, box_size, channels)
    assert len(iv["shapes"]["crops"]) == 1 + len(crop_generator.dset.targets)


def test_crop_generator_start(prepared_crop_generator):  # includes test for training queues
    prepared_crop_generator.start()
    assert not prepared_crop_generator.coord.joined
    assert not prepared_crop_generator.exception_occurred
    assert len(prepared_crop_generator.queue_threads) == prepared_crop_generator.config["train"]["sampling"]["workers"]
    prepared_crop_generator.stop()


def test_crop_generator_sample_batch(prepared_crop_generator):
    prepared_crop_generator.start()
    pool_index = np.zeros((prepared_crop_generator.config["train"]["model"]["params"]["batch_size"],), dtype=int)
    prepared_crop_generator.ready_to_sample = True
    data = prepared_crop_generator.sample_batch(pool_index)
    assert np.array(data[0]).shape == (prepared_crop_generator.config["train"]["model"]["params"]["batch_size"],
                                       prepared_crop_generator.config["dataset"]["locations"]["box_size"],
                                       prepared_crop_generator.config["dataset"]["locations"]["box_size"],
                                       len(prepared_crop_generator.config["dataset"]["images"]["channels"]))
    assert data[1].shape == (prepared_crop_generator.config["train"]["model"]["params"]["batch_size"], prepared_crop_generator.dset.targets[0].shape[1])
    assert data[2] == 0
    prepared_crop_generator.stop()


def test_crop_generator_generate(prepared_crop_generator):
    prepared_crop_generator.start()
    generator = prepared_crop_generator.generate()
    prepared_crop_generator.ready_to_sample = True
    test_steps = 3
    for i in range(test_steps):
        data = next(generator)
        assert np.array(data[0]).shape == (prepared_crop_generator.config["train"]["model"]["params"]["batch_size"],
                                           prepared_crop_generator.config["dataset"]["locations"]["box_size"],
                                           prepared_crop_generator.config["dataset"]["locations"]["box_size"],
                                           len(prepared_crop_generator.config["dataset"]["images"]["channels"]))
        assert len(data[1]) == len(prepared_crop_generator.dset.targets)
        for item in data[1]:
            assert item.shape == (prepared_crop_generator.config["train"]["model"]["params"]["batch_size"], prepared_crop_generator.dset.targets[0].shape[1])
    prepared_crop_generator.stop()


def test_crop_generator_stop(prepared_crop_generator):
    prepared_crop_generator.start()
    assert not prepared_crop_generator.coord.joined
    prepared_crop_generator.stop()
    assert prepared_crop_generator.coord.joined


def test_single_image_crop_generator_init(config, dataset):
    generator = deepprofiler.imaging.cropping.SingleImageCropGenerator(config, dataset)
    assert generator.config == config
    assert generator.dset == dataset


def test_single_image_crop_generator_start(single_image_crop_generator):
    single_image_crop_generator.start()
    assert single_image_crop_generator.config["train"]["model"]["params"]["batch_size"] == single_image_crop_generator.config["train"]["validation"]["batch_size"]
    assert hasattr(single_image_crop_generator, "input_variables")


def test_single_image_crop_generator_prepare_image(single_image_crop_generator, make_struct, out_dir, config):
    num_classes = len(set(single_image_crop_generator.dset.meta.data["Class"]))
    image = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
    meta = single_image_crop_generator.dset.meta.data.iloc[0]
    path = os.path.abspath(os.path.join(config["paths"]["locations"], meta["Metadata_Plate"]))
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path,
        "{}-{}-{}.csv".format(meta["Metadata_Well"],
        meta["Metadata_Site"],
        "Nuclei"))
    locations = pd.DataFrame({
        "Nuclei_Location_Center_X": np.random.randint(0, 128, 10),
        "Nuclei_Location_Center_Y": np.random.randint(0, 128, 10)
    })
    locations.to_csv(path, index=False)
    assert os.path.exists(path)
    single_image_crop_generator.start()
    crop_locations = single_image_crop_generator.prepare_image(None, image, meta)
    num_crops = len(crop_locations)
    assert num_crops == 10
    assert single_image_crop_generator.batch_size == single_image_crop_generator.config["train"]["validation"]["batch_size"]
    assert np.array(single_image_crop_generator.image_pool).shape == (10,
                                                                  single_image_crop_generator.config["dataset"]["locations"]["box_size"],
                                                                  single_image_crop_generator.config["dataset"]["locations"]["box_size"],
                                                                  len(single_image_crop_generator.config["dataset"]["images"]["channels"]))
    assert np.array(single_image_crop_generator.label_pool).shape == (10, num_classes)


def test_single_image_crop_generator_generate(single_image_crop_generator, make_struct, out_dir, config):
    num_classes = len(set(single_image_crop_generator.dset.meta.data["Class"]))
    image = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
    meta = single_image_crop_generator.dset.meta.data.iloc[0]
    path = os.path.abspath(os.path.join(config["paths"]["locations"], meta["Metadata_Plate"]))
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path,
                        "{}-{}-{}.csv".format(meta["Metadata_Well"],
                                              meta["Metadata_Site"],
                                              "Nuclei"))
    locations = pd.DataFrame({
        "Nuclei_Location_Center_X": np.random.randint(0, 128, 10),
        "Nuclei_Location_Center_Y": np.random.randint(0, 128, 10)
    })
    locations.to_csv(path, index=False)
    assert os.path.exists(path)
    single_image_crop_generator.start()
    crop_locations = single_image_crop_generator.prepare_image(None, image, meta)
    num_crops = len(crop_locations)
    for i, item in enumerate(single_image_crop_generator.generate(None)):
        assert np.array(item[0]).shape == (10,
                                       single_image_crop_generator.config["dataset"]["locations"]["box_size"],
                                       single_image_crop_generator.config["dataset"]["locations"]["box_size"],
                                       len(single_image_crop_generator.config["dataset"]["images"]["channels"]))
        assert np.array(item[1]).shape == (10, num_classes)
        assert i == 0
