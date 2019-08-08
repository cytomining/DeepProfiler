import os
import random
import json

import numpy as np
import pandas as pd
import pytest
import skimage.io
import tensorflow as tf

import deepprofiler.dataset.image_dataset
import deepprofiler.dataset.metadata
import deepprofiler.dataset.target
import deepprofiler.imaging.cropping


def test_crop_graph():
    num_crops = 100
    channels = 3
    box_size = 16
    image_ph = tf.placeholder(tf.float32, shape=(10, 128, 128, channels + 1), name="raw_images")
    boxes_ph = tf.placeholder(tf.float32, shape=(num_crops, 4), name="cell_boxes")
    box_ind_ph = tf.placeholder(tf.int32, shape=(num_crops,), name="box_indicators")
    mask_ind_ph = tf.placeholder(tf.int32, shape=(num_crops,), name="mask_indicators")
    op = deepprofiler.imaging.cropping.crop_graph(image_ph, boxes_ph, box_ind_ph, mask_ind_ph, box_size, mask_boxes=True)
    sess = tf.InteractiveSession()
    assert tuple(tf.shape(op).eval()) == (num_crops, box_size, box_size, channels)
    op = deepprofiler.imaging.cropping.crop_graph(image_ph, boxes_ph, box_ind_ph, None, box_size, mask_boxes=False)
    assert tuple(tf.shape(op).eval()) == (num_crops, box_size, box_size, channels + 1)
    sess.close()


def test_crop_generator_init(config, crop_dataset):
    generator = deepprofiler.imaging.cropping.CropGenerator(config, crop_dataset)
    assert generator.config == config
    assert generator.dset == crop_dataset


def test_crop_generator_build_input_graph(crop_generator):
    crop_generator.build_input_graph()
    assert crop_generator.input_variables["image_ph"].get_shape().as_list() == [None,
                                                                                crop_generator.config["train"]["dset"]["height"],
                                                                                crop_generator.config["train"]["dset"]["width"],
                                                                                len(crop_generator.config["dataset"]["images"]["channels"])]
    assert crop_generator.input_variables["boxes_ph"].get_shape().as_list() == [None, 4]
    assert crop_generator.input_variables["box_ind_ph"].get_shape().as_list() == [None]
    assert len(crop_generator.input_variables["targets_phs"]) == len(crop_generator.dset.targets)
    assert crop_generator.input_variables["mask_ind_ph"].get_shape().as_list() == [None]
    assert len(crop_generator.input_variables["labeled_crops"]) == 1 + len(crop_generator.dset.targets)
    assert crop_generator.input_variables["labeled_crops"][0].get_shape().as_list() == [None,
                                                                                        crop_generator.config["train"]["sampling"]["box_size"],
                                                                                        crop_generator.config["train"]["sampling"]["box_size"],
                                                                                        len(crop_generator.config["dataset"]["images"]["channels"])]
    for target in crop_generator.input_variables["labeled_crops"][1:]:
        assert target.get_shape().as_list() == [None]


def test_crop_generator_build_augmentation_graph(crop_generator):
    crop_generator.build_input_graph()
    crop_generator.build_augmentation_graph()
    assert crop_generator.train_variables["image_batch"].get_shape().as_list() == [None,
                                                                                   crop_generator.config["train"]["sampling"]["box_size"],
                                                                                   crop_generator.config["train"]["sampling"]["box_size"],
                                                                                   len(crop_generator.config["dataset"]["images"]["channels"])]


def test_crop_generator_start(prepared_crop_generator):  # includes test for training queues
    sess = tf.Session()
    prepared_crop_generator.start(sess)
    assert not prepared_crop_generator.coord.joined
    assert not prepared_crop_generator.exception_occurred
    assert len(prepared_crop_generator.queue_threads) == prepared_crop_generator.config["train"]["queueing"]["loading_workers"]
    prepared_crop_generator.stop(sess)


def test_crop_generator_sample_batch(prepared_crop_generator):
    sess = tf.Session()
    prepared_crop_generator.start(sess)
    pool_index = np.zeros((prepared_crop_generator.config["train"]["model"]["params"]["batch_size"],), dtype=int)
    prepared_crop_generator.ready_to_sample = True
    data = prepared_crop_generator.sample_batch(pool_index)
    assert np.array(data[0]).shape == (prepared_crop_generator.config["train"]["model"]["params"]["batch_size"],
                                       prepared_crop_generator.config["train"]["sampling"]["box_size"],
                                       prepared_crop_generator.config["train"]["sampling"]["box_size"],
                                       len(prepared_crop_generator.config["dataset"]["images"]["channels"]))
    assert data[1].shape == (prepared_crop_generator.config["train"]["model"]["params"]["batch_size"], prepared_crop_generator.dset.targets[0].shape[1])
    assert data[2] == 0
    prepared_crop_generator.stop(sess)


def test_crop_generator_generate(prepared_crop_generator):
    sess = tf.Session()
    prepared_crop_generator.start(sess)
    generator = prepared_crop_generator.generate(sess)
    prepared_crop_generator.ready_to_sample = True
    test_steps = 3
    for i in range(test_steps):
        data = next(generator)
        assert np.array(data[0]).shape == (prepared_crop_generator.config["train"]["model"]["params"]["batch_size"],
                                           prepared_crop_generator.config["train"]["sampling"]["box_size"],
                                           prepared_crop_generator.config["train"]["sampling"]["box_size"],
                                           len(prepared_crop_generator.config["dataset"]["images"]["channels"]))
        assert len(data[1]) == len(prepared_crop_generator.dset.targets)
        for item in data[1]:
            assert item.shape == (prepared_crop_generator.config["train"]["model"]["params"]["batch_size"], prepared_crop_generator.dset.targets[0].shape[1])
    prepared_crop_generator.stop(sess)


def test_crop_generator_stop(prepared_crop_generator):
    sess = tf.Session()
    prepared_crop_generator.start(sess)
    assert not prepared_crop_generator.coord.joined
    prepared_crop_generator.stop(sess)
    assert prepared_crop_generator.coord.joined


def test_single_image_crop_generator_init(config, crop_dataset):
    generator = deepprofiler.imaging.cropping.SingleImageCropGenerator(config, crop_dataset)
    assert generator.config == config
    assert generator.dset == crop_dataset


def test_single_image_crop_generator_start(single_image_crop_generator):
    sess = tf.Session()
    single_image_crop_generator.start(sess)
    assert single_image_crop_generator.config["train"]["model"]["params"]["batch_size"] == single_image_crop_generator.config["train"]["validation"]["batch_size"]
    assert hasattr(single_image_crop_generator, "input_variables")
    assert single_image_crop_generator.angles.get_shape().as_list() == [None]
    assert single_image_crop_generator.aligned_labeled[0].get_shape().as_list() == [None,
                                                                                    single_image_crop_generator.config["train"]["sampling"]["box_size"],
                                                                                    single_image_crop_generator.config["train"]["sampling"]["box_size"],
                                                                                    len(single_image_crop_generator.config["dataset"]["images"]["channels"])]
    assert single_image_crop_generator.aligned_labeled[1].get_shape().as_list() == [None]


def test_single_image_crop_generator_prepare_image(single_image_crop_generator, make_struct, out_dir, config):
    num_classes = len(set(single_image_crop_generator.dset.meta.data["Target"]))
    image = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
    meta = single_image_crop_generator.dset.meta.data.iloc[0]
    path = os.path.abspath(os.path.join(config["paths"]["locations"], meta["Metadata_Plate"]))
    os.makedirs(path)
    path = os.path.join(path,
        "{}-{}-{}.csv".format(meta["Metadata_Well"],
        meta["Metadata_Site"],
        single_image_crop_generator.config["train"]["sampling"]["locations_field"]))
    locations = pd.DataFrame({
        "R_Location_Center_X": np.random.randint(0, 128, (single_image_crop_generator.config["train"]["sampling"]["locations"])),
        "R_Location_Center_Y": np.random.randint(0, 128, (single_image_crop_generator.config["train"]["sampling"]["locations"]))
    })
    locations.to_csv(path, index=False)
    assert os.path.exists(path)
    sess = tf.Session()
    single_image_crop_generator.start(sess)
    num_crops = single_image_crop_generator.prepare_image(sess, image, meta)
    assert num_crops == single_image_crop_generator.config["train"]["sampling"]["locations"]
    assert single_image_crop_generator.batch_size == single_image_crop_generator.config["train"]["validation"]["batch_size"]
    assert np.array(single_image_crop_generator.image_pool).shape == (single_image_crop_generator.config["train"]["sampling"]["locations"],
                                                                      single_image_crop_generator.config["train"]["sampling"]["box_size"],
                                                                      single_image_crop_generator.config["train"]["sampling"]["box_size"],
                                                                      len(single_image_crop_generator.config["dataset"]["images"]["channels"]))
    assert np.array(single_image_crop_generator.label_pool).shape == (single_image_crop_generator.config["train"]["sampling"]["locations"], num_classes)


def test_single_image_crop_generator_generate(single_image_crop_generator, make_struct, out_dir, config):
    num_classes = len(set(single_image_crop_generator.dset.meta.data["Target"]))
    image = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
    meta = single_image_crop_generator.dset.meta.data.iloc[0]
    path = os.path.abspath(os.path.join(config["paths"]["locations"], meta["Metadata_Plate"]))
    os.makedirs(path)
    path = os.path.join(path,
                        "{}-{}-{}.csv".format(meta["Metadata_Well"],
                                              meta["Metadata_Site"],
                                              single_image_crop_generator.config["train"]["sampling"]["locations_field"]))
    locations = pd.DataFrame({
        "R_Location_Center_X": np.random.randint(0, 128, (single_image_crop_generator.config["train"]["sampling"]["locations"])),
        "R_Location_Center_Y": np.random.randint(0, 128, (single_image_crop_generator.config["train"]["sampling"]["locations"]))
    })
    locations.to_csv(path, index=False)
    assert os.path.exists(path)
    sess = tf.Session()
    single_image_crop_generator.start(sess)
    num_crops = single_image_crop_generator.prepare_image(sess, image, meta)
    for i, item in enumerate(single_image_crop_generator.generate(sess)):
        assert np.array(item[0]).shape == (single_image_crop_generator.config["train"]["sampling"]["locations"],
                                           single_image_crop_generator.config["train"]["sampling"]["box_size"],
                                           single_image_crop_generator.config["train"]["sampling"]["box_size"],
                                           len(single_image_crop_generator.config["dataset"]["images"]["channels"]))
        assert np.array(item[1]).shape == (single_image_crop_generator.config["train"]["sampling"]["locations"], num_classes)
        assert i == 0
