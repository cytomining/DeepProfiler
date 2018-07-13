import os
import random

import numpy as np
import pandas as pd
import pytest
import skimage.io
import tensorflow as tf

import deepprofiler.dataset.image_dataset
import deepprofiler.dataset.metadata
import deepprofiler.dataset.target
import deepprofiler.imaging.cropping


def __rand_array():
    return np.array(random.sample(range(100), 12))


@pytest.fixture(scope='function')
def out_dir(tmpdir):
    return os.path.abspath(tmpdir.mkdir('test'))


@pytest.fixture(scope='function')
def config(out_dir):
    return {
        "image_set": {
            "mask_objects": False,
            "channels": ['R', 'G', 'B'],
            "width": 128,
            "height": 128,
            "path": out_dir
        },
        "sampling": {
            "images": 12,
            "box_size": 16,
            "locations": 10,
            "locations_field": 'R'
        },
        "training": {
            "minibatch": 2,
            "output": out_dir
        },
        "validation": {
            "minibatch": 2,
            "output": out_dir
        },
        "queueing": {
            "loading_workers": 2,
            "queue_size": 6
        },
        "model": {
            "type": "mixup",
            "alpha": 0.2,
            "sequence_length": 5,
            "params": {
                "batch_size": 2
            }
        }
    }


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
        'Split': [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        'Target': [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
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
    dset = deepprofiler.dataset.image_dataset.ImageDataset(metadata, 'Sampling', ['R', 'G', 'B'], out_dir, keygen)
    target = deepprofiler.dataset.target.MetadataColumnTarget('Target', metadata.data['Target'].unique())
    dset.add_target(target)
    return dset


@pytest.fixture(scope='function')
def crop_generator(config, dataset):
    return deepprofiler.imaging.cropping.CropGenerator(config, dataset)


@pytest.fixture(scope='function')
def single_image_crop_generator(config, dataset):
    return deepprofiler.imaging.cropping.SingleImageCropGenerator(config, dataset)


@pytest.fixture(scope='function')
def set_crop_generator(config, dataset):
    return deepprofiler.imaging.cropping.SetCropGenerator(config, dataset)


@pytest.fixture(scope='function')
def single_image_crop_set_generator(config, dataset):
    return deepprofiler.imaging.cropping.SingleImageCropSetGenerator(config, dataset)


@pytest.fixture(scope='function')
def prepared_crop_generator(crop_generator, out_dir):
    images = np.random.randint(0, 256, (128, 128, 36), dtype=np.uint8)
    for i in range(0, 36, 3):
        skimage.io.imsave(os.path.join(out_dir, crop_generator.dset.meta.data['R'][i // 3]), images[:, :, i])
        skimage.io.imsave(os.path.join(out_dir, crop_generator.dset.meta.data['G'][i // 3]), images[:, :, i + 1])
        skimage.io.imsave(os.path.join(out_dir, crop_generator.dset.meta.data['B'][i // 3]), images[:, :, i + 2])
    crop_generator.build_input_graph()
    crop_generator.build_augmentation_graph()
    return crop_generator


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


def test_crop_generator_init(config, dataset):
    generator = deepprofiler.imaging.cropping.CropGenerator(config, dataset)
    assert generator.config == config
    assert generator.dset == dataset


def test_crop_generator_build_input_graph(crop_generator):
    crop_generator.build_input_graph()
    assert crop_generator.input_variables['image_ph'].get_shape().as_list() == [None,
                                                                                crop_generator.config['image_set']['height'],
                                                                                crop_generator.config['image_set']['width'],
                                                                                len(crop_generator.config['image_set']['channels'])]
    assert crop_generator.input_variables['boxes_ph'].get_shape().as_list() == [None, 4]
    assert crop_generator.input_variables['box_ind_ph'].get_shape().as_list() == [None]
    assert len(crop_generator.input_variables['targets_phs']) == len(crop_generator.dset.targets)
    assert crop_generator.input_variables['mask_ind_ph'].get_shape().as_list() == [None]
    assert len(crop_generator.input_variables['labeled_crops']) == 1 + len(crop_generator.dset.targets)
    assert crop_generator.input_variables['labeled_crops'][0].get_shape().as_list() == [None,
                                                                                        crop_generator.config['sampling']['box_size'],
                                                                                        crop_generator.config['sampling']['box_size'],
                                                                                        len(crop_generator.config['image_set']['channels'])]
    for target in crop_generator.input_variables['labeled_crops'][1:]:
        assert target.get_shape().as_list() == [None]


def test_crop_generator_build_augmentation_graph(crop_generator):
    crop_generator.build_input_graph()
    crop_generator.build_augmentation_graph()
    assert crop_generator.train_variables['image_batch'].get_shape().as_list() == [None,
                                                                                   crop_generator.config['sampling']['box_size'],
                                                                                   crop_generator.config['sampling']['box_size'],
                                                                                   len(crop_generator.config['image_set']['channels'])]


def test_crop_generator_start(prepared_crop_generator):  # includes test for training queues
    sess = tf.Session()
    prepared_crop_generator.start(sess)
    assert not prepared_crop_generator.coord.joined
    assert not prepared_crop_generator.exception_occurred
    assert len(prepared_crop_generator.queue_threads) == prepared_crop_generator.config['queueing']['loading_workers']
    prepared_crop_generator.stop(sess)


def test_crop_generator_sample_batch(prepared_crop_generator):
    sess = tf.Session()
    prepared_crop_generator.start(sess)
    pool_index = np.zeros((prepared_crop_generator.config["model"]["params"]['batch_size'],), dtype=int)
    prepared_crop_generator.ready_to_sample = True
    data = prepared_crop_generator.sample_batch(pool_index)
    assert np.array(data[0]).shape == (prepared_crop_generator.config["model"]["params"]['batch_size'],
                                       prepared_crop_generator.config['sampling']['box_size'],
                                       prepared_crop_generator.config['sampling']['box_size'],
                                       len(prepared_crop_generator.config['image_set']['channels']))
    assert data[1].shape == (prepared_crop_generator.config["model"]["params"]['batch_size'], prepared_crop_generator.dset.targets[0].shape[1])
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
        assert np.array(data[0]).shape == (prepared_crop_generator.config["model"]["params"]['batch_size'],
                                           prepared_crop_generator.config['sampling']['box_size'],
                                           prepared_crop_generator.config['sampling']['box_size'],
                                           len(prepared_crop_generator.config['image_set']['channels']))
        assert len(data[1]) == len(prepared_crop_generator.dset.targets)
        for item in data[1]:
            assert item.shape == (prepared_crop_generator.config["model"]["params"]['batch_size'], prepared_crop_generator.dset.targets[0].shape[1])
    prepared_crop_generator.stop(sess)


def test_crop_generator_stop(prepared_crop_generator):
    sess = tf.Session()
    prepared_crop_generator.start(sess)
    assert not prepared_crop_generator.coord.joined
    prepared_crop_generator.stop(sess)
    assert prepared_crop_generator.coord.joined


def test_single_image_crop_generator_init(config, dataset):
    generator = deepprofiler.imaging.cropping.SingleImageCropGenerator(config, dataset)
    assert generator.config == config
    assert generator.dset == dataset


def test_single_image_crop_generator_start(single_image_crop_generator):
    sess = tf.Session()
    single_image_crop_generator.start(sess)
    assert single_image_crop_generator.config["model"]["params"]["batch_size"] == single_image_crop_generator.config["validation"]["minibatch"]
    assert hasattr(single_image_crop_generator, 'input_variables')
    assert single_image_crop_generator.angles.get_shape().as_list() == [None]
    assert single_image_crop_generator.aligned_labeled[0].get_shape().as_list() == [None,
                                                                                    single_image_crop_generator.config['sampling']['box_size'],
                                                                                    single_image_crop_generator.config['sampling']['box_size'],
                                                                                    len(single_image_crop_generator.config['image_set']['channels'])]
    assert single_image_crop_generator.aligned_labeled[1].get_shape().as_list() == [None]


def test_single_image_crop_generator_prepare_image(single_image_crop_generator, tmpdir):
    image = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
    meta = single_image_crop_generator.dset.meta.data.iloc[0]
    tmpdir.mkdir(os.path.join("test", meta['Metadata_Plate']))
    path = os.path.abspath(tmpdir.mkdir(os.path.join("test", meta['Metadata_Plate'], 'locations')))
    path = os.path.join(path,
        '{}-{}-{}.csv'.format(meta['Metadata_Well'],
        meta['Metadata_Site'],
        single_image_crop_generator.config['sampling']['locations_field']))
    locations = pd.DataFrame({
        'R_Location_Center_X': np.random.randint(0, 128, (single_image_crop_generator.config['sampling']['locations'])),
        'R_Location_Center_Y': np.random.randint(0, 128, (single_image_crop_generator.config['sampling']['locations']))
    })
    locations.to_csv(path, index=False)
    assert os.path.exists(path)
    sess = tf.Session()
    single_image_crop_generator.start(sess)
    num_crops = single_image_crop_generator.prepare_image(sess, image, meta)
    assert num_crops == single_image_crop_generator.config['sampling']['locations']
    assert single_image_crop_generator.batch_size == single_image_crop_generator.config["validation"]["minibatch"]
    assert np.array(single_image_crop_generator.image_pool).shape == (single_image_crop_generator.config['sampling']['locations'],
                                                                      single_image_crop_generator.config['sampling']['box_size'],
                                                                      single_image_crop_generator.config['sampling']['box_size'],
                                                                      len(single_image_crop_generator.config['image_set']['channels']))
    assert np.array(single_image_crop_generator.label_pool).shape == (single_image_crop_generator.config['sampling']['locations'],)


def test_single_image_crop_generator_generate(single_image_crop_generator, tmpdir):
    image = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
    meta = single_image_crop_generator.dset.meta.data.iloc[0]
    tmpdir.mkdir(os.path.join("test", meta['Metadata_Plate']))
    path = os.path.abspath(tmpdir.mkdir(os.path.join("test", meta['Metadata_Plate'], 'locations')))
    path = os.path.join(path,
                        '{}-{}-{}.csv'.format(meta['Metadata_Well'],
                                              meta['Metadata_Site'],
                                              single_image_crop_generator.config['sampling']['locations_field']))
    locations = pd.DataFrame({
        'R_Location_Center_X': np.random.randint(0, 128, (single_image_crop_generator.config['sampling']['locations'])),
        'R_Location_Center_Y': np.random.randint(0, 128, (single_image_crop_generator.config['sampling']['locations']))
    })
    locations.to_csv(path, index=False)
    assert os.path.exists(path)
    sess = tf.Session()
    single_image_crop_generator.start(sess)
    num_crops = single_image_crop_generator.prepare_image(sess, image, meta)
    for i, item in enumerate(single_image_crop_generator.generate(sess)):
        assert np.array(item[0]).shape == (single_image_crop_generator.config['sampling']['locations'],
                                           single_image_crop_generator.config['sampling']['box_size'],
                                           single_image_crop_generator.config['sampling']['box_size'],
                                           len(single_image_crop_generator.config['image_set']['channels']))
        assert np.array(item[1]).shape == (single_image_crop_generator.config['sampling']['locations'],)
        assert i == 0


def test_set_crop_generator_init(config, dataset):
    generator = deepprofiler.imaging.cropping.SetCropGenerator(config, dataset)
    assert generator.config == config
    assert generator.dset == dataset


def test_set_crop_generator_start(set_crop_generator, out_dir):
    set_crop_generator = prepared_crop_generator(set_crop_generator, out_dir)
    sess = tf.Session()
    set_crop_generator.start(sess)
    assert not set_crop_generator.coord.joined
    assert not set_crop_generator.exception_occurred
    assert len(set_crop_generator.queue_threads) == set_crop_generator.config['queueing']['loading_workers']
    assert set_crop_generator.batch_size == set_crop_generator.config["model"]["params"]["batch_size"]
    assert len(set_crop_generator.target_sizes) == len(set_crop_generator.dset.targets)
    assert set_crop_generator.set_manager.alpha == set_crop_generator.config["model"]["alpha"]
    assert set_crop_generator.set_manager.table_size == set_crop_generator.config["queueing"]["queue_size"]
    assert set_crop_generator.set_manager.target_size == set_crop_generator.target_sizes[0]
    assert set_crop_generator.set_manager.crops.shape == (set_crop_generator.config["queueing"]["queue_size"],
                                                          set_crop_generator.config['sampling']['box_size'],
                                                          set_crop_generator.config['sampling']['box_size'],
                                                          len(set_crop_generator.config['image_set']['channels']))
    set_crop_generator.stop(sess)


def test_set_crop_generator_generate(set_crop_generator, out_dir):
    set_crop_generator = prepared_crop_generator(set_crop_generator, out_dir)
    sess = tf.Session()
    set_crop_generator.start(sess)
    generator = set_crop_generator.generate(sess)
    set_crop_generator.ready_to_sample = True
    test_steps = 3
    for i in range(test_steps):
        data = next(generator)
        assert np.array(data[0]).shape == (set_crop_generator.config["model"]["params"]["batch_size"],
                                           set_crop_generator.config['sampling']['box_size'],
                                           set_crop_generator.config['sampling']['box_size'],
                                           len(set_crop_generator.config['image_set']['channels']))
        assert np.array(data[1]).shape == (set_crop_generator.config["model"]["params"]['batch_size'],
                                           set_crop_generator.dset.targets[0].shape[1])
    set_crop_generator.stop(sess)


def test_single_image_crop_set_generator_init(config, dataset):
    generator = deepprofiler.imaging.cropping.SingleImageCropSetGenerator(config, dataset)
    assert generator.config == config
    assert generator.dset == dataset


def test_single_image_crop_set_generator_start(single_image_crop_set_generator):
    sess = tf.Session()
    single_image_crop_set_generator.start(sess)
    assert single_image_crop_set_generator.config["model"]["params"]["batch_size"] == single_image_crop_set_generator.config["validation"]["minibatch"]
    assert hasattr(single_image_crop_set_generator, 'input_variables')
    assert single_image_crop_set_generator.angles.get_shape().as_list() == [None]
    assert single_image_crop_set_generator.aligned_labeled[0].get_shape().as_list() == [None,
                                                                                        single_image_crop_set_generator.config['sampling']['box_size'],
                                                                                        single_image_crop_set_generator.config['sampling']['box_size'],
                                                                                        len(single_image_crop_set_generator.config['image_set']['channels'])]
    assert single_image_crop_set_generator.aligned_labeled[1].get_shape().as_list() == [None]


def test_single_image_crop_set_generator_generate(single_image_crop_set_generator, tmpdir):
    image = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
    meta = single_image_crop_set_generator.dset.meta.data.iloc[0]
    tmpdir.mkdir(os.path.join("test", meta['Metadata_Plate']))
    path = os.path.abspath(tmpdir.mkdir(os.path.join("test", meta['Metadata_Plate'], 'locations')))
    path = os.path.join(path,
                        '{}-{}-{}.csv'.format(meta['Metadata_Well'],
                                              meta['Metadata_Site'],
                                              single_image_crop_set_generator.config['sampling']['locations_field']))
    locations = pd.DataFrame({
        'R_Location_Center_X': np.random.randint(0, 128, (single_image_crop_set_generator.config['sampling']['locations'])),
        'R_Location_Center_Y': np.random.randint(0, 128, (single_image_crop_set_generator.config['sampling']['locations']))
    })
    locations.to_csv(path, index=False)
    assert os.path.exists(path)
    sess = tf.Session()
    single_image_crop_set_generator.start(sess)
    num_crops = single_image_crop_set_generator.prepare_image(sess, image, meta)
    for i, item in enumerate(single_image_crop_set_generator.generate(sess)):
        print(item[0].shape, item[1].shape)
        assert item[0].shape == (
            single_image_crop_set_generator.config["sampling"]["locations"],
            single_image_crop_set_generator.config["model"]["sequence_length"],
            single_image_crop_set_generator.config['sampling']['box_size'],
            single_image_crop_set_generator.config['sampling']['box_size'],
            len(single_image_crop_set_generator.config['image_set']['channels'])
        )
        assert item[1].shape == (single_image_crop_set_generator.config["sampling"]["locations"],)
        assert i == 0
