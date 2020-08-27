import os

import numpy as np
import pandas as pd
import skimage.io
import tensorflow as tf

import deepprofiler.dataset.image_dataset
import deepprofiler.dataset.metadata
import deepprofiler.dataset.target
import deepprofiler.imaging.cropping
import plugins.crop_generators.autoencoder_crop_generator


def test_autoencoder_crop_generator():
    assert issubclass(plugins.crop_generators.autoencoder_crop_generator.GeneratorClass, deepprofiler.imaging.cropping.CropGenerator)
    assert issubclass(plugins.crop_generators.autoencoder_crop_generator.SingleImageGeneratorClass, deepprofiler.imaging.cropping.SingleImageCropGenerator)


def test_generator_class_generate(config, dataset, out_dir):
    crop_generator = plugins.crop_generators.autoencoder_crop_generator.GeneratorClass(config, dataset)
    images = np.random.randint(0, 256, (128, 128, 36), dtype=np.uint8)
    for i in range(0, 36, 3):
        skimage.io.imsave(os.path.join(out_dir, crop_generator.dset.meta.data["R"][i // 3]), images[:, :, i])
        skimage.io.imsave(os.path.join(out_dir, crop_generator.dset.meta.data["G"][i // 3]), images[:, :, i + 1])
        skimage.io.imsave(os.path.join(out_dir, crop_generator.dset.meta.data["B"][i // 3]), images[:, :, i + 2])
    crop_generator.build_input_graph()
    crop_generator.build_augmentation_graph()
    sess = tf.Session()
    crop_generator.start(sess)
    generator = crop_generator.generate(sess)
    crop_generator.ready_to_sample = True
    test_steps = 3
    for i in range(test_steps):
        data = next(generator)
        np.testing.assert_array_equal(data[0], data[1])
    crop_generator.stop(sess)


def test_single_image_generator_class_generate(config, dataset, tmpdir):
    crop_generator = plugins.crop_generators.autoencoder_crop_generator.SingleImageGeneratorClass(config, dataset)
    image = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
    meta = crop_generator.dset.meta.data.iloc[0]
    tmpdir.mkdir(os.path.join("test", meta["Metadata_Plate"]))
    path = os.path.abspath(tmpdir.mkdir(os.path.join("test", meta["Metadata_Plate"], "locations")))
    path = os.path.join(path,
                        "{}-{}-{}.csv".format(meta["Metadata_Well"],
                                              meta["Metadata_Site"],
                                              "Nuclei"))
    locations = pd.DataFrame({
        "R_Location_Center_X": np.random.randint(0, 128, 10),
        "R_Location_Center_Y": np.random.randint(0, 128, 10)
    })
    locations.to_csv(path, index=False)
    assert os.path.exists(path)
    sess = tf.Session()
    crop_generator.start(sess)
    crop_locations = crop_generator.prepare_image(sess, image, meta)
    num_crops = len(crop_locations)
    for i, item in enumerate(crop_generator.generate(sess)):
        np.testing.assert_array_equal(item[0], item[1])
        assert i == 0
