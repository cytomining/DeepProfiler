import os
import json
import random
import pytest
import importlib
import numpy as np
import pandas as pd
import skimage.io

import deepprofiler.dataset.image_dataset
import deepprofiler.dataset.metadata
import deepprofiler.imaging.cropping
import deepprofiler.learning.profiling
import deepprofiler.dataset.target

# Common functions and fixtures
def __rand_array():
    return np.array(random.sample(range(100), 12))


@pytest.fixture(scope="function")
def out_dir(tmpdir):
    return os.path.abspath(tmpdir.mkdir("test"))


@pytest.fixture(scope="function")
def config(out_dir):
    with open(os.path.join("tests", "files", "config", "test.json"), "r") as f:
        config = json.load(f)
    for path in config["paths"]:
        config["paths"][path] = out_dir + config["paths"].get(path)
    config["paths"]["root"] = out_dir
    return config


@pytest.fixture(scope="function")
def make_struct(config):
    for key, path in config["paths"].items():
        if key not in ["index", "config_file", "root"]:
            os.makedirs(path)
    return


# test specific fixtures
@pytest.fixture(scope="function")
def imaging_metadata(out_dir, make_struct, config):
    filename = os.path.join(config["paths"]["metadata"], "index.csv")
    df = pd.DataFrame({
        "Metadata_Plate": __rand_array(),
        "Metadata_Well": __rand_array(),
        "Metadata_Site": __rand_array(),
        "R": [str(x) + ".png" for x in __rand_array()],
        "G": [str(x) + ".png" for x in __rand_array()],
        "B": [str(x) + ".png" for x in __rand_array()],
        "Class": ["0", "1", "2", "3", "0", "1", "2", "3", "0", "1", "2", "3"],
        "Sampling": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        "Split": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    }, dtype=int)
    df.to_csv(filename, index=False)
    meta = deepprofiler.dataset.metadata.Metadata(filename)
    train_rule = lambda data: data["Split"].astype(int) == 0
    val_rule = lambda data: data["Split"].astype(int) == 1
    meta.splitMetadata(train_rule, val_rule)
    return meta


@pytest.fixture(scope="function")
def imaging_dataset(imaging_metadata, config, make_struct):
    keygen = lambda r: "{}/{}-{}".format(r["Metadata_Plate"], r["Metadata_Well"], r["Metadata_Site"])
    return deepprofiler.dataset.image_dataset.ImageDataset(imaging_metadata, "Sampling", ["R", "G", "B"], config["paths"]["root"], keygen)


@pytest.fixture(scope="function")
def boxes_dataset(imaging_metadata, config, make_struct):
    keygen = lambda r: "{}/{}-{}".format(r["Metadata_Plate"], r["Metadata_Well"], r["Metadata_Site"])
    return deepprofiler.dataset.image_dataset.ImageDataset(imaging_metadata, "Sampling", ["R", "G", "B"], config["paths"]["root"], keygen)


@pytest.fixture(scope="function")
def loadbatch(boxes_dataset, imaging_metadata, out_dir, config, make_struct):
    images = np.random.randint(0, 256, (128, 128, 36), dtype=np.uint8)
    for i in range(0, 36, 3):
        skimage.io.imsave(os.path.join(out_dir, boxes_dataset.meta.data["R"][i // 3]), images[:, :, i])
        skimage.io.imsave(os.path.join(out_dir, boxes_dataset.meta.data["G"][i // 3]), images[:, :, i + 1])
        skimage.io.imsave(os.path.join(out_dir, boxes_dataset.meta.data["B"][i // 3]), images[:, :, i + 2])
    result = deepprofiler.imaging.boxes.load_batch(boxes_dataset, config)
    return result


@pytest.fixture(scope="function")
def crop_metadata(config, make_struct):
    filename = os.path.join(config["paths"]["metadata"], "index.csv")
    df = pd.DataFrame({
        "Metadata_Plate": __rand_array(),
        "Metadata_Well": __rand_array(),
        "Metadata_Site": __rand_array(),
        "R": [str(x) + ".png" for x in __rand_array()],
        "G": [str(x) + ".png" for x in __rand_array()],
        "B": [str(x) + ".png" for x in __rand_array()],
        "Sampling": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        "Split": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        "Target": [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
    }, dtype=int)
    df.to_csv(filename, index=False)
    meta = deepprofiler.dataset.metadata.Metadata(filename)
    train_rule = lambda data: data["Split"].astype(int) == 0
    val_rule = lambda data: data["Split"].astype(int) == 1
    meta.splitMetadata(train_rule, val_rule)
    return meta


@pytest.fixture(scope="function")
def crop_dataset(crop_metadata, config, make_struct):
    keygen = lambda r: "{}/{}-{}".format(r["Metadata_Plate"], r["Metadata_Well"], r["Metadata_Site"])
    dset = deepprofiler.dataset.image_dataset.ImageDataset(crop_metadata, "Sampling", ["R", "G", "B"], config["paths"]["root"], keygen)
    target = deepprofiler.dataset.target.MetadataColumnTarget("Target", crop_metadata.data["Target"].unique())
    dset.add_target(target)
    return dset


@pytest.fixture(scope="function")
def crop_generator(config, crop_dataset):
    return deepprofiler.imaging.cropping.CropGenerator(config, crop_dataset)


@pytest.fixture(scope="function")
def single_image_crop_generator(config, crop_dataset):
    return deepprofiler.imaging.cropping.SingleImageCropGenerator(config, crop_dataset)


@pytest.fixture(scope="function")
def prepared_crop_generator(crop_generator, out_dir):
    images = np.random.randint(0, 256, (128, 128, 36), dtype=np.uint8)
    for i in range(0, 36, 3):
        skimage.io.imsave(os.path.join(out_dir, crop_generator.dset.meta.data["R"][i // 3]), images[:, :, i])
        skimage.io.imsave(os.path.join(out_dir, crop_generator.dset.meta.data["G"][i // 3]), images[:, :, i + 1])
        skimage.io.imsave(os.path.join(out_dir, crop_generator.dset.meta.data["B"][i // 3]), images[:, :, i + 2])
    crop_generator.build_input_graph()
    crop_generator.build_augmentation_graph()
    return crop_generator


@pytest.fixture(scope="function")
def model_dataset(imaging_metadata, out_dir, config, make_struct):
    keygen = lambda r: "{}/{}-{}".format(r["Metadata_Plate"], r["Metadata_Well"], r["Metadata_Site"])
    dset = deepprofiler.dataset.image_dataset.ImageDataset(imaging_metadata, "Sampling", ["R", "G", "B"], config["paths"]["root"], keygen)
    target = deepprofiler.dataset.target.MetadataColumnTarget("Class", imaging_metadata.data["Class"].unique())
    dset.add_target(target)
    return dset


@pytest.fixture(scope="function")
def model_data(imaging_metadata, out_dir, config, make_struct):
    images = np.random.randint(0, 256, (128, 128, 36), dtype=np.uint8)
    for i in range(0, 36, 3):
        skimage.io.imsave(os.path.join(config["paths"]["root"], imaging_metadata.data["R"][i // 3]), images[:, :, i])
        skimage.io.imsave(os.path.join(config["paths"]["root"], imaging_metadata.data["G"][i // 3]), images[:, :, i + 1])
        skimage.io.imsave(os.path.join(config["paths"]["root"], imaging_metadata.data["B"][i // 3]), images[:, :, i + 2])


@pytest.fixture(scope="function")
def model(config, model_dataset, crop_generator_plugin, val_crop_generator_plugin):
    def create():
        module = importlib.import_module("plugins.models.{}".format(config["train"]["model"]["name"]))
        importlib.invalidate_caches()
        dpmodel = module.ModelClass(config, model_dataset, crop_generator_plugin, val_crop_generator_plugin)
        return dpmodel
    return create


@pytest.fixture(scope="function")
def val_crop_generator(config):
    module = importlib.import_module("plugins.crop_generators.{}".format(config["train"]["model"]["crop_generator"]))
    importlib.invalidate_caches()
    generator = module.SingleImageGeneratorClass
    return generator


@pytest.fixture(scope="function")
def locations(out_dir, imaging_metadata, config, make_struct):
    for i in range(len(imaging_metadata.data.index)):
        meta = imaging_metadata.data.iloc[i]
        path = os.path.abspath(os.path.join(config["paths"]["locations"], meta["Metadata_Plate"]))
        os.makedirs(path, exist_ok=True)
        path = os.path.abspath(os.path.join(path, "{}-{}-{}.csv".format(meta["Metadata_Well"],
                                                  meta["Metadata_Site"],
                                                  config["train"]["sampling"]["locations_field"])))
        locs = pd.DataFrame({
            "R_Location_Center_X": np.random.randint(0, 128, (config["train"]["sampling"]["locations"])),
            "R_Location_Center_Y": np.random.randint(0, 128, (config["train"]["sampling"]["locations"]))
        })
        locs.to_csv(path, index=False)


@pytest.fixture(scope="function")
def checkpoint(config, crop_dataset):
    crop_generator = importlib.import_module(
        "plugins.crop_generators.{}".format(config["train"]["model"]["crop_generator"])) \
        .GeneratorClass
    profile_crop_generator = importlib.import_module(
        "plugins.crop_generators.{}".format(config["train"]["model"]["crop_generator"])) \
        .SingleImageGeneratorClass
    dpmodel = importlib.import_module("plugins.models.{}".format(config["train"]["model"]["name"])) \
        .ModelClass(config, crop_dataset, crop_generator, profile_crop_generator)
    dpmodel.feature_model.compile(dpmodel.optimizer, dpmodel.loss)
    filename = os.path.join(config["paths"]["checkpoints"], config["profile"]["checkpoint"])
    dpmodel.feature_model.save_weights(filename)
    return filename


@pytest.fixture(scope="function")
def profile(config, crop_dataset):
    return deepprofiler.learning.profiling.Profile(config, crop_dataset)


@pytest.fixture(scope="function")
def crop_generator_plugin(config):
    module = importlib.import_module("plugins.crop_generators.{}".format(config["train"]["model"]["crop_generator"]))
    importlib.invalidate_caches()
    generator = module.GeneratorClass
    return generator


@pytest.fixture(scope="function")
def val_crop_generator_plugin(config):
    module = importlib.import_module("plugins.crop_generators.{}".format(config["train"]["model"]["crop_generator"]))
    importlib.invalidate_caches()
    generator = module.SingleImageGeneratorClass
    return generator


@pytest.fixture(scope="function")
def session():
    configuration = tf.ConfigProto()
    configuration.gpu_options.visible_device_list = "0"
    session = tf.Session(config=configuration)
    return session


@pytest.fixture(scope="function")
def validation(config, dataset, crop_generator, session):
    return deepprofiler.learning.validation.Validation(config, dataset, crop_generator, session)
