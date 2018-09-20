# TODO: test disabled because it takes too long on travis

# from comet_ml import Experiment
# import importlib
# import os
# import pytest
# import keras
# import numpy as np
# import random
# import pandas as pd
#
# import deepprofiler.imaging.cropping
# import deepprofiler.dataset.image_dataset
# import deepprofiler.dataset.metadata
# import deepprofiler.dataset.target
# import plugins.models.inception_resnet_v2
#
#
# def __rand_array():
#     return np.array(random.sample(range(100), 12))
#
#
# @pytest.fixture(scope="function")
# def out_dir(tmpdir):
#     return os.path.abspath(tmpdir.mkdir("test"))
#
#
# @pytest.fixture(scope="function")
# def config(out_dir):
#     return {
#         "model": {
#             "name": "inception_resnet_v2",
#             "crop_generator": "crop_generator",
#             "pretrained": False,
#             "params": {
#                 "epochs": 3,
#                 "steps": 10,
#                 "learning_rate": 0.0001,
#                 "batch_size": 16
#             },
#         },
#         "sampling": {
#             "images": 12,
#             "box_size": 139,
#             "locations": 2,
#             "locations_field": "R"
#         },
#         "image_set": {
#             "channels": ["R", "G", "B"],
#             "mask_objects": False,
#             "width": 256,
#             "height": 256,
#             "path": out_dir
#         },
#         "training": {
#             "learning_rate": 0.001,
#             "output": out_dir,
#             "epochs": 2,
#             "steps": 12,
#             "minibatch": 2,
#             "visible_gpus": "0"
#         },
#         "queueing": {
#             "loading_workers": 2,
#             "queue_size": 2
#         },
#         "validation": {
#             "minibatch": 2,
#             "output": out_dir,
#             "api_key":"[REDACTED]",
#             "project_name":"pytests",
#             "frame":"train",
#             "sample_first_crops": True,
#             "top_k": 2
#         }
#     }
#
#
# @pytest.fixture(scope="function")
# def metadata(out_dir):
#     filename = os.path.join(out_dir, "metadata.csv")
#     df = pd.DataFrame({
#         "Metadata_Plate": __rand_array(),
#         "Metadata_Well": __rand_array(),
#         "Metadata_Site": __rand_array(),
#         "R": [str(x) + ".png" for x in __rand_array()],
#         "G": [str(x) + ".png" for x in __rand_array()],
#         "B": [str(x) + ".png" for x in __rand_array()],
#         "Class": ["0", "1", "2", "3", "0", "1", "2", "3", "0", "1", "2", "3"],
#         "Sampling": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
#         "Split": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
#     }, dtype=int)
#     df.to_csv(filename, index=False)
#     meta = deepprofiler.dataset.metadata.Metadata(filename)
#     train_rule = lambda data: data["Split"].astype(int) == 0
#     val_rule = lambda data: data["Split"].astype(int) == 1
#     meta.splitMetadata(train_rule, val_rule)
#     return meta
#
#
# @pytest.fixture(scope="function")
# def dataset(metadata, out_dir):
#     keygen = lambda r: "{}/{}-{}".format(r["Metadata_Plate"], r["Metadata_Well"], r["Metadata_Site"])
#     dset = deepprofiler.dataset.image_dataset.ImageDataset(metadata, "Sampling", ["R", "G", "B"], out_dir, keygen)
#     target = deepprofiler.dataset.target.MetadataColumnTarget("Class", metadata.data["Class"].unique())
#     dset.add_target(target)
#     return dset
#
#
# @pytest.fixture(scope="function")
# def generator():
#     return deepprofiler.imaging.cropping.CropGenerator
#
#
# @pytest.fixture(scope="function")
# def val_generator():
#     return deepprofiler.imaging.cropping.SingleImageCropGenerator
#
#
# def test_define_model(config, dataset):
#     model, optimizer, loss = plugins.models.inception_resnet_v2.define_model(config, dataset)
#     assert isinstance(model, keras.Model)
#     assert isinstance(optimizer, str) or isinstance(optimizer, keras.optimizers.Optimizer)
#     assert isinstance(loss, str) or callable(loss)
#
#
# def test_init(config, dataset, generator, val_generator):
#     dpmodel = plugins.models.inception_resnet_v2.ModelClass(config, dataset, generator, val_generator)
#     model, optimizer, loss = plugins.models.inception_resnet_v2.define_model(config, dataset)
#     assert dpmodel.model.__eq__(model)
#     assert dpmodel.optimizer.__eq__(optimizer)
#     assert dpmodel.loss.__eq__(loss)
