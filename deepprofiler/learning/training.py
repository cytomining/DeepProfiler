import gc
import os
import importlib

import tensorflow as tf
import keras

import deepprofiler.learning.model
import deepprofiler.imaging.cropping


#################################################
## MAIN TRAINING ROUTINE
#################################################

def learn_model(config, dset, epoch=1, seed=None):

    model_module = importlib.import_module("plugins.models.{}".format(config['model']['name']))
    # crop_module = importlib.import_module("plugins.crop_generators.{}".format(config['model']['crop_generator']))
    importlib.invalidate_caches()

    model = model_module.define_model(config, dset)
    # crop_generator = crop_module.define_crop_generator(config, dset)
    crop_generator = deepprofiler.imaging.cropping.CropGenerator(config, dset)

    dpmodel = deepprofiler.learning.model.DeepProfilerModel(model, config, dset, crop_generator)
    if seed:
        dpmodel.seed(seed)
    dpmodel.train(epoch)
