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

def learn_model(config, dset, epoch):

    model_module = importlib.import_module("plugins.models.{}".format(config['model']['model_name']))
    crop_module = importlib.import_module("plugins.crop_generators.{}".format(config['model']['crop_generator_name']))
    importlib.invalidate_caches()

    model = model_module.define_model(config, dset)
    crop_generator = crop_module.define_crop_generator(config, dset)

    dpmodel = deepprofiler.learning.model.DeepProfilerModel(model, config, dset, crop_generator)

    dpmodel.train(epoch)
