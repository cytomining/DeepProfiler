from comet_ml import Experiment
import importlib
import os
import numpy as np
import pandas
import glob
import skimage.transform
import pickle

import tensorflow as tf
from tensorflow.contrib import slim
from keras import backend as K

import deepprofiler.learning.training
import deepprofiler.imaging.boxes
import deepprofiler.imaging.cropping
from deepprofiler.dataset.utils import tic, toc

import keras
from keras.models import Model


def crop_transform(crop_ph, image_size):
    crops_shape = crop_ph.shape
    resized_crops = tf.image.resize_images(crop_ph, size=(image_size, image_size))
    reordered_channels = tf.transpose(resized_crops, [3, 0, 1, 2])
    reshaped_data = tf.reshape(reordered_channels, shape=[-1, image_size, image_size, 1])
    rgb_data = tf.image.grayscale_to_rgb(reshaped_data)
    return rgb_data


class Profile(object):
    
    def __init__(self, config, dset):
        self.config = config
        self.dset = dset
        self.num_channels = len(self.config["prepare"]["images"]["channels"])
        self.crop_generator = importlib.import_module("plugins.crop_generators.{}".format(config["train"]['model']['crop_generator']))\
            .GeneratorClass
        self.profile_crop_generator = importlib.import_module(
            "plugins.crop_generators.{}".format(config["train"]['model']['crop_generator'])) \
            .SingleImageGeneratorClass
        self.dpmodel = importlib.import_module("plugins.models.{}".format(config["train"]['model']['name']))\
            .ModelClass(config, dset, self.crop_generator, self.profile_crop_generator)
        self.profile_crop_generator = self.profile_crop_generator(config, dset)


    def configure(self):
        if self.config["profile"]["checkpoint"] is not None:
            checkpoint = self.config["paths"]["checkpoints"]+"/"+self.config["profile"]["checkpoint"]
            self.dpmodel.feature_model.load_weights(checkpoint)
        self.feat_extractor = keras.Model(self.dpmodel.feature_model.inputs, self.dpmodel.feature_model.get_layer(
            self.config["profile"]["feature_layer"]).output)
        
        # Session configuration
        configuration = tf.ConfigProto()
        configuration.gpu_options.allow_growth = True
        configuration.gpu_options.visible_device_list = self.config["profile"]["gpu"]
        self.sess = tf.Session(config=configuration)
        self.profile_crop_generator.start(self.sess)


    def check(self, meta):
        output_folder = self.config["paths"]["features"]
        output_file = self.config["paths"]["features"] + "/{}_{}_{}.npz"
        output_file = output_file.format( meta["Metadata_Plate"], meta["Metadata_Well"], meta["Metadata_Site"])

        # Check if features were computed before
        if os.path.isfile(output_file):
            print("Already done:", output_file)
            return False
        else:
            return True
    
    # Function to process a single image
    def extract_features(self, key, image_array, meta):  # key is a placeholder
        output_file = self.config["paths"]["features"] + "/{}_{}_{}.npz"
        output_file = output_file.format( meta["Metadata_Plate"], meta["Metadata_Well"], meta["Metadata_Site"])

        batch_size = self.config["train"]["validation"]["batch_size"]
        image_key, image_names, outlines = self.dset.getImagePaths(meta)
        total_crops = self.profile_crop_generator.prepare_image(
                                   self.sess,
                                   image_array,
                                   meta,
                                   self.config["train"]["validation"]["sample_first_crops"]
                            )
        num_features = self.config["train"]["model"]["params"]["feature_dim"]
        repeats = "channel_repeats" in self.config["prepare"]["images"].keys()
        # Initialize data buffer
        if repeats:
            data = np.zeros(shape=(self.num_channels, total_crops, num_features))
        else:
            data = np.zeros(shape=(total_crops, num_features))
        b = 0
        start = tic()

        # Extract features in batches
        batches = []
        for batch in self.profile_crop_generator.generate(self.sess):
            crops = batch[0]
            feats = self.feat_extractor.predict(crops)
            if repeats:
                feats = np.reshape(feats, (self.num_channels, batch_size, num_features))
                data[:, b * batch_size:(b + 1) * batch_size, :] = feats
            else:
                data[b * batch_size:(b + 1) * batch_size, :] = feats
            b += 1
            batches.append(batch)

        # Save features
        np.savez_compressed(output_file, f=data)
        toc(image_key + " (" + str(data.shape[0]) + " cells)", start)

        
def profile(config, dset):
    profile = Profile(config, dset)
    profile.configure()
    dset.scan(profile.extract_features, frame="all", check=profile.check)
    print("Profiling: done")
