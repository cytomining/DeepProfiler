import os
import numpy as np
import pandas
import glob
import skimage.transform
import pickle

import tensorflow as tf
from tensorflow.contrib import slim

import deepprofiler.learning.training
import deepprofiler.imaging.boxes
import deepprofiler.imaging.cropping
from deepprofiler.dataset.utils import tic, toc
import deepprofiler.learning.models

import keras
from keras.models import Model

num_features = 1536

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
        self.num_channels = len(self.config["image_set"]["channels"])
        crop_shape = (
                self.config["sampling"]["box_size"],      # height
                self.config["sampling"]["box_size"],      # width
                len(self.config["image_set"]["channels"]) # channels
                )
        self.raw_crops = tf.placeholder(tf.float32, shape=(None, crop_shape[0], crop_shape[1], crop_shape[2]))
        
    def configure_inception_resnet(self):

        try:
            from nets import inception
            from datasets import dataset_utils
        except:
            import sys
            print("Make sure you have installed tensorflow/models and it's accessible in the environment")
            print("export PYTHONPATH=/home/ubuntu/models/slim")
            sys.exit()

        image_size = inception.inception_resnet_v2.default_image_size

        self.crop_generator = deepprofiler.imaging.cropping.SingleImageCropGenerator(self.config, self.dset)
        # Setup pretrained model 
        network_input = crop_transform(self.raw_crops, image_size)
        url = self.config["profiling"]["url"]
        checkpoint = self.config["profiling"]["checkpoint"]
        if not os.path.isfile(checkpoint):
            dataset_utils.download_and_uncompress_tarball(url, os.path.dirname(checkpoint))
        with slim.arg_scope(inception.inception_resnet_v2_arg_scope()):
            _, self.endpoints = inception.inception_resnet_v2(network_input, num_classes=1001, is_training=False)
        init_fn = slim.assign_from_checkpoint_fn(checkpoint, slim.get_model_variables())
    
        # Session configuration
        configuration = tf.ConfigProto()
        configuration.gpu_options.allow_growth = True
        configuration.gpu_options.visible_device_list = self.config["profiling"]["gpu"]
   
        self.sess = tf.Session(config=configuration)
        init_fn(self.sess)
        self.crop_generator.start(self.sess)

    def configure_resnet(self):
         # Create model and load weights
        batch_size = self.config["validation"]["minibatch"]
        self.config["training"]["minibatch"] = batch_size
        feature_layer = self.config["profiling"]["feature_layer"]

        if self.config["model"]["type"] in ["convnet", "mixup"]:
            input_shape = (
                self.config["sampling"]["box_size"],      # height
                self.config["sampling"]["box_size"],      # width
                len(self.config["image_set"]["channels"]) # channels
            )
            self.model = deepprofiler.learning.models.create_keras_resnet(input_shape, self.dset.targets, is_training=False)
            self.crop_generator = deepprofiler.imaging.cropping.SingleImageCropGenerator(self.config, self.dset)
       
        self.model.load_weights(self.config["profiling"]["checkpoint"])

        feature_embedding = self.model.get_layer(self.config["profiling"]["feature_layer"]).output
        self.feat_extractor = keras.backend.function([self.model.input], [feature_embedding])

        # Session configuration
        configuration = tf.ConfigProto()
        configuration.gpu_options.allow_growth = True
        configuration.gpu_options.visible_device_list = self.config["profiling"]["gpu"]
   
        self.sess = tf.Session(config=configuration)
        self.crop_generator.start(self.sess)

    def check(self, meta):
        output_folder = self.config["profiling"]["output_dir"]
        if not os.path.isdir(output_folder):
            os.mkdir(self.config["profiling"]["output_dir"])
        
        output_file = self.config["profiling"]["output_dir"] + "/{}_{}_{}.npz"
        output_file = output_file.format( meta["Metadata_Plate"], meta["Metadata_Well"], meta["Metadata_Site"])

        # Check if features were computed before
        if os.path.isfile(output_file):
            print("Already done:", output_file)
            return False
        else:
            return True

    
    # Function to process a single image
    def extract_features(self, key, image_array, meta):
        output_file = self.config["profiling"]["output_dir"] + "/{}_{}_{}.npz"
        output_file = output_file.format( meta["Metadata_Plate"], meta["Metadata_Well"], meta["Metadata_Site"])

        batch_size = self.config["validation"]["minibatch"]
        image_key, image_names, outlines = self.dset.getImagePaths(meta)
        total_crops = self.crop_generator.prepare_image(
                                   self.sess,
                                   image_array,
                                   meta,
                                   self.config["validation"]["sample_first_crops"]
                            )

        # Initialize data buffer
        data = np.zeros(shape=(self.num_channels, total_crops, num_features))
        b = 0
        start = tic()

        # Extract features in batches
        batches = []
        for batch in self.crop_generator.generate(self.sess):
            crops = batch[0]
            if self.config["model"]["type"] == "inception_resnet":
                feats = self.sess.run(self.endpoints['PreLogitsFlatten'], feed_dict={self.raw_crops:crops})
            if self.config["model"]["type"] in ["convnet", "mixup"]:
                feats = self.feat_extractor((batch[0], 0))
            feats = np.reshape(feats, (self.num_channels, batch_size, num_features))
            data[:, b * batch_size:(b + 1) * batch_size, :] = feats
            b += 1
            batches.append(batch)

        # Concatentate features of all channels
        data = np.moveaxis(data, 0, 1)
        data = np.reshape(data, (data.shape[0], data.shape[1]*data.shape[2]))

        # Save features
        np.savez_compressed(output_file, f=data)
        toc(image_key + " (" + str(data.shape[0]) + " cells)", start)

        # Save crops TODO: parameterize saving crops or a sample of them.
        if False:
            batch_data = {"total_crops": total_crops, "batches": batches}
            with open(output_file.replace(".npz", ".pkl"), "wb") as batch_file:
                pickle.dump(batch_data, batch_file)

        
def profile(config, dset):
    profile = Profile(config, dset)
    if config["model"]["type"] == "inception_resnet":
        profile.configure_inception_resnet()
    if config["model"]["type"] in ["convnet", "mixup"]:
        profile.configure_resnet()
    dset.scan(profile.extract_features, frame="all", check=profile.check)
    print("Profiling: done")
