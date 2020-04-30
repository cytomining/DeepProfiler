import importlib
import os

import keras
import numpy as np
import tensorflow as tf
from keras import backend as K

from deepprofiler.dataset.utils import tic, toc


class Profile(object):
    
    def __init__(self, config, dset):
        self.config = config
        self.dset = dset
        self.num_channels = len(self.config["dataset"]["images"]["channels"])
        self.crop_generator = importlib.import_module("plugins.crop_generators.{}".format(config["train"]["model"]["crop_generator"]))\
            .GeneratorClass
        self.profile_crop_generator = importlib.import_module(
            "plugins.crop_generators.{}".format(config["train"]["model"]["crop_generator"])) \
            .SingleImageGeneratorClass
        self.dpmodel = importlib.import_module("plugins.models.{}".format(config["train"]["model"]["name"]))\
            .ModelClass(config, dset, self.crop_generator, self.profile_crop_generator)
        self.profile_crop_generator = self.profile_crop_generator(config, dset)

    def configure(self):        
        # Main session configuration
        configuration = tf.ConfigProto()
        configuration.gpu_options.allow_growth = True
        self.sess = tf.Session(config=configuration)
        self.profile_crop_generator.start(self.sess)
        K.set_session(self.sess)
        
        # Create feature extractor
        self.sess.run(tf.global_variables_initializer())
        if self.config["profile"]["checkpoint"] != "None":
            checkpoint = self.config["paths"]["checkpoints"]+"/"+self.config["profile"]["checkpoint"]
            self.dpmodel.feature_model.load_weights(checkpoint)

        self.feat_extractor = keras.Model(
            self.dpmodel.feature_model.inputs, 
            self.dpmodel.feature_model.get_layer(self.config["profile"]["feature_layer"]).output
        )
        self.feat_extractor.summary()

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
        start = tic()
        output_file = self.config["paths"]["features"] + "/{}_{}_{}.npz"
        output_file = output_file.format( meta["Metadata_Plate"], meta["Metadata_Well"], meta["Metadata_Site"])

        batch_size = self.config["profile"]["batch_size"]
        image_key, image_names, outlines = self.dset.get_image_paths(meta)
        total_crops = self.profile_crop_generator.prepare_image(
                                   self.sess,
                                   image_array,
                                   meta,
                                   False
                            )
        if total_crops == 0:
            print("No cells to profile:", output_file)
            return
        num_features = self.config["train"]["model"]["params"]["feature_dim"]
        repeats = self.config["train"]["model"]["crop_generator"] == "repeat_channel_crop_generator"
        
        # Extract features
        crops = next(self.profile_crop_generator.generate(self.sess))[0]  # single image crop generator yields one batch
        feats = self.feat_extractor.predict(crops, batch_size=batch_size)
        if repeats:
            feats = np.reshape(feats, (self.num_channels, total_crops, -1))
            feats = np.concatenate(feats, axis=-1)
            
        # Save features
        while len(feats.shape) > 2:  # 2D mean spatial pooling
            feats = np.mean(feats, axis=1)

        np.savez_compressed(output_file, f=feats)
        toc(image_key + " (" + str(total_crops) + " cells)", start)

        
def profile(config, dset):
    profile = Profile(config, dset)
    profile.configure()
    dset.scan(profile.extract_features, frame="all", check=profile.check)
    print("Profiling: done")
