import os
import numpy as np
import tensorflow as tf
import pickle

import deepprofiler.imaging.cropping

import keras

class Validation(object):

    def __init__(self, config, dset, crop_generator, session):
        self.config = config
        self.dset = dset
        self.crop_generator = crop_generator
        self.session = session
        self.config["queueing"]["min_size"] = 0
        self.batch_images = []
        self.batch_labels = []

    def process_batches(self, key, image_array, meta):
        # Prepare image for cropping
        total_crops = self.crop_generator.prepare_image(
                                   self.session, 
                                   image_array, 
                                   meta, 
                                   self.config["validation"]["sample_first_crops"]
                            )
        if total_crops > 0:
            # We expect all crops in a single batch
            batches = [b for b in self.crop_generator.generate(self.session)]
            self.batch_images.append(batches[0][0])
            self.batch_labels.append(batches[0][1])

def validate(config, dset, crop_generator, session):

    validation = Validation(config, dset, crop_generator, session)
    dset.scan(validation.process_batches, frame=config["validation"]["frame"])

    validation.batch_labels = np.concatenate(validation.batch_labels)
    num_classes = np.max(validation.batch_labels) + 1
    return np.concatenate(validation.batch_images), keras.utils.to_categorical(validation.batch_labels,num_classes=num_classes)
