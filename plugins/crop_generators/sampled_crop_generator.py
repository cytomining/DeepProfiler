import os
import numpy as np
import pandas as pd
import skimage.io
import tensorflow as tf

import deepprofiler.imaging.cropping

tf.compat.v1.disable_v2_behavior()

## Wrapper for Keras ImageDataGenerator
## The Keras generator is not completely useful, because it makes assumptions about
## color (grayscale or RGB). We need flexibility for color channels, and augmentations
## tailored to multi-dimensional microscopy images. It's based on PIL rather than skimage.
## In addition, the samples loaded in this generator have unfolded channels, which
## requires us to fold them back to a tensor before feeding them to a CNN.


class GeneratorClass(deepprofiler.imaging.cropping.CropGenerator):

    def __init__(self, config, dset):
        super(GeneratorClass, self).__init__(config, dset)
        #self.datagen = tf.keras.preprocessing.image.ImageDataGenerator()
        self.directory = config["paths"]["single_cell_sample"]
        self.num_channels = len(config["dataset"]["images"]["channels"])
        self.box_size = self.config["dataset"]["locations"]["box_size"]
        self.batch_size = self.config["train"]["model"]["params"]["batch_size"]


    def start(self, session):
        self.samples = pd.read_csv(os.path.join(self.directory, "sc-metadata.csv"))
        self.samples = self.samples.sample(frac=1.0).reset_index(drop=True)
        self.num_classes = len(self.samples["Target"].unique())
        '''
        self.generator = self.datagen.flow_from_dataframe(
                dataframe=samples, 
                x_col="Image_Name",
                y_col="Class_Name",
                class_mode="categorical",
                directory=self.directory,
                color_mode="grayscale",
                target_size=(self.box_size, self.box_size * self.num_channels),
                batch_size=self.config["train"]["model"]["params"]["batch_size"]
        )
        '''

    def generate(self, sess, global_step=0):
        pointer = 0
        while True:
            #try:
                x = np.zeros([self.batch_size, self.box_size, self.box_size, self.num_channels])
                y = []
                for i in range(self.batch_size):
                    if pointer >= len(self.samples):
                        self.samples = self.samples.sample(frac=1.0).reset_index(drop=True)
                        pointer = 0
                    filename = os.path.join(self.directory, self.samples.loc[pointer, "Image_Name"])
                    im = skimage.io.imread(filename).astype(np.float32)
                    x[i,:,:,:] = deepprofiler.imaging.cropping.fold_channels(im)
                    y.append(self.samples.loc[pointer, "Target"])
                    pointer += 1
                yield(x, tf.keras.utils.to_categorical(y, num_classes=self.num_classes))
            #except:
            #   break


    def generate_old(self, sess, global_step=0):
        while True:
            try:
                x_, y = next(self.generator)
                x = np.zeros([x_.shape[0], self.box_size, self.box_size, self.num_channels])
                for i in range(x_.shape[0]):
                    x[i,:,:,:] = deepprofiler.imaging.cropping.fold_channels(x_[i])
                yield (x, y) #tf.keras.utils.to_categorical(y, num_classes=self.num_classes))
            except:
                break


    def stop(self, session):
        session.close()
        return

## Reusing the Single Image Crop Generator. No changes needed

SingleImageGeneratorClass = deepprofiler.imaging.cropping.SingleImageCropGenerator
