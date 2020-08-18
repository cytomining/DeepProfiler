import os
import numpy as np
import pandas as pd
import tensorflow as tf

import deepprofiler.imaging.cropping

## Wrapper for Keras ImageDataGenerator
## The Keras generator is not completely useful, because it makes assumptions about
## color (grayscale or RGB). We need flexibility for color channels, and augmentations
## tailored to multi-dimensional microscopy images. It's based on PIL rather than skimage.
## In addition, the samples loaded in this generator have unfolded channels, which
## requires us to fold them back to a tensor before feeding them to a CNN.

class GeneratorClass(deepprofiler.imaging.cropping.CropGenerator):

    def __init__(self, config, dset):
        super(GeneratorClass, self).__init__(config, dset)
        self.datagen = tf.keras.preprocessing.image.ImageDataGenerator()
        self.directory = config["paths"]["single_cell_sample"]
        self.num_channels = len(config["dataset"]["images"]["channels"])
        self.box_size = config["dataset"]["locations"]["box_size"]
        
    def start(self, session):
        samples = pd.read_csv(os.path.join(self.directory, "sc-metadata.csv"))
        self.num_classes = len(samples["Target"].unique())
        self.generator = self.datagen.flow_from_dataframe(
                dataframe=samples, 
                x_col="Image_Name",
                y_col="Target",
                class_mode="raw",
                directory=self.directory,
                color_mode="grayscale",
                target_size=(self.box_size, self.box_size * self.num_channels),
                batch_size=self.config["train"]["model"]["params"]["batch_size"]
        )

    def generate(self, sess, global_step=0):
        while True:
            try:
                x_, y = next(self.generator)
                x = np.zeros([x_.shape[0], self.box_size, self.box_size, self.num_channels])
                for i in range(x_.shape[0]):
                    x[i,:,:,:] = deepprofiler.imaging.cropping.fold_channels(x_[i])
                yield (x, tf.keras.utils.to_categorical(y, num_classes=self.num_classes))
            except:
                break

    def stop(self, session):
        session.close()
        return

## TODO: Next steps:
## 1. Fix the session closing error at the end (not a big deal for now, but better to fix it)
## 2. DONE => Integrate augmentations (this is important)
## 3. Reconsider the cache usage statistics and steps per epoch in ImageDataset

## Reusing the Single Image Crop Generator. No changes needed

SingleImageGeneratorClass = deepprofiler.imaging.cropping.SingleImageCropGenerator
