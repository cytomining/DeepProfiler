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
        self.all_cells = pd.read_csv(os.path.join(self.directory, "sc_metadata.csv"))
        #self.samples = self.samples.sample(frac=1.0).reset_index(drop=True)
        self.balanced_sample()
        self.expected_steps = self.samples.shape[0] / self.batch_size
        self.num_classes = len(self.samples["Target"].unique())


    def balanced_sample(self):
        # Obtain distribution of single cells per class
        #df = self.all_cells[self.all_cells.Training_Status_Alpha == "Training"].sample(frac=1.0).reset_index(drop=True)
        df = self.all_cells[self.all_cells.Next_Training_Status == "Training"].sample(frac=1.0).reset_index(drop=True)

        counts = df.groupby("Class_Name").count().reset_index()[["Class_Name", "Key"]]
        sample_size = int(counts.Key.median())
        counts = {r.Class_Name: r.Key for k,r in counts.iterrows()}

        # Sample the same number of cells per class
        class_samples = []
        for cls in df.Class_Name.unique():
            class_samples.append(df[df.Class_Name == cls].sample(n=sample_size, replace=counts[cls] < sample_size))
        self.samples = pd.concat(class_samples)

        # Randomize order
        self.samples = self.samples.sample(frac=1.0).reset_index(drop=True)
        print(" >> Shuffling training sample with",len(self.samples),"examples")


    def generate(self, sess, global_step=0):
        pointer = 0
        while True:
            #try:
                x = np.zeros([self.batch_size, self.box_size, self.box_size, self.num_channels])
                y = []
                for i in range(self.batch_size):
                    if pointer >= len(self.samples):
                        self.balanced_sample()
                        pointer = 0
                    filename = os.path.join(self.directory, self.samples.loc[pointer, "Image_Name"])
                    im = skimage.io.imread(filename).astype(np.float32)
                    x[i,:,:,:] = deepprofiler.imaging.cropping.fold_channels(im)
                    y.append(self.samples.loc[pointer, "Target"])
                    pointer += 1
                yield(x, tf.keras.utils.to_categorical(y, num_classes=self.num_classes))
            #except:
            #   break


    def stop(self, session):
        session.close()
        return

## Reusing the Single Image Crop Generator. No changes needed

SingleImageGeneratorClass = deepprofiler.imaging.cropping.SingleImageCropGenerator
