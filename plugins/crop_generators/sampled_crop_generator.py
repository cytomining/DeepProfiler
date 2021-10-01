import os
import numpy as np
import pandas as pd
import skimage.io
import tensorflow as tf
import tqdm

import deepprofiler.imaging.cropping

tf.compat.v1.disable_v2_behavior()
tf.config.run_functions_eagerly(False)

## Wrapper for Keras ImageDataGenerator
## The Keras generator is not completely useful, because it makes assumptions about
## color (grayscale or RGB). We need flexibility for color channels, and augmentations
## tailored to multi-dimensional microscopy images. It's based on PIL rather than skimage.
## In addition, the samples loaded in this generator have unfolded channels, which
## requires us to fold them back to a tensor before feeding them to a CNN.


class GeneratorClass(deepprofiler.imaging.cropping.CropGenerator):

    def __init__(self, config, dset, mode="Training"):
        super(GeneratorClass, self).__init__(config, dset)
        #self.datagen = tf.keras.preprocessing.image.ImageDataGenerator()
        self.directory = config["paths"]["single_cell_sample"]
        self.num_channels = len(config["dataset"]["images"]["channels"])
        self.box_size = self.config["dataset"]["locations"]["box_size"]
        self.batch_size = self.config["train"]["model"]["params"]["batch_size"]
        self.mode = mode

        # Load metadata
        self.all_cells = pd.read_csv(os.path.join(self.directory, "sc-metadata.csv"))
        self.target = "Class_Name"#config["train"]["partition"]["targets"][0]

        # Index targets for one-hot encoded labels
        self.split_data = self.all_cells[self.all_cells[self.config["train"]["partition"]["split_field"]] ==
                                         self.mode].reset_index(drop=True)



        self.classes = list(self.all_cells[self.target].unique())
        self.num_classes = len(self.classes)
        self.classes.sort()
        self.classes = {self.classes[i]: i for i in range(self.num_classes)}

        # Identify targets and samples
        self.balanced_sample()
        self.expected_steps = (self.samples.shape[0] // self.batch_size) + \
                                   int(self.samples.shape[0] % self.batch_size > 0)

        # Report number of classes globally
        self.config["num_classes"] = self.num_classes
        print(" >> Number of classes:", self.num_classes)


    def start(self, session):
        #self.all_cells = pd.read_csv(os.path.join(self.directory, "sc_metadata.csv"))
        #self.all_cells = pd.read_csv(os.path.join(self.directory, "expanded_sc_metadata_tengenes.csv"))
        #self.samples = self.samples.sample(frac=1.0).reset_index(drop=True)
        pass

    def balanced_sample(self):
        # Obtain distribution of single cells per class
        counts = self.split_data.groupby("Class_Name").count().reset_index()[["Class_Name", "Key"]]
        sample_size = int(counts.Key.median())
        counts = {r.Class_Name: r.Key for k,r in counts.iterrows()}

        # Sample the same number of cells per class
        class_samples = []
        for cls in self.split_data.Class_Name.unique():
            class_samples.append(self.split_data[self.split_data.Class_Name == cls].sample(n=sample_size, replace=counts[cls] < sample_size))
        self.samples = pd.concat(class_samples)

        # Randomize order
        if self.mode == "Training":
            print(" >> Shuffling training sample with",len(self.samples),"examples")
            self.samples = self.samples.sample(frac=1.0).reset_index()
        else:
            self.samples = self.samples.sample(frac=0.1).reset_index()
            print(self.samples[self.target].value_counts())


    def generator(self, sess, global_step=0):
        pointer = 0
        while True:
            x = np.zeros([self.batch_size, self.box_size, self.box_size, self.num_channels])
            y = []
            for i in range(self.batch_size):
                if pointer >= len(self.samples):
                    self.balanced_sample()
                    pointer = 0
                filename = os.path.join(self.directory, self.samples.loc[pointer, "Image_Name"])
                im = skimage.io.imread(filename).astype(np.float32)
                x[i,:,:,:] = deepprofiler.imaging.cropping.fold_channels(im)
                y.append(self.classes[self.samples.loc[pointer, self.target]])
                pointer += 1
            yield(x, tf.keras.utils.to_categorical(y, num_classes=self.num_classes))


    def generate(self):
        pointer = 0
        for k in range(self.expected_steps):
            x = np.zeros([self.batch_size, self.box_size, self.box_size, self.num_channels])
            y = []
            for i in range(self.batch_size):
                if pointer >= len(self.samples):
                    break
                filename = os.path.join(self.directory, self.samples.loc[pointer, "Image_Name"])
                im = skimage.io.imread(filename).astype(np.float32)
                x[i,:,:,:] = deepprofiler.imaging.cropping.fold_channels(im)
                y.append(self.classes[self.samples.loc[pointer, self.target]])
                pointer += 1
            if len(y) < x.shape[0]:
                x = x[0:len(y),...]
            yield(x, tf.keras.utils.to_categorical(y, num_classes=self.num_classes))


    def stop(self, session):
        pass

## Reusing the Single Image Crop Generator. No changes needed

SingleImageGeneratorClass = deepprofiler.imaging.cropping.SingleImageCropGenerator
