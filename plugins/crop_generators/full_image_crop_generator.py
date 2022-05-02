import os
import numpy as np
import pandas as pd
import skimage.io
import tensorflow as tf

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

    def __init__(self, config, dset, mode="training"):
        super(GeneratorClass, self).__init__(config, dset)
        if self.config['prepare']['compression']['implement']:
            self.directory = self.config["paths"]["compressed_images"]
        else:
            self.directory = self.config["paths"]["images"]

        self.num_channels = len(self.config["dataset"]["images"]["channels"])
        self.box_size = self.config["dataset"]["locations"]["box_size"]
        self.batch_size = self.config["train"]["model"]["params"]["batch_size"]
        self.mode = mode

        # Load metadata
        self.all_images = pd.read_csv(self.config["paths"]["index"])
        self.target = self.config["train"]["partition"]["targets"][0]

        # Index targets for one-hot encoded labels
        self.split_data = self.all_images[self.all_images[self.config["train"]["partition"]["split_field"]].isin(
            self.config["train"]["partition"][self.mode])].reset_index(drop=True)

        self.classes = list(self.all_images[self.target].unique())
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
        pass

    def balanced_sample(self):
        # Obtain distribution of images per class
        self.split_data['Key'] = self.split_data.Metadata_Plate + '/' + \
                                 self.split_data.Metadata_Well + '/' + self.split_data.Metadata_Site
        counts = self.split_data.groupby(self.target).count().reset_index()[[self.target, 'Key']]
        print(counts)
        sample_size = int(counts.Key.median())
        counts = {r[self.target]: r.Key for k, r in counts.iterrows()}

        # Sample the same number of images per class
        class_samples = []
        for cls in self.split_data[self.target].unique():
            class_samples.append(self.split_data[self.split_data[self.target] == cls].sample(
                n=sample_size, replace=counts[cls] < sample_size))
        samples = pd.concat(class_samples)

        # Randomize order
        samples = samples.sample(frac=1.0).reset_index(drop=True)
        self.samples = samples

        # Report numbers
        if self.mode == "training":
            print(" >> Shuffling training sample with", len(self.samples), "examples")
        else:
            print(" >> Validation samples per class:", np.mean(self.samples[self.target].value_counts()))

    def generator(self, sess, global_step=0):
        pointer = 0
        while True:
            x = np.zeros([self.batch_size, self.box_size, self.box_size, self.num_channels])
            y = []
            for i in range(self.batch_size):
                if pointer >= len(self.samples):
                    self.balanced_sample()
                    pointer = 0
                im = np.zeros((self.box_size, self.box_size*self.num_channels))
                for ch in range(self.num_channels):
                    filename = os.path.join(self.directory, self.samples.loc[pointer,
                                            self.config["dataset"]["images"]["channels"][ch]].replace('.tif', '.png'))
                    a = np.resize(skimage.io.imread(filename).astype(np.float32), (self.box_size, self.box_size))
                    im[0:self.box_size, ch*self.box_size:(ch+1)*self.box_size] = a
                x[i, :, :, :] = deepprofiler.imaging.cropping.fold_channels(im, last_channel=-1)
                y.append(self.classes[self.samples.loc[pointer, self.target]])
                pointer += 1
            yield (x, tf.keras.utils.to_categorical(y, num_classes=self.num_classes))

    def generate(self):
        pointer = 0
        for k in range(self.expected_steps):
            x = np.zeros([self.batch_size, self.box_size, self.box_size, self.num_channels])
            y = []
            for i in range(self.batch_size):
                if pointer >= len(self.samples):
                    break

                im = np.zeros((self.box_size, self.box_size*self.num_channels))
                print(im.shape)
                for ch in range(self.num_channels):
                    filename = os.path.join(self.directory, self.samples.loc[pointer,
                                            self.config["dataset"]["images"]["channels"][ch]].replace('.tif', '.png'))
                    a = np.resize(skimage.io.imread(filename).astype(np.float32), (self.box_size, self.box_size))
                    im[0:self.box_size, ch*self.box_size:(ch+1)*self.box_size] = a
                x[i, :, :, :] = deepprofiler.imaging.cropping.fold_channels(im, last_channel=-1)
                y.append(self.classes[self.samples.loc[pointer, self.target]])
                pointer += 1
            if len(y) < x.shape[0]:
                x = x[0:len(y), ...]
            yield (x, tf.keras.utils.to_categorical(y, num_classes=self.num_classes))

    def stop(self, session):
        pass


## Reusing the Single Image Crop Generator. No changes needed

SingleImageGeneratorClass = deepprofiler.imaging.cropping.SingleImageCropGenerator
