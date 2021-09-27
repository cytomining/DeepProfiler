import os
import numpy as np
import pandas as pd
import skimage.io
import tensorflow as tf
import tqdm

import deepprofiler.imaging.cropping

tf.compat.v1.disable_v2_behavior()

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
        self.all_cells = pd.read_csv(os.path.join(self.directory, "expanded_sc_metadata_tengenes.csv"))
        # ALPHA SET HACK:
        self.all_cells.loc[(self.all_cells.Training_Status == "Unused") & self.all_cells.Metadata_Plate.isin([41756,41757]), "Training_Status_Alpha"] = "Validation"
        self.target = config["train"]["partition"]["targets"][0]

        # Keep track of the real number of channels for internal object use
        if mode == "Training": 
            self.config["real_channels"] = config["dataset"]["images"]["channels"]
        else:
            self.num_channels = len(self.config["real_channels"])

        # Distribute channels in separate records in the reference index
        self.split_data = self.all_cells[self.all_cells.Training_Status_Alpha == self.mode].reset_index(drop=True)
        before = len(self.split_data)
        channels_data = [self.split_data.copy() for k in range(self.num_channels)]
        for k in range(self.num_channels):
            channels_data[k]["Channel"] = k
        self.split_data = pd.concat(channels_data, axis=0)
        after = len(self.split_data)
        print(" >> Records before separating channels:", before, ". After:", after)

        # Index targets for one-hot encoded labels
        self.classes = list(self.split_data[self.target].unique())
        self.num_classes = len(self.classes)
        self.classes.sort()
        self.classes = {self.classes[i]:i for i in range(self.num_classes)}

        # Identify targets and samples
        self.balanced_sample()
        self.expected_steps = (self.samples.shape[0] // self.batch_size) + int(self.samples.shape[0] % self.batch_size > 0)

        # Report number of classes and channels globally
        self.config["num_classes"] = self.num_classes
        self.config["dataset"]["images"]["channels"] = ["Individual"] # Alter the number of channels for the rest of the program!
        print(" >> Number of classes:", self.num_classes, ". Number of channels:", len(self.config["dataset"]["images"]["channels"]))


    def start(self, session):
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
            self.samples = self.samples.sample(frac=0.005).reset_index()
            print(self.samples[self.target].value_counts())


    def load_sample_image(self, pointer):
        filename = os.path.join(self.directory, self.samples.loc[pointer, "Image_Name"])
        im = skimage.io.imread(filename).astype(np.float32)
        channel = self.samples.loc[pointer, "Channel"]
        folded = deepprofiler.imaging.cropping.fold_channels(im)
        return folded[:,:,channel]


    def generator(self, sess, global_step=0):
        pointer = 0
        while True:
            x = np.zeros([self.batch_size, self.box_size, self.box_size, 1])
            y = []
            for i in range(self.batch_size):
                if pointer >= len(self.samples):
                    self.balanced_sample()
                    pointer = 0
                x[i,:,:,0] = self.load_sample_image(pointer) 
                y.append(self.classes[self.samples.loc[pointer, self.target]])
                pointer += 1
            yield(x, tf.keras.utils.to_categorical(y, num_classes=self.num_classes))


    def generate(self):
        pointer = 0
        for k in range(self.expected_steps):
            x = np.zeros([self.batch_size, self.box_size, self.box_size, 1])
            y = []
            for i in range(self.batch_size):
                if pointer >= len(self.samples):
                    break
                x[i,:,:,0] = self.load_sample_image(pointer) 
                y.append(self.classes[self.samples.loc[pointer, self.target]])
                pointer += 1
            if len(y) < x.shape[0]:
                x = x[0:len(y),...]
            yield(x, tf.keras.utils.to_categorical(y, num_classes=self.num_classes))


    def stop(self, session):
        pass

## Class for generating crops from single images with separated channels

def separate_channels(crops, network_input_size):
    #resized_crops = tf.compat.v1.image.resize_images(crops, size=(network_input_size, network_input_size))
    reordered_channels = tf.transpose(crops, [3, 0, 1, 2])
    reshaped_data = tf.reshape(reordered_channels, shape=[-1, network_input_size, network_input_size, 1])
    #rgb_data = tf.image.grayscale_to_rgb(reshaped_data)
    # Transform pixels in the range [-1,1] required for InceptionResNetv2
    #crop_min = tf.reduce_min(rgb_data, axis=[1,2,3], keepdims=True)
    #crop_max = tf.reduce_max(rgb_data, axis=[1,2,3], keepdims=True)
    #norm_rgb = ((rgb_data - crop_min)/(crop_max - crop_min))*2.0 - 1.0
    #return norm_rgb
    return reshaped_data


class SingleImageGeneratorClass(deepprofiler.imaging.cropping.SingleImageCropGenerator):

    def __init__(self, config, dset):
        # Recover the real set of channels
        config["dataset"]["images"]["channels"] = config["real_channels"]

        # Then initialize the crop generator
        super().__init__(config, dset)
        width = self.config["dataset"]["locations"]["box_size"]
        height = width
        channels = len(self.config["dataset"]["images"]["channels"])
        self.crop_ph = tf.compat.v1.placeholder(tf.float32, (None, width, height, channels))
        self.resized = separate_channels(self.crop_ph, width)

    def generate(self, session, global_step=0):
        crops = session.run(self.resized, feed_dict={self.crop_ph:self.image_pool})
        labels = np.tile(self.label_pool, [3,1])

        global_step += 1

        yield crops, labels

