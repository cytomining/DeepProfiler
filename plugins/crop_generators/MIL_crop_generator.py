import os
import numpy as np
import pandas as pd
import skimage.io
import tensorflow as tf
import random
from tqdm import tqdm

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
        self.directory = self.config["paths"]["single_cell_set"]
        self.num_channels = len(self.config["dataset"]["images"]["channels"])
        self.box_size = self.config["dataset"]["locations"]["box_size"]
        self.batch_size = self.config["train"]["model"]["params"]["batch_size"]
        self.mode = mode
        self.bag_size = self.config["train"]["model"]["params"]["bag_size"]

        # Object masking mode and number of channels
        if self.config["dataset"]["locations"]["mask_objects"]:
            self.last_channel = 0
        else:
            self.last_channel = self.num_channels

        # Load metadata
        self.all_cells = pd.read_csv(self.config["paths"]["sc_index"])
        self.target = self.config["train"]["partition"]["targets"][0]

        # Index targets for one-hot encoded labels
        self.split_data = self.all_cells[self.all_cells[self.config["train"]["partition"]["split_field"]].isin(
            self.config["train"]["partition"][self.mode])].reset_index(drop=True)

        self.classes = list(self.all_cells[self.target].unique())
        self.num_classes = len(self.classes)
        self.classes.sort()
        self.classes = {self.classes[i]: i for i in range(self.num_classes)}

        self.controls = self.split_data[self.split_data[self.target] ==
                                        self.config["dataset"]["metadata"]["control_value"]]

        # Identify targets and samples
        self.create_bags()
        self.expected_steps = len(self.bags)
        shuffle = list(zip(self.bags, self.bags_labels))
        random.shuffle(shuffle)
        self.bags, self.bags_labels = zip(*shuffle)
        # Report number of classes globally
        self.config["num_classes"] = self.num_classes
        print(" >> Number of classes:", self.num_classes)

    def start(self, session):
        pass

    def create_bags(self):
        # Obtain distribution of single cells per positive class
        self.bags = []
        self.bags_labels = []
        counts = self.split_data[self.split_data[self.target] != self.config["dataset"]["metadata"][
            "control_value"]].groupby(self.target).count().reset_index()[[self.target, "Key"]]

        sample_size = int(counts.Key.median())
        for c in tqdm(self.classes, desc="Creating bags"):
            if c == self.config["dataset"]["metadata"]["control_value"]:
                for __ in range(sample_size):
                    sample = self.controls.sample(self.bag_size, replace=False)
                    self.bags.append(sample)
                    self.bags_labels.append(self.classes[c])
            else:
                for __ in range(sample_size):
                    positive_samples_count = random.randint(1, self.bag_size-1)
                    negative_samples_count = self.bag_size-positive_samples_count
                    positive_samples = self.split_data[self.split_data[self.target] == c].sample(
                        positive_samples_count, replace=False)
                    negative_samples = self.controls.sample(negative_samples_count, replace=False)
                    sample = pd.concat((positive_samples, negative_samples))
                    self.bags.append(sample)
                    self.bags_labels.append(self.classes[c])

        # Report numbers
        if self.mode == "training":
            print(" >> Number of training bags", len(self.bags))
        else:
            print(" >> Number of validation bags", len(self.bags))

    def generator(self, sess, global_step=0):
        pointer = 0
        image_loader = deepprofiler.dataset.utils.Parallel(
            (self.config["train"]["sampling"]["workers"], self.last_channel)
        )
        while True:
            y = []
            bag_paths = []
            for i in range(1):
                if pointer >= len(self.bags):
                    self.create_bags()
                    pointer = 0

                for j in range(len(self.bags[pointer])):
                    bag_paths.append(os.path.join(self.directory, self.bags[pointer].iloc[j].Image_Name))
                    y.append(self.bags_labels[pointer])

                pointer += 1

            x = np.zeros([self.bag_size, self.box_size, self.box_size, self.num_channels])
            images = image_loader.compute(load_and_crop, bag_paths)
            print(len(self.bags_labels), len(self.bags), pointer)
            for i in range(len(bag_paths)):
                x[i, :, :, :] = images[i]
            yield x, tf.keras.utils.to_categorical(y, num_classes=self.num_classes)

        image_loader.close()

    def generate(self, sess):
        pointer = 0
        image_loader = deepprofiler.dataset.utils.Parallel(
            (self.config["train"]["sampling"]["workers"], self.last_channel)
        )
        while True:
            y = []
            bag_paths = []
            for i in range(1):
                if pointer >= len(self.bags):
                    self.create_bags()
                    pointer = 0

                for j in range(len(self.bags[pointer])):
                    bag_paths.append(os.path.join(self.directory, self.bags[pointer].iloc[j].Image_Name))
                    y.append(self.bags_labels[pointer])

                pointer += 1

            x = np.zeros([self.bag_size, self.box_size, self.box_size, self.num_channels])
            images = image_loader.compute(load_and_crop, bag_paths)
            for i in range(len(bag_paths)):
                x[i, :, :, :] = images[i]

            if len(y) < x.shape[0]:
                x = x[0:len(y), ...]
            yield x, tf.keras.utils.to_categorical(y, num_classes=self.num_classes)
        image_loader.close()

    def stop(self, session):
        pass


def load_and_crop(params):
    paths, others = params
    im = skimage.io.imread(paths).astype(np.float32)
    im = deepprofiler.imaging.cropping.fold_channels(im, last_channel=others[1])
    return im


# Reusing the Single Image Crop Generator. No changes needed
SingleImageGeneratorClass = deepprofiler.imaging.cropping.SingleImageCropGenerator
