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
        self.directory = self.config["paths"]["single_cell_set"]
        self.num_channels = len(self.config["dataset"]["images"]["channels"])
        self.box_size = self.config["dataset"]["locations"]["box_size"]
        self.batch_size = self.config["train"]["model"]["params"]["batch_size"]
        self.mode = mode

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
        # Obtain distribution of single cells per class
        counts = self.split_data.groupby(self.target).count().reset_index()[[self.target, "Key"]]
        print(counts)
        sample_size = int(counts.Key.median())
        counts = {r[self.target]: r.Key for k, r in counts.iterrows()}

        # Sample the same number of cells per class
        class_samples = []
        for cls in self.split_data[self.target].unique():
            class_samples.append(self.split_data[self.split_data[self.target] == cls].sample(
                n=sample_size, replace=counts[cls] < sample_size))
        samples = pd.concat(class_samples)

        # Randomize order
        samples = samples.sample(frac=1.0).reset_index(drop=True)

        if False:  # TODO: Remove mock conditional to activate batching by plates permanently
            # Group batches by Plate
            samples["Plate"] = samples["Key"].str.split("/", expand=True)[0]
            samples["BatchID"] = 0
            for k, r in samples.groupby("Plate").count().iterrows():
                samples.loc[samples.Plate == k, "BatchID"] = range(r.Key)

            samples["BatchID"] = samples["BatchID"] // self.batch_size
            self.samples = samples.sort_values(by=["BatchID", "Plate"]).reset_index(drop=True)
        else:
            self.samples = samples

        # Report numbers
        if self.mode == "training":
            print(" >> Shuffling training sample with", len(self.samples), "examples")
        else:
            print(" >> Validation samples per class:", np.mean(self.samples[self.target].value_counts()))

    def generator(self, sess, global_step=0):
        pointer = 0
        image_loader = deepprofiler.dataset.utils.Parallel(
            (self.config["train"]["sampling"]["workers"], self.last_channel)
        )
        while True:
            y = []
            batch_paths = []
            for i in range(self.batch_size):
                if pointer >= len(self.samples):
                    self.balanced_sample()
                    pointer = 0

                batch_paths.append(os.path.join(self.directory, self.samples.iloc[pointer].Image_Name))
                y.append(self.classes[self.samples.loc[pointer, self.target]])
                pointer += 1

            x = np.zeros([self.batch_size, self.box_size, self.box_size, self.num_channels])
            images = image_loader.compute(load_and_crop, batch_paths)
            for i in range(len(batch_paths)):
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
            batch_paths = []
            for i in range(self.batch_size):
                if pointer >= len(self.samples):
                    pointer = 0
                    break

                batch_paths.append(os.path.join(self.directory, self.samples.iloc[pointer].Image_Name))
                y.append(self.classes[self.samples.loc[pointer, self.target]])
                pointer += 1

            x = np.zeros([self.batch_size, self.box_size, self.box_size, self.num_channels])
            images = image_loader.compute(load_and_crop, batch_paths)
            for i in range(len(batch_paths)):
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
