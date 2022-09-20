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

    def __init__(self, config, dset, mode="training"):
        super(GeneratorClass, self).__init__(config, dset)
        self.directory = config["paths"]["single_cell_set"]
        self.num_channels = len(config["dataset"]["images"]["channels"])
        self.box_size = self.config["dataset"]["locations"]["box_size"]
        self.batch_size = self.config["train"]["model"]["params"]["batch_size"]
        self.mode = mode

        # Object masking mode and number of channels
        if self.config["dataset"]["locations"]["mask_objects"]:
            self.last_channel = 0
        else:
            self.last_channel = self.num_channels

        # Load metadata
        self.all_cells = pd.read_csv(config["paths"]["sc_index"])
        self.target = config["train"]["partition"]["targets"][0]

        # Index targets for one-hot encoded labels
        self.split_data = self.all_cells[self.all_cells[self.config["train"]["partition"]["split_field"]].isin(
                                         self.config["train"]["partition"][self.mode])].reset_index(drop=True)
        self.classes = list(self.split_data[self.target].unique())
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

        # Online labels
        if self.mode == "training":
            self.out_dir = config["paths"]["results"] + "soft_labels/"
            os.makedirs(self.out_dir, exist_ok=True)
            self.init_online_labels()

    def start(self, session):
        pass

    def balanced_sample(self):
        # Obtain distribution of single cells per class
        counts = self.split_data.groupby(self.target).count().reset_index()[[self.target, "Key"]]
        sample_size = int(counts.Key.median())
        counts = {r[self.target]: r.Key for k, r in counts.iterrows()}

        # Sample the same number of cells per class
        class_samples = []
        for cls in self.split_data[self.target].unique():
            class_samples.append(self.split_data[self.split_data[self.target] == cls].sample(
                n=sample_size, replace=counts[cls] < sample_size))
        self.samples = pd.concat(class_samples)

        # Randomize order
        self.samples = self.samples.sample(frac=1.0).reset_index()

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
                x[i, :, :, :] = deepprofiler.imaging.cropping.fold_channels(im, last_channel=self.last_channel)
                y.append([self.soft_labels[self.samples.loc[pointer, "index"], :]])
                pointer += 1
            yield x, np.concatenate(y, axis=0)

    def generate(self, sess, global_step=0):
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

    def generate_to_predict(self):
        pointer = 0
        dataframe = self.split_data
        steps = (len(self.split_data) // self.batch_size) + int(len(self.split_data) % self.batch_size > 0)
        msg = "Predicting soft labels"
        for k in tqdm.tqdm(range(steps), desc=msg):
            x = np.zeros([self.batch_size, self.box_size, self.box_size, self.num_channels])
            y = []
            for i in range(self.batch_size):
                if pointer >= len(dataframe):
                    pointer = 0
                    break

                filename = os.path.join(self.directory, dataframe.loc[pointer, "Image_Name"])
                im = skimage.io.imread(filename).astype(np.float32)
                x[i, :, :, :] = deepprofiler.imaging.cropping.fold_channels(im, last_channel=self.last_channel)
                y.append(self.classes[dataframe.loc[pointer, self.target]])
                pointer += 1
            if len(y) < x.shape[0]:
                x = x[0:len(y), ...]
            yield x, tf.keras.utils.to_categorical(y, num_classes=self.num_classes)

    def init_online_labels(self):
        LABEL_SMOOTHING = self.config["train"]["model"]["params"]["online_label_smoothing"]
        self.soft_labels = np.zeros((self.split_data.shape[0], self.num_classes)) + LABEL_SMOOTHING/self.num_classes
        print("Soft labels:", self.soft_labels.shape)
        for k, r in self.split_data.iterrows():
            label = self.classes[self.split_data.loc[k, self.target]]
            self.soft_labels[k, label] += 1. - LABEL_SMOOTHING
        print("Total labels:", np.sum(self.soft_labels))
        sl = pd.DataFrame(data=self.soft_labels)
        sl.to_csv(self.out_dir + "0000.csv", index=False)

    def update_online_labels(self, model, epoch):
        # Prepare parameters and predictions
        LAMBDA = self.config["train"]["model"]["params"]["online_lambda"]
        predictions = []

        # Get predictions with the model
        for batch in self.generate_to_predict():
            predictions.append(model.predict(batch[0]))

        # Update soft labels
        predictions = np.concatenate(predictions, axis=0)
        self.soft_labels = (1 - LAMBDA)*self.soft_labels + LAMBDA*predictions
        print(" >> Labels updated", predictions.shape)

        # Save labels for this epoch
        sl = pd.DataFrame(data=self.soft_labels)
        sl.to_csv(self.out_dir + "{:04d}.csv".format(epoch+1), index=False)

    def stop(self, session):
        pass


def load_and_crop(params):
    paths, others = params
    im = skimage.io.imread(paths).astype(np.float32)
    im = deepprofiler.imaging.cropping.fold_channels(im, last_channel=others[1])
    return im


# Reusing the Single Image Crop Generator. No changes needed
SingleImageGeneratorClass = deepprofiler.imaging.cropping.SingleImageCropGenerator
