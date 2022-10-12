import os
import numpy as np
import pandas as pd
import skimage.io
import tensorflow as tf
import multiprocessing

import deepprofiler.imaging.cropping
import deepprofiler.dataset.pixels
import deepprofiler.dataset.utils

tf.compat.v1.disable_v2_behavior()
tf.config.run_functions_eagerly(False)


class GeneratorClass(deepprofiler.imaging.cropping.CropGenerator):

    def __init__(self, config, dset, mode="training"):
        super(GeneratorClass, self).__init__(config, dset)
        self.num_channels = len(self.config["dataset"]["images"]["channels"])
        self.box_size = self.config["dataset"]["locations"]["box_size"]
        self.view_size = self.config["dataset"]["locations"]["view_size"]
        self.batch_size = self.config["train"]["model"]["params"]["batch_size"]
        self.mode = mode

        # Load metadata
        self.all_images = pd.read_csv(self.config["paths"]["index"])
        if self.config['prepare']['compression']['implement']:
            self.directory = self.config["paths"]["compressed_images"]
            self.all_images.replace({'.tiff': '.png', '.tif': '.png'}, inplace=True, regex=True)
        else:
            self.directory = self.config["paths"]["images"]

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
        counts = self.split_data.groupby(self.target).count().reset_index()[[self.target, 'Metadata_Site']]
        sample_size = int(counts.Metadata_Site.median())
        counts = {r[self.target]: r.Metadata_Site for k, r in counts.iterrows()}

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

    def get_image_paths(self, r):
        return [os.path.join(self.directory, r[ch]) for ch in self.config["dataset"]["images"]["channels"]]

    def generator(self, sess, global_step=0):
        pointer = 0
        image_loader = deepprofiler.dataset.utils.Parallel(
                [self.config["dataset"]["locations"]["view_size"], True],
                self.config["train"]["sampling"]["workers"]
        )
        while True:
            # Prepare batch metadata
            y = []
            batch_paths = []
            for i in range(self.batch_size):
                if pointer >= len(self.samples):
                    self.balanced_sample()
                    pointer = 0

                batch_paths.append(self.get_image_paths(self.samples.iloc[pointer]))
                y.append(self.classes[self.samples.loc[pointer, self.target]])
                pointer += 1

            # Load batch images
            x = np.zeros([self.batch_size, self.view_size, self.view_size, self.num_channels])
            images = image_loader.compute(load_and_crop, batch_paths)
            for i in range(len(batch_paths)):
                x[i, :, :, :] = images[i]

            inputs = [x, np.asarray([[0, 0, 1, 1]]*len(y)), np.arange(0, len(y))]
            yield inputs, tf.keras.utils.to_categorical(y, num_classes=self.num_classes)

        image_loader.close()

    def generate(self, sess):
        pointer = 0
        image_loader = deepprofiler.dataset.utils.Parallel(
                [self.config["dataset"]["locations"]["view_size"], False], 
                self.config["train"]["sampling"]["workers"]
        )
        while True:
            # Prepare metadata
            y = []
            batch_paths = []
            for i in range(self.batch_size):
                if pointer >= len(self.samples):
                    pointer = 0
                    break

                batch_paths.append(self.get_image_paths(self.samples.iloc[pointer]))
                y.append(self.classes[self.samples.loc[pointer, self.target]])
                pointer += 1

            # Load images
            x = np.zeros([self.batch_size, self.view_size, self.view_size, self.num_channels])
            images = image_loader.compute(load_and_crop, batch_paths)
            for i in range(len(images)):
                x[i, :, :, :] = images[i] 

            if len(y) < x.shape[0]:
                x = x[0:len(y), ...]

            inputs = [x, np.asarray([[0, 0, 1, 1]]*len(y)), np.arange(0, len(y))]
            yield inputs, tf.keras.utils.to_categorical(y, num_classes=self.num_classes)

        image_loader.close()

    def stop(self, session):
        pass


def load_and_crop(params):
    paths, others = params
    view_size, random = others
    im = deepprofiler.dataset.pixels.openImage(paths, None)
    im = im / 255.
    H, W, C = im.shape
    if random:
        q = np.random.randint(0, H - view_size)
    else:
        q = (H - view_size)//2
    return im[q:q+view_size, q:q+view_size, :]


# Reusing the Single Image Crop Generator. No changes needed
SingleImageGeneratorClass = deepprofiler.imaging.cropping.SingleImageCropGenerator
