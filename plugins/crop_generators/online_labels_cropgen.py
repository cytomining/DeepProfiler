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

        ## UNCOMMENT FOR ALPHA SET
        #self.all_cells.loc[(self.all_cells.Training_Status == "Unused") & self.all_cells.Metadata_Plate.isin([41756,41757]), "Training_Status_Alpha"] = "Validation"

        ## UNCOMMENT FOR SINGLE CELL BALANCED SET
        self.all_cells.loc[self.all_cells.Training_Status == "Training", "Training_Status"] = "XXX"
        self.all_cells.loc[self.all_cells.Training_Status == "SingleCellTraining", "Training_Status"] = "Training"
        self.all_cells.loc[self.all_cells.Training_Status == "Validation", "Training_Status"] = "YYY"
        self.all_cells.loc[self.all_cells.Training_Status == "SingleCellValidation", "Training_Status"] = "Validation"

        self.target = config["train"]["partition"]["targets"][0]

        # Index targets for one-hot encoded labels
        #self.split_data = self.all_cells[self.all_cells.Training_Status_TenGenes == self.mode].reset_index(drop=True)
        self.split_data = self.all_cells[self.all_cells.Training_Status == self.mode].reset_index(drop=True)
        self.classes = list(self.split_data[self.target].unique())
        self.num_classes = len(self.classes)
        self.classes.sort()
        self.classes = {self.classes[i]: i for i in range(self.num_classes)}

        # Identify targets and samples
        self.balanced_sample()
        self.expected_steps = (self.samples.shape[0] // self.batch_size) + int(self.samples.shape[0] % self.batch_size > 0)

        # Report number of classes globally
        self.config["num_classes"] = self.num_classes
        print(" >> Number of classes:", self.num_classes)

        # Online labels
        if self.mode == "Training":
            self.out_dir = config["paths"]["results"] + "soft_labels/"
            os.makedirs(self.out_dir, exist_ok=True)
            self.init_online_labels()


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
            self.samples = self.samples.sample(frac=1.0).reset_index()
        else:
            self.samples = self.samples.sample(frac=0.1).reset_index()
            print(self.samples[self.target].value_counts())


    def generate(self, sess, global_step=0):
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
                y.append([self.soft_labels[self.samples.loc[pointer, "index"], :]])
                pointer += 1
            yield(x, np.concatenate(y, axis=0))


    def generator(self, source="samples"):
        pointer = 0
        if source == "splits":
            dataframe = self.split_data
            steps = (len(self.split_data) // self.batch_size) + int(len(self.split_data) % self.batch_size > 0)
            msg = "Predicting soft labels"
        else:
            dataframe = self.samples
            steps = self.expected_steps
            msg = "Loading validation data"

        for k in tqdm.tqdm(range(steps), desc=msg):
            x = np.zeros([self.batch_size, self.box_size, self.box_size, self.num_channels])
            y = []
            for i in range(self.batch_size):
                if pointer >= len(dataframe):
                    break
                filename = os.path.join(self.directory, dataframe.loc[pointer, "Image_Name"])
                im = skimage.io.imread(filename).astype(np.float32)
                x[i, :, :, :] = deepprofiler.imaging.cropping.fold_channels(im)
                y.append(self.classes[dataframe.loc[pointer, self.target]])
                pointer += 1
            if len(y) < x.shape[0]:
                x = x[0:len(y), ...]
            yield(x, tf.keras.utils.to_categorical(y, num_classes = self.num_classes))


    def init_online_labels(self):
        LABEL_SMOOTHING = 0.2
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
        LAMBDA = 0.01
        predictions = []

        # Get predictions with the model
        model.get_layer("augmentation_layer").is_training = False
        for batch in self.generator(source = "splits"):
            predictions.append(model.predict(batch[0]))
        model.get_layer("augmentation_layer").is_training = True

        # Update soft labels
        predictions = np.concatenate(predictions, axis=0)
        self.soft_labels = (1 - LAMBDA)*self.soft_labels + LAMBDA*predictions
        print(" >> Labels updated", predictions.shape)

        # Save labels for this epoch
        sl = pd.DataFrame(data=self.soft_labels)
        sl.to_csv(self.out_dir + "{:04d}.csv".format(epoch+1), index=False)


    def stop(self, session):
        pass

## Reusing the Single Image Crop Generator. No changes needed

SingleImageGeneratorClass = deepprofiler.imaging.cropping.SingleImageCropGenerator
