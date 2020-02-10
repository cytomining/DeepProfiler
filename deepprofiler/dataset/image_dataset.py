import numpy as np
import pandas as pd

import deepprofiler.dataset.pixels
import deepprofiler.dataset.utils
import deepprofiler.dataset.metadata
import deepprofiler.dataset.target
import deepprofiler.imaging.boxes


class ImageLocations(object):

    def __init__(self, metadata_training, getImagePaths, targets):
        self.keys = []
        self.images = []
        self.targets = []
        self.outlines = []
        for i, r in metadata_training.iterrows():
            key, image, outl = getImagePaths(r)
            self.keys.append(key)
            self.images.append(image)
            self.targets.append([t.get_values(r) for t in targets])
            self.outlines.append(outl)
        print("Reading single-cell locations")


    def load_loc(self, params):
        # Load cell locations for one image
        i, config = params
        loc = deepprofiler.imaging.boxes.get_locations(self.keys[i], config)
        loc["ID"] = loc.index
        loc["ImageKey"] = self.keys[i]
        loc["ImagePaths"] = "#".join(self.images[i])
        loc["Target"] = self.targets[i][0]
        loc["Outlines"] = self.outlines[i]
        print("Image", i, ":", len(loc), "cells", end="\r")
        return loc


    def load_locations(self, config):
        process = deepprofiler.dataset.utils.Parallel(config, numProcs=config["train"]["sampling"]["workers"])
        data = process.compute(self.load_loc, [x for x in range(len(self.keys))])
        process.close()
        return data


class ImageDataset():

    def __init__(self, metadata, sampling_field, channels, dataRoot, keyGen, config):
        self.meta = metadata      # Metadata object with a valid dataframe
        self.channels = channels  # List of column names corresponding to each channel file
        self.root = dataRoot      # Path to the directory of images
        self.keyGen = keyGen      # Function that returns the image key given its record in the metadata
        self.sampling_field = sampling_field # Field in the metadata used to sample images evenly
        self.sampling_values = metadata.data[sampling_field].unique()
        self.targets = []
        self.outlines = None
        self.config = config
        self.load_factor = 0.25


    def get_image_paths(self, r):
        key = self.keyGen(r)
        image = [self.root + "/" + r[ch] for ch in self.channels]
        outlines = self.outlines
        if outlines is not None:
            outlines = self.outlines + r["Outlines"]
        return (key, image, outlines)

    def prepare_training_locations(self):
        image_loc = ImageLocations(self.meta.train, self.get_image_paths, self.targets)
        locations = image_loc.load_locations(self.config)

        locations = pd.concat(locations)
        self.training_images = locations.groupby(["ImageKey", "Target"])["ID"].count().reset_index()

        workers = self.config["train"]["sampling"]["workers"]
        batch_size = self.config["train"]["model"]["params"]["batch_size"]
        queue_size = self.config["train"]["sampling"]["queue_size"]
        self.sampling_factor = self.config["train"]["sampling"]["factor"]

        self.total_single_cells = len(locations)
        self.sample_images = int(np.median(self.training_images.groupby("Target").count()["ID"]))
        targets = len(self.training_images["Target"].unique())
        self.sample_locations = int(np.median(self.training_images["ID"]))
        self.cells_per_epoch = int(targets * self.sample_images * self.sample_locations * self.sampling_factor)
        self.images_per_worker = int(batch_size / workers)
        self.queue_coverage = 100*(queue_size / self.cells_per_epoch)
        self.steps_per_epoch = int(self.cells_per_epoch / batch_size)

        self.image_rotation = 0
        self.queue_records = 0
        self.shuffle_training_images()


    def show_setup(self):
        print(" || => Total single cells:", self.total_single_cells)
        print(" || => Median # of images per class:", self.sample_images)
        print(" || => Number of classes:", len(self.training_images["Target"].unique()))
        print(" || => Median # of cells per image:", self.sample_locations)
        print(" || => Single cells sampled per epoch:", self.cells_per_epoch)
        print(" || => Images sampled per worker:", self.images_per_worker)
        print(" || => Queue data coverage: {}%".format(int(self.queue_coverage)))
        print(" || => Steps per epoch:", self.steps_per_epoch)
 

    def show_stats(self):
        if (self.sample_locations * self.load_factor * self.sampling_factor) < 1.0:
            factor = 1
        else:
            factor = self.load_factor

        worker_efficiency = int(100 * (self.image_rotation / self.training_sample.shape[0]) * factor)
        queue_usage = int(100 * self.queue_records / self.cells_per_epoch)
        print("Training set coverage: {}% (worker efficiency). Data rotation: {}% (queue usage).".format(
                  worker_efficiency,
                  queue_usage)
        )
        self.image_rotation = 0
        self.queue_records = 0
        return {'worker_efficiency': worker_efficiency, 'queue_usage': queue_usage }

    def shuffle_training_images(self):
        sample = []
        for c in self.meta.train[self.sampling_field].unique():
            mask = self.meta.train[self.sampling_field] == c
            available = self.meta.train[mask].shape[0]
            rec = self.meta.train[mask].sample(n=self.sample_images, replace=available < self.sample_images)
            sample.append(rec)

        self.training_sample = pd.concat(sample)
        self.training_sample = self.training_sample.sample(frac=1.0).reset_index(drop=True)
        self.batch_pointer = 0

    def get_train_batch(self, lock):
        lock.acquire()
        df = self.training_sample[self.batch_pointer:self.batch_pointer + self.images_per_worker].copy()
        self.batch_pointer += self.images_per_worker
        self.image_rotation += self.images_per_worker
        if self.batch_pointer > self.training_sample.shape[0]:
            self.shuffle_training_images()
        lock.release()

        batch = {"keys": [], "images": [], "targets": [], "locations": []}
        sample = max(1, int(self.sample_locations * self.load_factor * self.sampling_factor))
        for k, r in df.iterrows():
            key, image, outl = self.get_image_paths(r)
            batch["keys"].append(key)
            batch["targets"].append([t.get_values(r) for t in self.targets])
            batch["images"].append(deepprofiler.dataset.pixels.openImage(image, outl))
            batch["locations"].append(deepprofiler.imaging.boxes.get_locations(key, self.config, random_sample=sample))

        return batch

    def scan(self, f, frame="train", check=lambda k: True):
        if frame == "all":
            frame = self.meta.data.iterrows()
        elif frame == "val":
            frame = self.meta.val.iterrows()
        else:
            frame = self.meta.train.iterrows()

        images = [(i, self.get_image_paths(r), r) for i, r in frame]
        for img in images:
            # img => [0] index key, [1] => [0:key, 1:paths, 2:outlines], [2] => metadata
            index = img[0]
            meta = img[2]
            if check(meta):
                image = deepprofiler.dataset.pixels.openImage(img[1][1], img[1][2])
                f(index, image, meta)
        return

    def number_of_records(self, dataset):
        if dataset == "all":
            return len(self.meta.data)
        elif dataset == "val":
            return len(self.meta.val)
        elif dataset == "train":
            return len(self.meta.train)
        else:
            return 0

    def add_target(self, new_target):
        self.targets.append(new_target)

def read_dataset(config):
    # Read metadata and split dataset in training and validation
    metadata = deepprofiler.dataset.metadata.Metadata(config["paths"]["index"], dtype=None)

    # Add outlines if specified
    outlines = None
    if "outlines" in config["prepare"].keys() and config["prepare"]["outlines"] != "":
        df = pd.read_csv(config["paths"]["metadata"] + "/outlines.csv")
        metadata.mergeOutlines(df)
        outlines = config["paths"]["root"] + "inputs/outlines/"

    print(metadata.data.info())

    # Split training data
    split_field = config["train"]["partition"]["split_field"]
    trainingFilter = lambda df: df[split_field].isin(config["train"]["partition"]["training_values"])
    validationFilter = lambda df: df[split_field].isin(config["train"]["partition"]["validation_values"])
    metadata.splitMetadata(trainingFilter, validationFilter)

    # Create a dataset
    keyGen = lambda r: "{}/{}-{}".format(r["Metadata_Plate"], r["Metadata_Well"], r["Metadata_Site"])
    dset = ImageDataset(
        metadata,
        config["dataset"]["metadata"]["label_field"],
        config["dataset"]["images"]["channels"],
        config["paths"]["images"],
        keyGen,
        config
    )

    # Add training targets
    for t in config["train"]["partition"]["targets"]:
        new_target = deepprofiler.dataset.target.MetadataColumnTarget(t, metadata.data[t].unique())
        dset.add_target(new_target)

    # Activate outlines for masking if needed
    if config["train"]["sampling"]["mask_objects"]:
        dset.outlines = outlines

    dset.prepare_training_locations()

    return dset


