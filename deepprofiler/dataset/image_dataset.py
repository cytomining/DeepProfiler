import os
import numpy as np
import pandas as pd

import deepprofiler.dataset.pixels
import deepprofiler.dataset.utils
import deepprofiler.dataset.metadata
import deepprofiler.dataset.target


class ImageDataset():

    def __init__(self, metadata, sampling_field, channels, dataRoot, keyGen):
        self.meta = metadata      # Metadata object with a valid dataframe
        self.channels = channels  # List of column names corresponding to each channel file
        self.root = dataRoot      # Path to the directory of images
        self.keyGen = keyGen      # Function that returns the image key given its record in the metadata
        self.sampling_field = sampling_field # Field in the metadata used to sample images evenly
        self.sampling_values = metadata.data[sampling_field].unique()
        self.targets = []
        self.outlines = None

    def getImagePaths(self, r):
        key = self.keyGen(r)
        image = [os.path.join(self.root, r[ch]) for ch in self.channels]
        outlines = self.outlines
        if outlines is not None:
            outlines = self.outlines + r["Outlines"]
        return (key, image, outlines)

    def sampleImages(self, sampling_values, nImgCat):
        keys = []
        images = []
        targets = []
        outlines = []
        for c in sampling_values:
            mask = self.meta.train[self.sampling_field] == c
            rec = self.meta.train[mask].sample(n=nImgCat, replace=True)
            for i, r in rec.iterrows():
                key, image, outl = self.getImagePaths(r)
                keys.append(key)
                images.append(image)
                targets.append([t.get_values(r) for t in self.targets])
                outlines.append(outl)
        return keys, images, targets, outlines

    def getTrainBatch(self, N):
        #s = deepprofiler.dataset.utils.tic()
        # Batch size is N
        values = self.sampling_values.copy()
        # 1. Sample categories
        if len(values) > N:
            np.random.shuffle(values)
            values = values[0:N]

        # 2. Define images per category
        nImgCat = int(N / len(values))
        residual = N % len(values)

        # 3. Select images per category
        keys, images, targets, outlines = self.sampleImages(values, nImgCat)
        if residual > 0:
            np.random.shuffle(values)
            rk, ri, rl, ro = self.sampleImages(values[0:residual], 1)
            keys += rk
            images += ri
            targets += rl
            outlines += ro

        # 4. Open images
        batch = {"keys": keys, "images": [], "targets": targets}
        for i in range(len(images)):
            image_array = deepprofiler.dataset.pixels.openImage(images[i], outlines[i])
            # TODO: Implement pixel normalization using control statistics
            #image_array -= 128.0
            batch["images"].append(image_array)
        #dataset.utils.toc("Loading batch", s)

        return batch

    def scan(self, f, frame="train", check=lambda k: True):
        if frame == "all":
            frame = self.meta.data.iterrows()
        elif frame == "val":
            frame = self.meta.val.iterrows()
        else:
            frame = self.meta.train.iterrows()

        images = [(i, self.getImagePaths(r), r) for i, r in frame]
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
        df = pd.read_csv(os.path.join(config["paths"]["metadata"], "outlines.csv"))
        metadata.mergeOutlines(df)
        outlines = os.path.join(config["paths"]["root"], "inputs", "outlines")

    print(metadata.data.info())

    # Split training data
    split_field = config["train"]["dset"]["split_field"]
    trainingFilter = lambda df: df[split_field].isin(config["train"]["dset"]["training_values"])
    validationFilter = lambda df: df[split_field].isin(config["train"]["dset"]["validation_values"])
    metadata.splitMetadata(trainingFilter, validationFilter)

    # Create a dataset
    keyGen = lambda r: os.path.join(r["Metadata_Plate"], "{}-{}".format(r["Metadata_Well"], r["Metadata_Site"]))

    dset = ImageDataset(
        metadata=metadata,
        sampling_field=config["train"]["sampling"]["field"],
        channels=config["prepare"]["images"]["channels"],
        dataRoot=os.path.join(config["paths"]["root"], config["paths"]["images"]),
        keyGen=keyGen
    )

    # Add training targets
    for t in config["train"]["dset"]["targets"]:
        new_target = deepprofiler.dataset.target.MetadataColumnTarget(t, metadata.data[t].unique())
        dset.add_target(new_target)

    # Activate outlines for masking if needed
    if config["train"]["dset"]["mask_objects"]:
        dset.outlines = outlines

    return dset
