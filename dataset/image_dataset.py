import numpy as np
import pandas as pd

import dataset.pixels
import dataset.utils
import dataset.metadata


class ImageDataset():

    def __init__(self, metadata, category, channels, dataRoot, keyGen):
        self.meta = metadata      # Metadata object with a valid dataframe
        self.category = category  # Column in the metadata that has category labels
        self.channels = channels  # List of column names corresponding to each channel file
        self.root = dataRoot      # Path to the directory of images
        self.keyGen = keyGen      # Function that returns the image key given its record in the metadata
        self.labels = self.meta.data[self.category].unique()
        self.outlines = None

    def getImagePaths(self, r):
        key = self.keyGen(r)
        image = [self.root + '/' + r[ch] for ch in self.channels]
        outlines = self.outlines
        if outlines is not None:
            outlines = self.outlines + r["Outlines"]
        return (key, image, outlines)

    def sampleImages(self, categ, nImgCat):
        keys = []
        images = []
        labels = []
        outlines = []
        for c in categ:
            mask = self.meta.train[self.category] == c
            rec = self.meta.train[mask].sample(n=nImgCat, replace=True)
            for i, r in rec.iterrows():
                key, image, outl = self.getImagePaths(r)
                keys.append(key)
                images.append(image)
                labels.append(c)
                outlines.append(outl)
        return keys, images, labels, outlines

    def getTrainBatch(self, N):
        #s = dataset.utils.tic()
        # Batch size is N
        categ = self.labels.copy()
        # 1. Sample categories
        if len(categ) > N:
            np.random.shuffle(categ)
            categ = categ[0:N]
        # 2. Define images per category
        nImgCat = int(N / len(categ))
        residual = N % len(categ)
        # 3. Select images per category
        keys, images, labels, outlines = self.sampleImages(categ, nImgCat)
        if residual > 0:
            np.random.shuffle(categ)
            rk, ri, rl, ro = self.sampleImages(categ[0:residual], 1)
            keys += rk
            images += ri
            labels += rl
            outlines += ro
        # 4. Open images
        batch = {'keys': keys, 'images': [], 'labels': labels}
        for i in range(len(images)):
            image_array = dataset.pixels.openImage(images[i], outlines[i])
            batch['images'].append(image_array)
        #dataset.utils.toc('Loading batch', s)
        return batch

    def scan(self, f, frame='train', check=lambda k: True):
        if frame == 'all':
            frame = self.meta.data.iterrows()
        elif frame == 'val':
            frame = self.meta.val.iterrows()
        else:
            frame = self.meta.train.iterrows()

        images = [(i, self.getImagePaths(r), r) for i, r in frame]
        for img in images:
            # img => [0] index key, [1] => [0:key, 1:paths, 2:outlines], [2] => metadata
            index = img[0]
            meta = img[2]
            if check(meta):
                image = dataset.pixels.openImage(img[1][1], img[1][2])
                f(index, image, meta)
        return

    def numberOfRecords(self, dataset):
        if dataset == 'all':
            return len(self.meta.data)
        elif dataset == 'val':
            return len(self.meta.val)
        elif dataset == 'train':
            return len(self.meta.train)
        else:
            return 0

    def numberOfClasses(self):
        return len(self.labels)

def read_dataset(config):
    # Read metadata and split dataset in training and validation
    metadata = dataset.metadata.Metadata(config["image_set"]["index"], dtype=None)

    # Add outlines if specified
    outlines = None
    if "outlines" in config["image_set"].keys() and config["image_set"]["outlines"] != "":
        df = pd.read_csv(config["image_set"]["metadata"] + "/outlines.csv")
        metadata.mergeOutlines(df)
        outlines = config["image_set"]["outlines"]

    print(metadata.data.info())

    # Split training data
    split_field = config["training"]["split_field"]
    trainingFilter = lambda df: df[split_field].isin(config["training"]["training_values"])
    validationFilter = lambda df: df[split_field].isin(config["training"]["validation_values"])
    metadata.splitMetadata(trainingFilter, validationFilter)

    # Create a dataset
    keyGen = lambda r: "{}/{}-{}".format(r["Metadata_Plate"], r["Metadata_Well"], r["Metadata_Site"])
    dset = ImageDataset(
        metadata,
        config["training"]["label_field"],
        config["image_set"]["channels"],
        config["image_set"]["path"],
        keyGen
    )
    if config["image_set"]["mask_objects"]:
        dset.outlines = outlines

    return dset


