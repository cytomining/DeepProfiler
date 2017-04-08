import numpy as np
import scipy.sparse
import json

from tqdm import tqdm

import dataset.pixels
import dataset.utils
import dataset.metadata


#class MultiLabelDataset():
class ImageDataset():
    def __init__(self, metadata, label_column, channels, dataRoot, keyGen):
        self.meta = metadata  # Metadata object with a valid dataframe
        self.label_column = label_column  # Column in the metadata that has category labels
        self.channels = channels  # List of column names corresponding to each channel file
        self.root = dataRoot  # Path to the directory of images
        self.keyGen = keyGen  # Function that returns the image key given its record in the metadata
        self.pixelProcessor = dataset.pixels.PixelProcessor()
        self.multi_label_matrix()

    def multi_label_matrix(self):
        all_labels = {}
        self.label_keys = set()
        print("Creating multi-label matrix")
        for i in tqdm(range(len(self.meta.data))):
            img_labels = json.loads(self.meta.data.iloc[i][self.label_column])
            all_labels[self.keyGen(self.meta.data.iloc[i])] = img_labels
            self.label_keys.update(img_labels.keys())

        self.label_keys = [i for i in map(int, list(self.label_keys))]
        self.label_keys.sort()
        self.label_image_index = {}
        rows = len(self.meta.data)
        cols = len(self.label_keys)
        L = scipy.sparse.dok_matrix((rows,cols), dtype=np.int8)
        print("Filling in sparse matrix with shape:", L.shape)
        for i in tqdm(range(len(self.meta.data))):
            img_labels = all_labels[self.keyGen(self.meta.data.iloc[i])]
            for key, value in img_labels.items():
                L[i,int(key)] = int(value)
                try: self.label_image_index[int(key)].append(i)
                except: self.label_image_index[int(key)] = [i]
        self.label_matrix = L
        

    def getImagePaths(self, r):
        key = self.keyGen(r)
        image = [self.root + '/' + r[ch] for ch in self.channels]
        return (key, image)

    def sampleImages(self, categ, nImgCat):
        keys = []
        images = []
        labels = []
        for c in categ:
            candidates = self.label_image_index[c].copy()
            np.random.shuffle(candidates)
            candidates = candidates[0:nImgCat]
            for i in candidates:
                key, image = self.getImagePaths(self.meta.data.iloc[i])
                keys.append(key)
                images.append(image)
                labels.append(self.label_matrix.getrow(i).todense())
        return keys, images, labels

    def getTrainBatch(self, N):
        #s = dataset.utils.tic()
        # Batch size is N
        categ = self.label_keys.copy()
        # 1. Sample categories
        if len(categ) > N:
            np.random.shuffle(categ)
            categ = categ[0:N]
        # 2. Define images per category
        nImgCat = int(N / len(categ))
        residual = N % len(categ)
        # 3. Select images per category
        keys, images, labels = self.sampleImages(categ, nImgCat)
        if residual > 0:
            np.random.shuffle(categ)
            rk, ri, rl = self.sampleImages(categ[0:residual], 1)
            keys += rk
            images += ri
            labels += rl
        # 4. Open images
        batch = {'keys': keys, 'images': [], 'labels': labels}
        for img in images:
            image_array = dataset.pixels.openImage(img, self.pixelProcessor)
            # TODO: Implement pixel normalization using control statistics
            image_array /= 128.0
            batch['images'].append(image_array)
        #dataset.utils.toc('Loading batch', s)
        return batch

    def scan(self, f, frame='train'):
        if frame == 'all':
            frame = self.meta.data.iterrows()
        elif frame == 'val':
            frame = self.meta.val.iterrows()
        else:
            frame = self.meta.train.iterrows()

        images = [(i, self.getImagePaths(r), r) for i, r in frame]
        for img in images:
            index = img[0]
            image = dataset.pixels.openImage(img[1][1], self.pixelProcessor)
            meta = img[2]
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
        return len(self.label_keys)

def read_dataset(config):
    # Read metadata and split dataset in training and validation
    metadata = dataset.metadata.Metadata(config["image_set"]["index"], dtype=None)
    trainingFilter = lambda df: df[config["training"]["split_field"]] == "1"
    validationFilter = lambda df: df[config["training"]["split_field"]] != "1"
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
    return dset
