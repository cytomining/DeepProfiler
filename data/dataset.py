import os
import sys
import numpy as np
import pixels as px
import utils as u

class Dataset():

    def __init__(self, metadata, dataRoot='./'):
        self.meta = metadata
        self.labels = {}
        for i in range(len(self.meta.categories)):
            self.labels[self.meta.categories[i]] = i
        self.root = dataRoot
        self.pixelProcessor = px.PixelProcessor()

    def getImagePaths(self, r):
        image = { 'DNA': self.root + r['Image_PathName_DAPI'] + '/' + r['Image_FileName_DAPI'],
                  'Tubulin': self.root + r['Image_PathName_Tubulin'] + '/' + r['Image_FileName_Tubulin'], 
                  'Actin': self.root + r['Image_PathName_Actin'] + '/' + r['Image_FileName_Actin']
                }
        return image

    def sampleImages(self, categ, nImgCat):
        images = []
        labels = []
        for c in categ:
            mask = self.meta.train['Label'] == c
            rec = self.meta.train[mask].sample(n=nImgCat, replace=True)
            for i,r in rec.iterrows():
                image = self.getImagePaths(r)
                images.append(image)
                labels.append(c)
        return images, labels

    def getTrainBatch(self, N):
        s = u.tic()
        # Batch size is N
        categ = self.meta.labels.values()
        # 1. Sample categories
        if len(categ) > N:
            np.random.shuffle(categ)
            categ = categ[0:N]
        # 2. Define images per category
        nImgCat = int(N/len(categ))
        residual = N % len(categ)
        # 3. Select images per category
        images, labels = self.sampleImages(categ, nImgCat)
        if residual > 0:
            np.random.shuffle(categ)
            ri,rl = self.sampleImages(categ[0:residual],1)
            images += ri
            labels += rl
        # 4. Open images
        batch = {'images':[], 'labels':labels}
        for img in images:
            batch['images'].append(px.openImage(img, self.pixelProcessor))
        u.toc('Loading batch', s)
        return batch

    def scan(self, f, frame='train'):
        if frame == 'all': frame = self.meta.data.iterrows()
        elif frame == 'val': frame = self.meta.val.iterrows()
        else: frame = self.meta.train.iterrows()

        images = [self.getImagePaths(r) for i,r in frame]
        for img in images:
            f(px.openImage(img, self.pixelProcessor))
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

    def nameOfLabel(self, label):
        keys = self.meta.labels.keys()
        values = self.meta.labels.values()
        idx = values.index(label)
        return keys[idx]


