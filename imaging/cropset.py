import numpy as np
import random
import pandas as pd
import tensorflow as tf


class CropSet(object):

    def __init__(self, set_size, table_size, crop_shape, target_size):
        self.set_size = set_size
        self.table_size = table_size
        self.target_size = target_size
        self.crops = np.zeros( (table_size, crop_shape[0][0], crop_shape[0][1], crop_shape[0][2]) )
        # TODO: Support for multiple targets
        self.labels = pd.DataFrame(data=np.zeros((table_size), dtype=np.int32),  columns=["target"])
        self.pointer = 0
        self.ready = False


    def add_crops(self, crops, labels):
        left_space = self.table_size - self.pointer
        load = crops.shape[0]
        if left_space >= load:
            self.crops[self.pointer:self.pointer+load, ...] = crops
            index = [i for i in range(self.pointer, self.pointer+load)]
            self.labels.loc[index, "target"] = np.argmax(labels, axis=1)
            self.pointer += load
        else:
            self.crops[self.pointer:self.pointer+left_space, ...] = crops[0:left_space, ...]
            index = [i for i in range(self.pointer, self.pointer+left_space)]
            self.labels.loc[index, "target"] = np.argmax(labels[0:left_space], axis=1)
            self.pointer = 0
            self.ready = True
            self.add_crops(crops[left_space:,...], labels[left_space:,...])


    def batch(self, batch_size):
        targets = self.labels["target"].unique()
        s, w, h, c = self.crops.shape
        data = np.zeros( (batch_size, self.set_size, w, h, c) )
        labels = np.zeros((batch_size, self.target_size))

        for i in range(batch_size):
            random.shuffle(targets)
            t = targets[0]
            sample = self.labels[self.labels["target"] == t]
            if len(sample) > self.set_size:
                sample = sample.sample(n=self.set_size, replace=False)
            else:
                sample = sample.sample(n=self.set_size, replace=True)
            index = sample.index.tolist()
            data[i,:,:,:,:] = self.crops[index, ...]
            labels[i, t] = 1.0
        return data, labels


class Mixup(CropSet):

    def __init__(self, alpha, table_size, crop_shape, target_size):
        super().__init__(2, table_size, crop_shape, target_size)
        self.alpha = alpha


    def batch(self, batch_size):
        targets = self.labels["target"].unique()
        s, w, h, c = self.crops.shape
        data = np.zeros( (batch_size, w, h, c) )
        labels = np.zeros((batch_size, self.target_size))
        
        for i in range(batch_size):
            lam = np.random.beta(self.alpha, self.alpha)
            sample = self.labels.sample(n=2)
            idx = sample.index.tolist()
            data[i,:,:,:] = lam*self.crops[idx[0],...] + (1. - lam)*self.crops[idx[1],...]
            labels[i, sample.loc[idx[0],"target"]] += lam
            labels[i, sample.loc[idx[1],"target"]] += 1. - lam
        return data, labels


class SameLabelMixup(CropSet):

    def __init__(self, alpha, table_size, crop_shape, target_size):
        super().__init__(2, table_size, crop_shape, target_size)
        self.alpha = alpha


    def batch(self, batch_size):
        targets = self.labels["target"].unique()
        s, w, h, c = self.crops.shape
        data = np.zeros( (batch_size, w, h, c) )
        labels = np.zeros((batch_size, self.target_size))

        for i in range(batch_size):
            lam = np.random.beta(self.alpha, self.alpha)
            random.shuffle(targets)
            t = targets[0]
            sample = self.labels[self.labels["target"] == t]
            if len(sample) <= 2:
                sample = sample.sample(n=2, replace=True)
            else:
                sample = sample.sample(n=2, replace=False)
            index = sample.index.tolist()
            data[i,:,:,:] = lam*self.crops[index[0], ...] + (1. - lam)*self.crops[index[1],...]
            labels[i, t] = 1.0
        return data, labels
