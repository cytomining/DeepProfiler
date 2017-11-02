import numpy as np
import random
import pandas as pd
import tensorflow as tf


class CropSet(object):

    def __init__(self, set_size, table_size, crop_shape):
        self.set_size
        self.table_size = table_size
        self.crops = np.zeros( (table_size, crop_shape[0][0], crop_shape[0][1], crop_shape[0][2]) )
        # TODO: Support for multiple targets
        self.labels = pd.DataFrame(data=np.zeros((table_size)),  columns=["target"])
        self.pointer = 0
        self.ready = False


    def add_crops(self, crops, labels):
        for i in range(crops.shape[0]):
            self.crops[self.pointer, ...] = crops[i]
            self.labels.loc[self.pointer, "target"] = labels[i]
            self.pointer += 1
            if self.pointer >= self.size:
                self.pointer = 0
                self.ready = True


    def batch(self, batch_size):
        targets = self.labels["target"].unique()
        random.shuffle(targets)
        s, w, h, c = self.crops.shape
        data = np.zeros( (batch_size, self.set_size, w, h, c) )
        labels = np.zeros((batch_size))

        for i in range(batch_size):
            t = targets[i]
            sample = self.labels[self.labels["target"] == t].sample(n=self.set_size)
            data[i,:,:,:,:] = self.crops[sample.index, ...]
            labels[i] = t
        return data
