import os
import sys
import pandas as pd

def replicate(df, value):
    if value == '3':
        return df.Replicate == 3
    else:
        return df.Replicate < 3

class ImageMetadata():

    def __init__(self, filename=None):
        if filename is not None:
            self.load(filename)

    def load(self, filename):
        print 'Reading metadata form',filename
        self.data = pd.read_csv(filename)

    def splitByReplicate(self, records):
        # One replicate for validation, two for training
        self.train = records[replicate(records,'1,2')].copy()
        self.val = records[replicate(records,'3')].copy()
        self.categories = self.train['Category'].unique()
        self.labels = dict([(self.categories[i],i) for i in range(len(self.categories))])
        self.train['Label'] = self.train['Category']
        self.train = self.train.replace({'Label':self.labels})
        self.val['Label'] = self.val['Category']
        self.val = self.val.replace({'Label':self.labels})

    def makeMonastrolSplit(self):
        self.data['Category'] = self.data['Image_Metadata_Compound']
        mask1 = self.data['Category'] == 'DMSO'
        mask2 = self.data['Category'] == 'monastrol'
        records = pd.concat( [self.data[mask1], self.data[mask2]] )
        self.splitByReplicate(records)
        print self.train.info()
       
    def makeMoASplit(self, moaGroundTruthFile):
        moa_data = pd.read_csv(moaGroundTruthFile)
        moa_data.columns = ['Image_Metadata_Compound','Image_Metadata_Concentration','MOA']
        self.data = self.data.merge(moa_data)
        self.data['Category'] = self.data['Image_Metadata_Compound']
        self.splitByReplicate(self.data)
        print self.train.info()
