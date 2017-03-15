import os
import sys
import pandas as pd
import data.utils as utils

def parseDelimiter(delimiter):
    if delimiter == "blanks":
        return '\s+'
    elif delimiter == "tabs":
        return '\t'
    else:
        return ','

## Generator of plates. Reads metadata and yields plates
def readPlates(metaFile):
    metadata = Metadata(metaFile)
    plates = metadata.data["Metadata_Plate"].unique()
    utils.logger.info("Total plates: " + str(len(plates)))
    #plate = metadata.filterRecords(lambda df: (df.Metadata_Plate == plates[0]) & (df.Metadata_Well == "a01"), copy=True)
    for i in range(2): #len(plates)):
        plate = metadata.filterRecords(lambda df: (df.Metadata_Plate == plates[i]) & (df.Metadata_Well == "a01"), copy=True)
        yield plate
    return

class Metadata():

    # The dtype argument indicates whether the data should be read as strings (object) 
    # or according to the data type (None)
    def __init__(self, filename=None, csvMode="single", delimiter="default", dtype=object):
        if filename is not None:
            if csvMode == "single":
                self.loadSingle(filename, delimiter, dtype)
            elif csvMode == "multi":
                self.loadMultiple(filename, delimiter, dtype)
            print(self.data.info())

    def loadSingle(self, filename, delim, dtype):
        print("Reading metadata form", filename)
        delimiter = parseDelimiter(delim)
        # Read csv files as strings without dropping NA symbols
        self.data = pd.read_csv(filename, delimiter, dtype=dtype, keep_default_na=False)

    def loadMultiple(self, filename, delim, dtype):
        frames = []
        delimiter = parseDelimiter(delim)
        with open(filename, "r") as filelist:
            for line in filelist:
                csvPath = line.replace("\n","")
                print("Reading from", csvPath)
                frames.append( pd.read_csv(csvPath, delimiter, dtype=dtype, keep_default_na=False) )
        self.data = pd.concat(frames)
        print("Multiple CSV files loaded")

    def filterRecords(self, filteringRule, copy=False):
        if copy:
            newMeta = Metadata()
            newMeta.data = self.data.loc[filteringRule(self.data), :].copy()
            return newMeta
        else:
            self.data = self.data.loc[filteringRule(self.data), :]

    def splitMetadata(self, trainingRule, validationRule):
        self.train = self.data[trainingRule(self.data)].copy()
        self.val = self.data[validationRule(self.data)].copy()

