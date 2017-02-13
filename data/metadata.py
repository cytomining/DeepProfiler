import os
import sys
import pandas as pd

def parseDelimiter(delimiter):
    if delimiter == "blanks":
        return '\s+'
    else:
        return ','

class Metadata():

    def __init__(self, filename=None, csvMode="single", delimiter="default"):
        if filename is not None:
            if csvMode == "single":
                self.loadSingle(filename, delimiter)
            elif csvMode == "multi":
                self.loadMultiple(filename, delimiter)
            print(self.data.info())

    def loadSingle(self, filename, delim):
        print("Reading metadata form", filename)
        delimiter = parseDelimiter(delim)
        self.data = pd.read_csv(filename, delimiter)

    def loadMultiple(self, filename, delim):
        frames = []
        delimiter = parseDelimiter(delim)
        with open(filename, "r") as filelist:
            for line in filelist:
                csvPath = line.replace("\n","")
                print("Reading from", csvPath)
                frames.append( pd.read_csv(csvPath, delimiter) )
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

