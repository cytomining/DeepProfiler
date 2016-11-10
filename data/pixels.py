import os
import sys
import numpy as np
import libtiff
import argparse
import utils
import metadata as meta
import dataset as ds
import cPickle as pickle

#################################################
## COMMON IMAGE HANDLING OPPERATIONS
#################################################

def openTIFF(path):
    tif = libtiff.TIFF.open(path)
    img = tif.read_image()
    data = np.asarray(img, dtype=np.float32)
    return data

def openImage(paths, pixelProcessor):
    dna = openTIFF(paths['DNA'])
    tub = openTIFF(paths['Tubulin'])
    act = openTIFF(paths['Actin'])
    assert dna.shape[0] == tub.shape[0] == act.shape[0]
    assert dna.shape[1] == tub.shape[1] == act.shape[1]
    img = np.zeros((dna.shape[0],dna.shape[1],3))
    img[:,:,0] = tub
    img[:,:,1] = act
    img[:,:,2] = dna
    return pixelProcessor.run(img)

#################################################
## PIXEL PROCESSING CLASSES
#################################################

class PixelProcessor():

    def process(self, pixels):
        return pixels

    def run(self, pixels):
        return self.process(pixels)

class PixelScaling(PixelProcessor):

    def __init__(self, interval='0,1', size=2.0**16):
        self.interval = interval
        self.size = size

    def process(self, pixels):
        if self.interval == '-1,1':
            pixels = (pixels - self.size/2)/(self.size/2)
        elif self.interval == '0,1':
            pixels = pixels/self.size
        elif self.interval == 'autolinear':
            for i in range(0,pixels.shape[2]):
                im = pixels[:,:,i]
                (minimum, maximum) = (im[im > 0].min(), im[np.isfinite(im)].max())
                pixels[:,:,i] = (im - minimum) / (maximum - minimum)
        elif self.interval == 'autolog':
            for i in range(0,pixels.shape[2]):
                im = pixels[:,:,i]
                (minimum, maximum) = (im[im > 0].min(), im[np.isfinite(im)].max())
                pixels[:,:,i] = (np.log(im.clip(minimum, maximum)) - np.log(minimum)) / (np.log(maximum) - np.log(minimum))
        return pixels

class PixelMeanCentering(PixelProcessor):

    def __init__(self, meanFile):
        data = pickle.load(open(meanFile,'r'))
        self.mean = data['mean']

    def process(self, pixels):
        return pixels - self.mean

class PixelZNorm(PixelProcessor):

    def __init__(self, meanFile):
        data = pickle.load(open(meanFile,'r'))
        self.mean = data['mean']
        self.std = data['std']

    def process(self, pixels):
        return (pixels - self.mean)/self.std


class PixelLinearNorm(PixelProcessor):

    def __init__(self, meanFile):
        data = pickle.load(open(meanFile,'r'))
        self.min = data['min']
        self.max = data['max']
        self.diff = self.max - self.min

    def process(self, pixels):
        pixels = (pixels - self.min)/self.diff
        return pixels.clip(0,1)

class PixelLogNorm(PixelProcessor):

    def __init__(self, meanFile):
        data = pickle.load(open(meanFile,'r'))
        self.minLog = np.log(data['min'])
        self.maxLog = np.log(data['max'])
        self.diffLog = self.maxLog - self.minLog
        self.min = data['min']
        self.max = data['max']

    def process(self, pixels):
        for i in range(pixels.shape[2]):
            pixels[:,:,i] = (np.log(pixels[:,:,i].clip(self.min[i], self.max[i])) - self.minLog[i]) / self.diffLog[i]
        return pixels


#################################################
## COMPUTATION OF MEAN AND STD IN A DATASET
#################################################

# Build pixel histogram for each channel
class Histogram():
    def __init__(self, bits, channels):
        self.depth = 2**bits
        self.channels = channels
        self.hist = np.zeros((channels, self.depth), dtype=np.float64)
        self.mins = 2**bits * np.ones((channels))
        self.maxs = np.zeros((channels))
        self.count = 0
        self.expected = 1
        
    def processImage(self, img):
        self.count += 1
        utils.printProgress(self.count, self.expected)
        for i in range(self.channels):
            counts = np.histogram(img[:,:,i], bins=self.depth, range=(0,self.depth))[0]
            self.hist[i] += counts.astype(np.float64)
            minval = np.min(img[:,:,i][ img[:,:,i]>0 ]) 
            maxval = np.max(img[:,:,i])
            if minval < self.mins[i]: self.mins[i] = minval
            if maxval > self.maxs[i]: self.maxs[i] = maxval

    def computeStats(self):
        bins = np.linspace(0,self.depth-1,self.depth)
        mean = np.zeros((self.channels))
        std = np.zeros((self.channels))
        for i in range(self.channels):
            probs = self.hist[i]/self.hist[i].sum()
            mean[i] = (bins * probs).sum()
            sigma2 = ((bins - mean[i])**2)*(probs)
            std[i] = np.sqrt( sigma2.sum() )
            print 'Mean',mean[i],'Std',std[i],'Min',self.mins[i],'Max',self.maxs[i]
        stats = {'mean':mean, 'std':std, 'min':self.mins, 'max':self.maxs}
        return stats

def computeMeanAndStd(metadataFilename, datasetDir, outputFile):
    metadata = meta.ImageMetadata(metadataFilename)
    metadata.makeMonastrolSplit()
    dataset = ds.Dataset(metadata, dataRoot=datasetDir)
    hist = Histogram(16, 3)
    hist.expected = dataset.numberOfRecords('val')
    dataset.scan(hist.processImage, frame='val')
    stats = hist.computeStats()
    with open(outputFile, 'wb') as output:
        pickle.dump(stats, output, protocol=pickle.HIGHEST_PROTOCOL)
    return hist

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata", help="Filename of the CSV metadata file")
    parser.add_argument("datadir", help="Directory containing images listed in the CSV")
    parser.add_argument("outfile", help="Filename of the resulting pickle file with the mean an std")
    args = parser.parse_args()
    computeMeanAndStd(args.metadata, args.datadir, args.outfile)

