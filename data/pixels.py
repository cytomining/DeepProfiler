import os
import sys
import argparse
import utils
import skimage
import skimage.io
import skimage.transform
import skimage.exposure
import scipy.misc
import numpy as np
import metadata as meta
import dataset as ds
import cPickle as pickle

#################################################
## COMMON IMAGE HANDLING OPPERATIONS
#################################################

def openImage(paths, pixelProcessor):
    channels = [ skimage.io.imread(p) for p in paths ]
    img = np.zeros( (channels[0].shape[0], channels[0].shape[1], len(channels)) )
    for c in range(len(channels)):
        img[:,:,c] = channels[c]
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
class ImageStatistics():
    def __init__(self, bits, channels, name=""):
        self.depth = 2**bits
        self.channels = channels
        self.name = name
        self.hist = np.zeros((channels, self.depth), dtype=np.float64)
        self.mins = 2**bits * np.ones((channels))
        self.maxs = np.zeros((channels))
        self.count = 0
        self.expected = 1
        self.meanImage = None
        
    def processImage(self, index, img, meta):
        self.addToMean(img)
        utils.logger.info("{} Image {} of {} ({:4.2f}%)".format(self.name, 
                          self.count, self.expected, 100*float(self.count)/self.expected))
        for i in range(self.channels):
            counts = np.histogram(img[:,:,i], bins=self.depth, range=(0,self.depth))[0]
            self.hist[i] += counts.astype(np.float64)
            minval = np.min(img[:,:,i][ img[:,:,i]>0 ]) 
            maxval = np.max(img[:,:,i])
            if minval < self.mins[i]: self.mins[i] = minval
            if maxval > self.maxs[i]: self.maxs[i] = maxval

    def addToMean(self, img):
        thumb = skimage.transform.resize(img, (540,540)) 
        if self.meanImage is None:
            self.meanImage = np.zeros_like(thumb, dtype=np.float64)
        self.count += 1
        self.meanImage += thumb
        return

    def percentile(self, prob, p):
        cum = np.cumsum(prob)
        pos = cum > p
        return np.argmax(pos)

    def computeStats(self):
        bins = np.linspace(0,self.depth-1,self.depth)
        mean = np.zeros((self.channels))
        std = np.zeros((self.channels))
        lower = np.zeros((self.channels))
        upper = np.zeros((self.channels))
        self.meanImage /= self.count
        print "Plate:",self.name
        for i in range(self.channels):
            probs = self.hist[i]/self.hist[i].sum()
            mean[i] = (bins * probs).sum()
            sigma2 = ((bins - mean[i])**2)*(probs)
            std[i] = np.sqrt( sigma2.sum() )
            lower[i] = self.percentile(probs, 0.0001)
            upper[i] = self.percentile(probs, 0.9999)
            msg = 'Mean {:8.2f} Std {:8.2f} Min {:5.0f} Max {:5.0f} Upper {:5.0f} Lower {:5.0f}'
            print msg.format(mean[i],std[i],self.mins[i],self.maxs[i],upper[i],lower[i])
        stats = {'mean':mean, 'std':std, 'min':self.mins, 'max':self.maxs, 'upper':upper, 'lower':lower, 
                 'hist':self.hist, "MeanImg":self.meanImage}
        return stats

#################################################
## COMPRESSION OF TIFF IMAGES INTO PNGs
#################################################

class Compress():
    def __init__(self, stats, channels, outDir):
        self.stats = stats
        self.channels = channels
        self.outDir = outDir
        self.count = 0
        self.expected = 1
        self.setFormats()

    def recomputePercentile(self, p, side="upper"):
        print "Percentiles for the",side," >> ",
        self.stats[side] = np.zeros((len(self.channels)))
        for i in range(len(self.channels)):
            probs = self.stats["hist"][i]/self.stats["hist"][i].sum()
            cum = np.cumsum(probs)
            pos = cum > p
            self.stats[side][i] = np.argmax(pos)
            print self.channels[i],':',self.stats[side][i],
        print ''

    def setFormats(self, sourceFormat="tiff", targetFormat="png"):
        self.sourceFormat = sourceFormat
        self.targetFormat = targetFormat

    def targetPath(self, origPath):
        basePath = "/".join( origPath.split("/")[0:-1] )
        os.system("mkdir -p " + self.outDir + basePath)
        return self.outDir + origPath.replace(self.sourceFormat,self.targetFormat)

    def processImage(self, index, img, meta):
        self.count += 1
        utils.printProgress(self.count, self.expected)
        for c in range(len(self.channels)):
            image = skimage.transform.downscale_local_mean(img[:,:,c], factors=(2,2))
            image[image < self.stats["lower"][c]] = self.stats["lower"][c]
            image[image > self.stats["upper"][c]] = self.stats["upper"][c]
            scipy.misc.imsave(self.targetPath(meta[self.channels[c]]), image)
        return
            

