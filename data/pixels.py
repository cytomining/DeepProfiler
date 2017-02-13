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

# Main image reading function. Images are treated as 3D tensors: (height, width, channels)
def openImage(paths, pixelProcessor):
    channels = [ skimage.io.imread(p) for p in paths ]
    img = np.zeros( (channels[0].shape[0], channels[0].shape[1], len(channels)) )
    for c in range(len(channels)):
        img[:,:,c] = channels[c]
    return pixelProcessor.run(img)

#################################################
## PIXEL PROCESSING CLASSES
#################################################

# Abstract class to extend operations that can be applied to images while reading them
class PixelProcessor():

    def process(self, pixels):
        return pixels

    def run(self, pixels):
        return self.process(pixels)

#################################################
## COMPUTATION OF ILLUMINATION STATISTICS
#################################################

# Build pixel histogram for each channel
class ImageStatistics():

    def __init__(self, bits, channels, name=""):
        self.depth = 2**bits
        #TODO: Change channels from number to the array with channel names
        self.channels = channels
        self.name = name
        self.hist = np.zeros((channels, self.depth), dtype=np.float64)
        self.count = 0
        self.expected = 1
        self.meanImage = None
        
    def processImage(self, index, img, meta):
        self.addToMean(img)
        self.count += 1
        utils.logger.info("Plate {} Image {} of {} ({:4.2f}%)".format(self.name, 
                          self.count, self.expected, 100*float(self.count)/self.expected))
        for i in range(self.channels):
            counts = np.histogram(img[:,:,i], bins=self.depth, range=(0,self.depth))[0]
            self.hist[i] += counts.astype(np.float64)

    # Accumulate the mean image. Useful for illumination correction purposes
    def addToMean(self, img):
        # TODO: parameterize the target size
        scale = 540
        thumb = skimage.transform.resize(img, (scale,scale))
        if self.meanImage is None:
            self.meanImage = np.zeros_like(thumb, dtype=np.float64)
        self.meanImage += thumb.astype(np.float64)
        return

    def percentile(self, prob, p):
        cum = np.cumsum(prob)
        pos = cum > p
        return np.argmax(pos)

    # Compute global statistics on pixels. Useful for contrast/brightness adjustment (a.k.a histogram stretching)
    def computeStats(self):
        bins = np.linspace(0,self.depth-1,self.depth)
        mean = np.zeros((self.channels))
        lower = np.zeros((self.channels))
        upper = np.zeros((self.channels))
        self.meanImage /= self.count
        for i in range(self.channels):
            probs = self.hist[i]/self.hist[i].sum()
            mean[i] = (bins * probs).sum()
            lower[i] = self.percentile(probs, 0.0001)
            upper[i] = self.percentile(probs, 0.9999)
        # TODO: add channel names here, and also the original image size
        stats = {'mean':mean, 'upper':upper, 'lower':lower, 'hist':self.hist, "MeanImg":self.meanImage}
        utils.logger.info('Plate ' + self.name + ' done')
        return stats

#################################################
## ILLUMINATION CORRECTION FUNCTION
#################################################

class IlluminationCorrection():

    ROBUST_FACTOR = .02  # For rescaling, take 2nd percentile value

    def __init__(self, stats, channels, targetDim):
        self.stats = stats
        self.channels = channels
        self.targetDim = targetDim

    # Based on the CellProfiler implementation of Illumination Correction
    # CellProfiler/cellprofiler/modules/correctilluminationcalculate.py
    def computeFunction(self, meanChannel, diskSize=25):
        #TODO: get np.type from other source or parameterize or compute :/
        # We currently assume 16 bit images
        filteredChannel = median(meanChannel.astype(np.uint16), disk(diskSize))
        filteredChannel = skimage.transform.resize(filteredChannel, self.targetDim)
        sortedPixels = filteredChannel[filteredChannel > 0]
        sortedPixels.sort()
        idx = int(sortedPixels.shape[0] * ROBUST_FACTOR)
        robustMinimum = sortedPixels[idx]
        filteredChannel[filteredChannel < robustMinimum] = robustMinimum
        illumCorrFunc = filteredChannel / robustMinimum
        return illumCorrFunc

    def computeAll(self, diskSize=25):
        illumCorrFunc = np.zeros( (self.targetDim[0], self.targetDim[1], len(self.channels)) )
        for ch in range(len(self.channels)):
            illumCorrFunc[:,:,ch] = self.computeFunction(self.stats["MeanImg"][:,:,ch])
        self.illumCorrFunc = illumCorrFunc
        return

    def apply(self, image):
        return image / self.illumCorrFunc


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

    # Allows to recalculate the percentiles computed by default in the ImageStatistics class
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

    # Main method. Downscales, stretches histogram, and saves as PNG
    def processImage(self, index, img, meta):
        self.count += 1
        utils.printProgress(self.count, self.expected)
        for c in range(len(self.channels)):
            # Downscale (TODO: Parameterize the resizing factor)
            image = skimage.transform.downscale_local_mean(img[:,:,c], factors=(2,2))
            # Stretch illumination values
            image[image < self.stats["lower"][c]] = self.stats["lower"][c]
            image[image > self.stats["upper"][c]] = self.stats["upper"][c]
            # TODO: add illumination correction here
            # Save as PNG (TODO: save using ImageMagick)
            scipy.misc.imsave(self.targetPath(meta[self.channels[c]]), image)
        return
            

