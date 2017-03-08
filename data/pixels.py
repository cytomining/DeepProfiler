import os
import sys
import argparse
import data.utils as utils
import data.metadata as meta
import skimage
import skimage.io
import skimage.transform
import skimage.exposure
import skimage.filters
import skimage.morphology
import scipy.misc
import numpy as np
import pickle as pickle

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

    def __init__(self, bits, channels, downScaleFactor, medianFilterSize, name=""):
        self.depth = 2**bits
        self.channels = channels
        self.name = name
        self.downScaleFactor = downScaleFactor
        self.medianFilterSize = medianFilterSize
        self.hist = np.zeros((len(channels), self.depth), dtype=np.float64)
        self.count = 0
        self.expected = 1
        self.meanImage = None
        self.originalImageSize = None
        
    def processImage(self, index, img, meta):
        self.addToMean(img)
        self.count += 1
        utils.logger.info("Plate {} Image {} of {} ({:4.2f}%)".format(self.name, 
                          self.count, self.expected, 100*float(self.count)/self.expected))
        for i in range(len(self.channels)):
            counts = np.histogram(img[:,:,i], bins=self.depth, range=(0,self.depth))[0]
            self.hist[i] += counts.astype(np.float64)

    # Accumulate the mean image. Useful for illumination correction purposes
    def addToMean(self, img):
        # Check image size (we assume all images have the same size)
        if self.originalImageSize is None:
            self.originalImageSize = img.shape
            self.scale = (img.shape[0]/self.downScaleFactor, img.shape[1]/self.downScaleFactor)
        else:
            if img.shape != self.originalImageSize:
                raise ValueError("Images in this plate don't match: required=",
                                 self.originalImageSize, " found=", img.shape)
        # Rescale original image to half
        thumb = skimage.transform.resize(img, self.scale)
        if self.meanImage is None:
            self.meanImage = np.zeros_like(thumb, dtype=np.float64)
        # Add image to current mean values
        self.meanImage += thumb.astype(np.float64)
        return

    def percentile(self, prob, p):
        cum = np.cumsum(prob)
        pos = cum > p
        return np.argmax(pos)

    # Compute global statistics on pixels. Useful for contrast/brightness adjustment (a.k.a histogram stretching)
    def computeStats(self):
        # Initialize counters
        bins = np.linspace(0,self.depth-1,self.depth)
        mean = np.zeros((len(self.channels)))
        lower = np.zeros((len(self.channels)))
        upper = np.zeros((len(self.channels)))
        self.meanImage /= self.count

        # Compute percentiles and histogram
        for i in range(len(self.channels)):
            probs = self.hist[i]/self.hist[i].sum()
            mean[i] = (bins * probs).sum()
            lower[i] = self.percentile(probs, 0.0001)
            upper[i] = self.percentile(probs, 0.9999)
        stats = {"mean_values":mean, "upper_percentiles":upper, "lower_percentiles":lower, "histogram":self.hist, 
                 "mean_image":self.meanImage, "channels":self.channels, "original_size":self.originalImageSize}

        # Compute illumination correction function and add it to the dictionary
        correct = IlluminationCorrection(stats, self.channels, self.originalImageSize)
        correct.computeAll(self.medianFilterSize)
        stats["illum_correction_function"] = correct.illumCorrFunc

        # Plate ready
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
        self.targetDim = (targetDim[0], targetDim[1])

    # Based on the CellProfiler implementation of Illumination Correction
    # CellProfiler/cellprofiler/modules/correctilluminationcalculate.py
    def channelFunction(self, meanChannel, diskSize):
        #TODO: get np.type from other source or parameterize or compute :/
        # We currently assume 16 bit images
        operator = skimage.morphology.disk(diskSize)
        filteredChannel = skimage.filters.median(meanChannel.astype(np.uint16), operator)
        filteredChannel = skimage.transform.resize(filteredChannel, self.targetDim)
        sortedPixels = filteredChannel[filteredChannel > 0]
        sortedPixels.sort()
        idx = int(sortedPixels.shape[0] * self.ROBUST_FACTOR)
        robustMinimum = sortedPixels[idx]
        filteredChannel[filteredChannel < robustMinimum] = robustMinimum
        illumCorrFunc = filteredChannel / robustMinimum
        return illumCorrFunc

    def computeAll(self, medianFilterSize):
        diskSize = medianFilterSize/2 # From diameter to radius
        illumCorrFunc = np.zeros( (self.targetDim[0], self.targetDim[1], len(self.channels)) )
        for ch in range(len(self.channels)):
            illumCorrFunc[:,:,ch] = self.channelFunction(self.stats["mean_image"][:,:,ch], diskSize)
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
        self.setScalingFactor(1.0)

    # Allows to recalculate the percentiles computed by default in the ImageStatistics class
    def recomputePercentile(self, p, side="upper"):
        print("Percentiles for the", side, " >> ", end='')
        self.stats[side] = np.zeros((len(self.channels)))
        for i in range(len(self.channels)):
            probs = self.stats["histogram"][i]/self.stats["histogram"][i].sum()
            cum = np.cumsum(probs)
            pos = cum > p
            self.stats[side][i] = np.argmax(pos)
            print(self.channels[i], ':', self.stats[side][i], ' ', end='')
        print('')

    # If the sourceFormat is the same as the target, no compression should be applied.
    def setFormats(self, sourceFormat="tiff", targetFormat="png"):
        self.sourceFormat = sourceFormat
        self.targetFormat = targetFormat
        if targetFormat != "png":
            raise ValueError("Only PNG compression is supported (target format should be png)")

    # Takes a percent factor to rescale the image preserving aspect ratio
    # If the number is between 0 and 1, the image is downscaled, otherwise is upscaled
    def setScalingFactor(self, factor):
        self.outputShape = [0,0]
        self.outputShape[0] = int(factor * self.stats["original_size"][0])
        self.outputShape[1] = int(factor * self.stats["original_size"][1])

    def targetPath(self, origPath):
        basePath = "/".join( origPath.split("/")[0:-1] )
        os.system("mkdir -p " + self.outDir + basePath)
        return self.outDir + origPath.replace(self.sourceFormat,self.targetFormat)

    # Main method. Downscales, stretches histogram, and saves as PNG
    def processImage(self, index, img, meta):
        self.count += 1
        utils.printProgress(self.count, self.expected)
        for c in range(len(self.channels)):
            # Illumination correction
            image = img[:,:,c] / self.stats["illum_correction_function"][:,:,c]
            # Downscale
            image = skimage.transform.resize(image, self.outputShape) 
            # Clip illumination values
            image[image < self.stats["lower_percentiles"][c]] = self.stats["lower_percentiles"][c]
            image[image > self.stats["upper_percentiles"][c]] = self.stats["upper_percentiles"][c]
            # Save resulting image in 8-bits PNG format
            #scipy.misc.imsave(self.targetPath(meta[self.channels[c]]), image)
            image = scipy.misc.toimage(image, low=0, high=255, mode="L",
                        cmin=self.stats["lower_percentiles"][c], cmax=self.stats["upper_percentiles"][c])
            image.save(self.targetPath(meta[self.channels[c]]))
        return
            

