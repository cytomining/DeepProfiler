import os
import sys
import data.utils as utils
import skimage.transform
import scipy.misc
import numpy as np

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
        self.metadataControlFilter = lambda x:False
        self.controls_distribution = np.zeros((len(channels), 2**8), dtype=np.float64)

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

    # Filter images that belong to control samples, to compute their histogram distribution
    def setControlSamplesFilter(self, filterFunc):
        self.metadataControlFilter = filterFunc
        self.controls_distribution = np.zeros((len(self.channels), 2**8), dtype=np.float64)

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
        image_name = origPath.split("/")[-1]
        filename = self.outDir + image_name.replace(self.sourceFormat,self.targetFormat)
        utils.check_path(filename)
        return filename

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
            image = scipy.misc.toimage(image, low=0, high=255, mode="L",
                        cmin=self.stats["lower_percentiles"][c], cmax=self.stats["upper_percentiles"][c])
            if self.metadataControlFilter(meta):
                self.controls_distribution[c] += image.histogram()
            image.save(self.targetPath(meta[self.channels[c]]))
        return

    def getUpdatedStats(self):
        self.stats["controls_distribution"] = self.controls_distribution
        return self.stats

