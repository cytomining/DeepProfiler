import skimage.transform
import skimage.filters
import skimage.morphology
import numpy as np

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

