import skimage.transform
import skimage.filters
import skimage.morphology
import numpy as np

#################################################
## ILLUMINATION CORRECTION FUNCTION
#################################################

class IlluminationCorrection():

    ROBUST_FACTOR = .02  # For rescaling, take 2nd percentile value

    def __init__(self, stats, channels, target_dim):
        self.stats = stats
        self.channels = channels
        self.target_dim = (target_dim[0], target_dim[1])


    # Based on the CellProfiler implementation of Illumination Correction
    # CellProfiler/cellprofiler/modules/correctilluminationcalculate.py
    def channel_function(self, mean_channel, disk_size):
        #TODO: get np.type from other source or parameterize or compute :/
        # We currently assume 16 bit images
        operator = skimage.morphology.disk(disk_size)
        print(mean_channel.min(), mean_channel.mean(), mean_channel.max())
        filtered_channel = skimage.filters.median(mean_channel.astype(np.uint16), operator)
        filtered_channel = skimage.transform.resize(filtered_channel, self.target_dim)
        print(filtered_channel.mean())
        sorted_pixels = filtered_channel[filtered_channel > 0]
        print(sorted_pixels.shape)
        sorted_pixels.sort()
        idx = int(sorted_pixels.shape[0] * self.ROBUST_FACTOR)
        robust_minimum = sorted_pixels[idx]
        filtered_channel[filtered_channel < robust_minimum] = robust_minimum
        illum_corr_func = filtered_channel / robust_minimum
        return illum_corr_func


    def compute_all(self, median_filter_size):
        disk_size = median_filter_size / 2 # From diameter to radius
        illum_corr_func = np.zeros((self.target_dim[0], self.target_dim[1], len(self.channels)))
        for ch in range(len(self.channels)):
            illum_corr_func[:,:,ch] = self.channel_function(self.stats["mean_image"][:, :, ch], disk_size)
        self.illum_corr_func = illum_corr_func


    def apply(self, image):
        return image / self.illum_corr_func

