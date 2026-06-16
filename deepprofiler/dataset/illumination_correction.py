"""Per-channel illumination correction function computation.

Implements the correction method from Singh et al. (2014): the mean image
across all sites in a plate is smoothed with a median filter to estimate
the illumination gradient, then divided by its robust minimum to produce a
multiplicative correction function.  Each channel is corrected independently.

The correction function is computed by
:class:`~deepprofiler.dataset.illumination_statistics.IlluminationStatistics`
and stored in the plate statistics pickle file.  It is applied per-channel
during compression in :mod:`deepprofiler.dataset.compression`.
"""

import numpy as np
import skimage.filters
import skimage.morphology
import skimage.transform


class IlluminationCorrection(object):
    """Compute a per-channel multiplicative illumination correction function.

    Fits a smooth spatial illumination model from the plate mean image and
    returns a correction array that can divide raw pixel values to flatten
    uneven illumination across the field of view.

    Args:
        stats: Dictionary of plate statistics as returned by
            :meth:`~deepprofiler.dataset.illumination_statistics.IlluminationStatistics.computeStats`.
            Must contain ``"mean_image"`` and ``"original_size"``.
        channels: List of channel names, used to determine the channel count.
        target_dim: ``(height, width)`` of the full-resolution images.
            The correction function is resized to this shape after smoothing.
    """

    def __init__(self, stats, channels, target_dim):
        """Initialise correction arrays from plate statistics."""
        self.stats = stats
        self.channels = channels
        self.target_dim = (target_dim[0], target_dim[1])
        self.illum_corr_func = np.zeros((self.target_dim[0], self.target_dim[1], len(self.channels)))

    def channel_function(self, mean_channel, disk_size):
        """Compute the correction function for one channel.

        Applies a median filter with a disk of radius ``disk_size`` to the
        downscaled mean image, resizes the result back to ``target_dim``,
        clips at the 2nd percentile to avoid division by near-zero values,
        and normalises by that clipped minimum so the correction function is
        1.0 in the brightest region and > 1.0 where the image is dim.

        Args:
            mean_channel: 2-D array — the mean image for this channel at
                downscaled resolution (as accumulated by
                :class:`~deepprofiler.dataset.illumination_statistics.IlluminationStatistics`).
            disk_size: Radius of the median filter disk in pixels.

        Returns:
            2-D float array of shape ``target_dim`` — divide raw pixel values
            by this to correct illumination.
        """
        operator = skimage.morphology.disk(disk_size)
        filtered_channel = skimage.filters.median(mean_channel.astype(np.uint16), footprint=operator)
        filtered_channel = skimage.transform.resize(
            filtered_channel, self.target_dim, mode="reflect", anti_aliasing=True, preserve_range=True
        )
        robust_minimum = np.percentile(filtered_channel, 2)
        filtered_channel = np.maximum(filtered_channel, robust_minimum)
        illum_corr_func = filtered_channel / robust_minimum
        return illum_corr_func

    def compute_all(self, median_filter_size):
        """Compute and store correction functions for all channels.

        Populates ``self.illum_corr_func`` with shape
        ``(height, width, num_channels)``.  Called once per plate by
        :meth:`~deepprofiler.dataset.illumination_statistics.IlluminationStatistics.computeStats`.

        Args:
            median_filter_size: Diameter (not radius) of the median filter
                disk.  Converted to radius internally.
        """
        disk_size = median_filter_size / 2
        for ch in range(len(self.channels)):
            self.illum_corr_func[:, :, ch] = self.channel_function(
                self.stats["mean_image"][:, :, ch], disk_size
            )

    def apply(self, image):
        """Divide a raw image by the correction function.

        Args:
            image: numpy array of shape ``(H, W, C)`` matching ``target_dim``
                and channel count.

        Returns:
            Corrected image array of the same shape.
        """
        return image / self.illum_corr_func
