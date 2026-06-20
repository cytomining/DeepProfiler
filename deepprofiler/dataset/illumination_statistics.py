"""Per-plate illumination statistics collection.

:class:`IlluminationStatistics` scans every image in a plate, accumulates a
pixel histogram per channel and a running mean image (at a downscaled
resolution), then computes percentile-based intensity bounds and the
illumination correction function via
:class:`~deepprofiler.dataset.illumination_correction.IlluminationCorrection`.

The resulting stats dict is serialised to a pickle file under
``config["paths"]["intensities"]`` and consumed by
:mod:`deepprofiler.dataset.compression` to perform per-plate illumination
correction and histogram stretching during image compression.

The public entry point is :func:`calculate_statistics`, which is invoked as
a parallel worker by the ``prepare`` CLI command.
"""

import os
import pickle as pickle

import numpy as np
import skimage.transform

import deepprofiler.dataset.image_dataset
import deepprofiler.dataset.utils as utils

from .illumination_correction import IlluminationCorrection


def illum_stats_filename(output_dir, plate_name):
    """Return the canonical path for a plate's illumination statistics pickle.

    Args:
        output_dir: Root intensities directory (``config["paths"]["intensities"]``).
        plate_name: Plate identifier string.

    Returns:
        Path string of the form ``{output_dir}/{plate_name}/{plate_name}.pkl``.
    """
    return "{}/{}/{}.pkl".format(output_dir, plate_name, plate_name)


def percentile(prob, p):
    """Find the intensity bin at which the cumulative probability exceeds ``p``.

    Args:
        prob: 1-D array of per-bin probabilities (should sum to 1).
        p: Probability threshold in ``[0, 1]``.

    Returns:
        Integer bin index (intensity value) at the ``p``-th percentile.
    """
    cum = np.cumsum(prob)
    pos = cum > p
    return np.argmax(pos)


class IlluminationStatistics():
    """Accumulate pixel statistics across all images in one plate.

    Builds a per-channel pixel histogram and a mean image (at downscaled
    resolution) by processing images one at a time via :meth:`processImage`.
    After all images are scanned, :meth:`computeStats` derives intensity
    percentiles, the mean image at full resolution, and the illumination
    correction function.

    Args:
        bits: Bit depth of the images (e.g. 16 for uint16).  Determines the
            histogram length (``2 ** bits`` bins).
        channels: List of channel names — determines the channel count.
        down_scale_factor: Factor by which images are downscaled before
            averaging (e.g. 4 → quarter-resolution mean image).
        median_filter_size: Diameter of the median filter disk used when
            computing the illumination correction function (passed through to
            :class:`~deepprofiler.dataset.illumination_correction.IlluminationCorrection`).
        name: Optional plate name used in log messages.
    """

    def __init__(self, bits, channels, down_scale_factor, median_filter_size, name=""):
        self.depth = 2 ** bits
        self.channels = channels
        self.name = name
        self.down_scale_factor = down_scale_factor
        self.median_filter_size = median_filter_size
        self.hist = np.zeros((len(channels), self.depth), dtype=np.float64)
        self.count = 0
        self.expected = 1
        self.mean_image = None
        self.original_image_size = None

    def processImage(self, index, img, meta):
        """Accumulate statistics for one image.

        Adds the image's per-channel pixel counts to ``self.hist`` and
        accumulates it into ``self.mean_image`` at downscaled resolution.
        Called by :meth:`~deepprofiler.dataset.image_dataset.ImageDataset.scan`.

        Args:
            index: Row index (unused, required by the scan interface).
            img: numpy array ``(H, W, C)`` — the full FOV image.
            meta: Metadata row (unused here, required by the scan interface).
        """
        self.addToMean(img)
        self.count += 1
        utils.logger.info("Plate {} Image {} of {} ({:4.2f}%)".format(
            self.name, self.count, self.expected, 100 * float(self.count) / self.expected
        ))
        for i in range(len(self.channels)):
            counts = np.histogram(img[:, :, i], bins=self.depth, range=(0, self.depth))[0]
            self.hist[i] += counts.astype(np.float64)

    def addToMean(self, img):
        """Downscale ``img`` and add it to the running sum for the mean image.

        All images in a plate are assumed to have the same spatial dimensions.
        The first call initialises ``self.mean_image`` and records
        ``self.original_image_size``; subsequent calls validate that the shape
        matches.

        Args:
            img: numpy array ``(H, W, C)`` at original resolution.

        Raises:
            ValueError: If ``img.shape`` differs from the first image seen.
        """
        if self.original_image_size is None:
            self.original_image_size = img.shape
            self.scale = (img.shape[0] / self.down_scale_factor, img.shape[1] / self.down_scale_factor)
        else:
            if img.shape != self.original_image_size:
                raise ValueError("Images in this plate don't match: required=",
                                 self.original_image_size, " found=", img.shape)
        thumb = skimage.transform.resize(img, self.scale, mode="reflect", anti_aliasing=True, preserve_range=True)
        if self.mean_image is None:
            self.mean_image = np.zeros_like(thumb, dtype=np.float64)
        self.mean_image += thumb

    def computeStats(self):
        """Finalise statistics and compute the illumination correction function.

        Divides ``self.mean_image`` by the image count to get a true mean,
        computes per-channel 0.01th and 99.99th intensity percentiles, then
        calls :class:`~deepprofiler.dataset.illumination_correction.IlluminationCorrection`
        to fit the correction function.

        Returns:
            Dict with keys:

            - ``"mean_values"`` — per-channel mean intensity (float array).
            - ``"upper_percentiles"`` / ``"lower_percentiles"`` — intensity
              bounds at the 99.99th and 0.01th percentiles.
            - ``"histogram"`` — raw per-channel pixel histograms.
            - ``"mean_image"`` — mean image at downscaled resolution.
            - ``"channels"`` — channel name list.
            - ``"original_size"`` — ``(H, W, C)`` of the full-resolution images.
            - ``"illum_correction_function"`` — ``(H, W, C)`` multiplicative
              correction array at full resolution.
        """
        bins = np.linspace(0, self.depth - 1, self.depth)
        mean = np.zeros((len(self.channels)))
        lower = np.zeros((len(self.channels)))
        upper = np.zeros((len(self.channels)))
        if self.mean_image is not None:
            self.mean_image /= self.count

        for i in range(len(self.channels)):
            probs = self.hist[i] / self.hist[i].sum()
            mean[i] = (bins * probs).sum()
            lower[i] = percentile(probs, 0.0001)
            upper[i] = percentile(probs, 0.9999)
        stats = {
            "mean_values": mean,
            "upper_percentiles": upper,
            "lower_percentiles": lower,
            "histogram": self.hist,
            "mean_image": self.mean_image,
            "channels": self.channels,
            "original_size": self.original_image_size,
        }

        correct = IlluminationCorrection(stats, self.channels, self.original_image_size)
        correct.compute_all(self.median_filter_size)
        stats["illum_correction_function"] = correct.illum_corr_func

        utils.logger.info("Plate " + self.name + " done")
        return stats


def calculate_statistics(args):
    """Compute and persist illumination statistics for one plate.

    Parallel worker function called by the ``prepare`` CLI command.  Skips
    plates whose stats file already exists, enabling restartable runs.

    The stats dict returned by
    :meth:`IlluminationStatistics.computeStats` is pickled to
    ``{config["paths"]["intensities"]}/{plate_name}/{plate_name}.pkl``.

    Args:
        args: ``[plate, config]`` where ``plate`` is a
            :class:`~deepprofiler.dataset.metadata.Metadata` slice for one
            plate and ``config`` is the experiment configuration dict.
    """
    plate, config = args
    plateName = plate.data["Metadata_Plate"].iloc[0]

    outfile = illum_stats_filename(config["paths"]["intensities"], plateName)

    if os.path.isfile(outfile):
        print(outfile, "exists")
        return

    keyGen = lambda r: "{}/{}-{}".format(r["Metadata_Plate"], r["Metadata_Well"], r["Metadata_Site"])
    dset = deepprofiler.dataset.image_dataset.ImageDataset(
        plate,
        config["dataset"]["metadata"]["label_field"],
        config["dataset"]["images"]["channels"],
        config["paths"]["images"],
        keyGen,
        config
    )

    hist = IlluminationStatistics(
        config["dataset"]["images"]["bits"],
        config["dataset"]["images"]["channels"],
        config["prepare"]["illumination_correction"]["down_scale_factor"],
        config["prepare"]["illumination_correction"]["median_filter_size"],
        name=plateName
    )
    hist.expected = dset.number_of_records("all")

    dset.scan(hist.processImage, frame="all")

    stats = hist.computeStats()

    utils.check_path(outfile)
    with open(outfile, "wb") as output:
        pickle.dump(stats, output)
