"""Image compression: illumination correction + histogram stretching → 8-bit PNG.

The ``prepare`` command runs this module after
:mod:`deepprofiler.dataset.illumination_statistics` to produce compressed
versions of each plate's images.  Compression is optional but recommended for
large datasets because:

- Raw uint16 TIFFs are large; 8-bit PNGs are 4-8x smaller.
- Histogram stretching at compression time avoids doing it per-crop at
  inference time.
- Compressed images are cached on disk, so repeated profiling runs are faster.

The public entry point is :func:`compress_plate`, called as a parallel worker.
:class:`Compress` handles the per-image logic.
"""

import os.path
import pickle as pickle

import numpy
import skimage.exposure
import skimage.io
import skimage.transform

import deepprofiler.dataset.illumination_statistics
import deepprofiler.dataset.image_dataset
import deepprofiler.dataset.utils


def png_dir(output_dir, plate_name):
    """Return the output directory for a plate's compressed PNG images.

    Args:
        output_dir: Root compressed-images directory
            (``config["paths"]["compressed_images"]``).
        plate_name: Plate identifier string.

    Returns:
        Path string ``{output_dir}/{plate_name}``.
    """
    return os.path.join(output_dir, plate_name)


class Compress():
    """Apply illumination correction and histogram stretching to one plate.

    After construction, call :meth:`process_image` for each image in the
    plate (via :meth:`~deepprofiler.dataset.image_dataset.ImageDataset.scan`).
    Each channel is corrected, downscaled, histogram-stretched to 8 bits, and
    saved as a PNG.

    Args:
        stats: Plate statistics dict as produced by
            :meth:`~deepprofiler.dataset.illumination_statistics.IlluminationStatistics.computeStats`.
            Must contain ``"illum_correction_function"``,
            ``"lower_percentiles"``, ``"upper_percentiles"``,
            ``"histogram"``, and ``"original_size"``.
        channels: List of channel column names (e.g. ``["DNA", "ER", ...]``).
        out_dir: Directory where compressed PNGs will be written.
    """

    def __init__(self, stats, channels, out_dir):
        self.stats = stats
        self.channels = channels
        self.out_dir = out_dir
        self.count = 0
        self.expected = 1
        self.source_format = "tiff"
        self.target_format = "png"
        self.output_shape = [0, 0]
        self.set_scaling_factor(1.0)
        self.metadata_control_filter = lambda x: False
        self.controls_distribution = numpy.zeros((len(channels), 2 ** 8), dtype=numpy.float64)

    def recompute_percentile(self, p, side="upper_percentile"):
        """Recompute a per-channel intensity percentile from the stored histogram.

        Updates ``self.stats[side]`` in place.  Useful for adjusting the
        stretch bounds (e.g. computing the 0.01th and 99.99th percentiles
        used during compression).

        Args:
            p: Probability threshold in ``[0, 1]``.
            side: Key to update in ``self.stats`` — typically
                ``"lower_percentile"`` or ``"upper_percentile"``.
        """
        print("Percentiles for the", side, " >> ", end="")
        self.stats[side] = numpy.zeros((len(self.channels)))
        for i in range(len(self.channels)):
            probs = self.stats["histogram"][i] / self.stats["histogram"][i].sum()
            cum = numpy.cumsum(probs)
            pos = cum > p
            self.stats[side][i] = numpy.argmax(pos)
            print(self.channels[i], ":", self.stats[side][i], " ", end="")
        print("")

    def set_control_samples_filter(self, filterFunc):
        """Register a predicate that identifies control wells for histogram tracking.

        Control-sample pixel distributions are accumulated separately and
        stored in the stats dict via :meth:`getUpdatedStats`, allowing
        downstream analysis to compare treated vs. control intensities.

        Args:
            filterFunc: Callable ``(meta_row) -> bool`` that returns ``True``
                for control samples.
        """
        self.metadata_control_filter = filterFunc
        self.controls_distribution = numpy.zeros((len(self.channels), 2 ** 8), dtype=numpy.float64)

    def set_formats(self, source_format="tiff", target_format="png"):
        """Set source and target image formats.

        Args:
            source_format: Extension of the raw input images (e.g. ``"tiff"``).
            target_format: Must be ``"png"`` — the only supported output format.

        Raises:
            ValueError: If ``target_format`` is not ``"png"``.
        """
        self.source_format = source_format
        self.target_format = target_format
        if target_format != "png":
            raise ValueError("Only PNG compression is supported (target format should be png)")

    def set_scaling_factor(self, factor):
        """Set the spatial downscaling factor applied during compression.

        Args:
            factor: Multiplier applied to the original image dimensions.
                Values < 1 downscale; 1.0 keeps the original size.
        """
        self.output_shape[0] = int(factor * self.stats["original_size"][0])
        self.output_shape[1] = int(factor * self.stats["original_size"][1])

    def target_path(self, origPath):
        """Derive the output PNG path from an input image path.

        Replaces the source format extension with ``"png"`` and prepends
        ``self.out_dir``.  Creates any missing parent directories.

        Args:
            origPath: Original image file path (string).

        Returns:
            Destination path string under ``self.out_dir``.
        """
        image_name = origPath.split("/")[-1]
        image_name = image_name.replace(self.source_format, self.target_format)
        filename = os.path.join(self.out_dir, image_name)
        deepprofiler.dataset.utils.check_path(filename)
        return filename

    def process_image(self, index, img, meta):
        """Compress one multi-channel image and write per-channel PNGs.

        For each channel:

        1. Divide by the illumination correction function.
        2. Downscale to ``self.output_shape``.
        3. Clip at the plate-level 0.05th / 99.95th percentiles (with an
           additional cap at the 99.99th plate percentile to handle saturated
           pixels).
        4. Stretch to ``[0, 255]`` and save as uint8 PNG.

        If ``self.metadata_control_filter(meta)`` is True, also accumulates
        the 8-bit histogram into ``self.controls_distribution``.

        Args:
            index: Row index from the metadata scan (unused).
            img: numpy array ``(H, W, C)`` — the raw full FOV image.
            meta: Metadata row (Pandas Series) for this image.
        """
        self.count += 1
        deepprofiler.dataset.utils.print_progress(self.count, self.expected)
        for c in range(len(self.channels)):
            image = img[:, :, c] / self.stats["illum_correction_function"][:, :, c]
            image = skimage.transform.resize(image, self.output_shape, mode="reflect", anti_aliasing=True)
            pmin, pmax = self.stats["lower_percentiles"][c], self.stats["upper_percentiles"][c]
            vmin, vmax = numpy.percentile(image, (0.05, 99.95))
            vmax = min(vmax, pmax)
            image = skimage.exposure.rescale_intensity(image, in_range=(vmin, vmax))
            image = skimage.img_as_ubyte(image)
            if self.metadata_control_filter(meta):
                self.controls_distribution[c] += numpy.histogram(image, bins=256)[0]
            skimage.io.imsave(self.target_path(meta[self.channels[c]]), image)

    def getUpdatedStats(self):
        """Return the stats dict with the accumulated control distribution appended.

        Returns:
            The ``self.stats`` dict with a ``"controls_distribution"`` key
            added (shape ``(num_channels, 256)``).
        """
        self.stats["controls_distribution"] = self.controls_distribution
        return self.stats


def compress_plate(args):
    """Compress all images in one plate to 8-bit PNG.

    Parallel worker function called by the ``prepare`` CLI command.  Loads
    the plate statistics pickle, constructs a :class:`Compress` object, scans
    all images via :meth:`~deepprofiler.dataset.image_dataset.ImageDataset.scan`,
    and writes the updated stats (with control histograms) back to the pickle.

    Args:
        args: ``[plate, config]`` where ``plate`` is a
            :class:`~deepprofiler.dataset.metadata.Metadata` slice for one
            plate and ``config`` is the experiment configuration dict.
    """
    plate, config = args
    plate_name = plate.data.iloc[0]["Metadata_Plate"]

    statsfile = deepprofiler.dataset.illumination_statistics.illum_stats_filename(
        config["paths"]["intensities"], plate_name
    )
    stats = pickle.load(open(statsfile, "rb"))

    keyGen = lambda r: "{}/{}-{}".format(r["Metadata_Plate"], r["Metadata_Well"], r["Metadata_Site"])
    dset = deepprofiler.dataset.image_dataset.ImageDataset(
        plate,
        config["dataset"]["metadata"]["label_field"],
        config["dataset"]["images"]["channels"],
        config["paths"]["images"],
        keyGen,
        config
    )

    plate_output_dir = png_dir(config["paths"]["compressed_images"], plate_name)
    compress = Compress(stats, config["dataset"]["images"]["channels"], plate_output_dir)
    compress.set_formats(source_format=config["dataset"]["images"]["file_format"], target_format="png")
    compress.set_scaling_factor(config["prepare"]["compression"]["scaling_factor"])
    compress.recompute_percentile(0.0001, side="lower_percentile")
    compress.recompute_percentile(0.9999, side="upper_percentile")
    compress.expected = dset.number_of_records("all")

    filter_func = lambda x: x[config["dataset"]["metadata"]["label_field"]] == config["dataset"]["metadata"]["control_value"]
    compress.set_control_samples_filter(filter_func)

    dset.scan(compress.process_image, frame="all")

    new_stats = compress.getUpdatedStats()
    with open(statsfile, "wb") as output:
        pickle.dump(new_stats, output)
