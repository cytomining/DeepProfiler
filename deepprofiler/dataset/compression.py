import pickle as pickle

import numpy
import scipy.stats
import skimage.transform
import os.path
import skimage
import skimage.io
import skimage.exposure

import deepprofiler.dataset.utils
import deepprofiler.dataset.illumination_statistics
import deepprofiler.dataset.image_dataset

def png_dir(output_dir, plate_name):
    return os.path.join(output_dir, plate_name)

#################################################
## COMPRESSION OF TIFF IMAGES INTO PNGs
#################################################

class Compress():
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
        self.metadata_control_filter = lambda x:False
        self.controls_distribution = numpy.zeros((len(channels), 2 ** 8), dtype=numpy.float64)

    # Allows to recalculate the percentiles computed by default in the ImageStatistics class
    def recompute_percentile(self, p, side="upper_percentile"):
        print("Percentiles for the", side, " >> ", end="")
        self.stats[side] = numpy.zeros((len(self.channels)))
        for i in range(len(self.channels)):
            probs = self.stats["histogram"][i]/self.stats["histogram"][i].sum()
            cum = numpy.cumsum(probs)
            pos = cum > p
            self.stats[side][i] = numpy.argmax(pos)
            print(self.channels[i], ":", self.stats[side][i], " ", end="")
        print("")

    # Filter images that belong to control samples, to compute their histogram distribution
    def set_control_samples_filter(self, filterFunc):
        self.metadata_control_filter = filterFunc
        self.controls_distribution = numpy.zeros((len(self.channels), 2 ** 8), dtype=numpy.float64)

    # If the sourceFormat is the same as the target, no compression should be applied.
    def set_formats(self, source_format="tiff", target_format="png"):
        self.source_format = source_format
        self.target_format = target_format
        if target_format != "png":
            raise ValueError("Only PNG compression is supported (target format should be png)")

    # Takes a percent factor to rescale the image preserving aspect ratio
    # If the number is between 0 and 1, the image is downscaled, otherwise is upscaled
    def set_scaling_factor(self, factor):
        self.output_shape[0] = int(factor * self.stats["original_size"][0])
        self.output_shape[1] = int(factor * self.stats["original_size"][1])

    def target_path(self, origPath):
        image_name = origPath.split("/")[-1]
        image_name = image_name.replace(self.source_format, self.target_format)
        filename = os.path.join(self.out_dir, image_name)
        deepprofiler.dataset.utils.check_path(filename)
        return filename

    # Main method. Downscales, stretches histogram, and saves as PNG
    def process_image(self, index, img, meta):
        self.count += 1
        deepprofiler.dataset.utils.print_progress(self.count, self.expected)
        for c in range(len(self.channels)):
            # Illumination correction
            image = img[:, :, c] / self.stats["illum_correction_function"][:, :, c]

            # Downscale
            image = skimage.transform.resize(image, self.output_shape, mode="reflect", anti_aliasing=True)

            # Clip illumination values (remove 0.1% of the illumination distribution)
            # Compare the 99.95 percentile of the image with the 99.99 percentile of the plate
            # Keep the smallest to compensate for saturated pixels before compression
            pmin, pmax = self.stats["lower_percentiles"][c], self.stats["upper_percentiles"][c]
            vmin, vmax = scipy.stats.scoreatpercentile(image, (0.05, 99.95))
            vmax = min(vmax, pmax)
            image = skimage.exposure.rescale_intensity(image, in_range=(vmin, vmax))

            # Save resulting image in 8-bits PNG format
            image = skimage.img_as_ubyte(image)
            if self.metadata_control_filter(meta):
                self.controls_distribution[c] += numpy.histogram(image, bins=256)[0]
            skimage.io.imsave(self.target_path(meta[self.channels[c]]), image)
        return

    def getUpdatedStats(self):
        self.stats["controls_distribution"] = self.controls_distribution
        return self.stats

#################################################
## COMPRESS IMAGES IN A PLATE
#################################################

def compress_plate(args):
    # Load parameters
    plate, config = args
    plate_name = plate.data.iloc[0]["Metadata_Plate"]

    # Dataset configuration
    statsfile = deepprofiler.dataset.illumination_statistics.illum_stats_filename(config["paths"]["intensities"], plate_name)
    stats = pickle.load( open(statsfile, "rb") )
    keyGen = lambda r: "{}/{}-{}".format(r["Metadata_Plate"], r["Metadata_Well"], r["Metadata_Site"])
    dset = deepprofiler.dataset.image_dataset.ImageDataset(
        plate,
        config["dataset"]["metadata"]["label_field"],
        config["dataset"]["images"]["channels"],
        config["paths"]["images"],
        keyGen,
        config
    )

    # Configure compression object
    plate_output_dir = png_dir(config["paths"]["compressed_images"], plate_name)
    compress = Compress(
        stats,
        config["dataset"]["images"]["channels"],
        plate_output_dir
    )
    compress.set_formats(source_format=config["dataset"]["images"]["file_format"], target_format="png")
    compress.set_scaling_factor(config["prepare"]["compression"]["scaling_factor"])
    compress.recompute_percentile(0.0001, side="lower_percentile")
    compress.recompute_percentile(0.9999, side="upper_percentile")
    compress.expected = dset.number_of_records("all")

    # Setup control samples filter (for computing control illumination statistics)
    filter_func = lambda x: x[config["dataset"]["metadata"]["label_field"]] == config["dataset"]["metadata"]["control_value"]
    compress.set_control_samples_filter(filter_func)

    # Run compression
    dset.scan(compress.process_image, frame="all")

    # Retrieve and store results
    new_stats = compress.getUpdatedStats()
    with open(statsfile,"wb") as output:
        pickle.dump(new_stats, output)
