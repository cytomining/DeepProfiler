import dataset.utils as utils
import dataset.image_dataset
import skimage.transform
import numpy as np
import pickle as pickle
from .illumination_correction import IlluminationCorrection


def illum_stats_filename(output_dir, plate_name):
    return output_dir + "/" + plate_name + "/intensities/" + plate_name + ".pkl"


#################################################
## COMPUTATION OF ILLUMINATION STATISTICS
#################################################

# Build pixel histogram for each channel
class IlluminationStatistics():
    def __init__(self, bits, channels, downScaleFactor, medianFilterSize, name=""):
        self.depth = 2 ** bits
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
                                                                      self.count, self.expected,
                                                                      100 * float(self.count) / self.expected))
        for i in range(len(self.channels)):
            counts = np.histogram(img[:, :, i], bins=self.depth, range=(0, self.depth))[0]
            self.hist[i] += counts.astype(np.float64)

    # Accumulate the mean image. Useful for illumination correction purposes
    def addToMean(self, img):
        # Check image size (we assume all images have the same size)
        if self.originalImageSize is None:
            self.originalImageSize = img.shape
            self.scale = (img.shape[0] / self.downScaleFactor, img.shape[1] / self.downScaleFactor)
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

    # Compute global statistics on pixels. 
    def computeStats(self):
        # Initialize counters
        bins = np.linspace(0, self.depth - 1, self.depth)
        mean = np.zeros((len(self.channels)))
        lower = np.zeros((len(self.channels)))
        upper = np.zeros((len(self.channels)))
        self.meanImage /= self.count

        # Compute percentiles and histogram
        for i in range(len(self.channels)):
            probs = self.hist[i] / self.hist[i].sum()
            mean[i] = (bins * probs).sum()
            lower[i] = self.percentile(probs, 0.0001)
            upper[i] = self.percentile(probs, 0.9999)
        stats = {"mean_values": mean, "upper_percentiles": upper, "lower_percentiles": lower, "histogram": self.hist,
                 "mean_image": self.meanImage, "channels": self.channels, "original_size": self.originalImageSize}

        # Compute illumination correction function and add it to the dictionary
        correct = IlluminationCorrection(stats, self.channels, self.originalImageSize)
        correct.computeAll(self.medianFilterSize)
        stats["illum_correction_function"] = correct.illumCorrFunc

        # Plate ready
        utils.logger.info('Plate ' + self.name + ' done')
        return stats


#################################################
## COMPUTE INTENSITY STATISTICS IN A SINGLE PLATE
#################################################

# TODO: try def calculate_stats(plate, config) DOES IT WORK???? I DUNNO
def calculate_statistics(args):
    # Load input parameters
    plate, config = args

    plateName = plate.data["Metadata_Plate"].iloc[0]

    # Create Dataset object
    keyGen = lambda r: "{}/{}-{}".format(r["Metadata_Plate"], r["Metadata_Well"], r["Metadata_Site"])

    dset = dataset.image_dataset.ImageDataset(
        plate,
        config["metadata"]["label_field"],
        config["original_images"]["channels"],
        config["original_images"]["path"],
        keyGen
    )

    # Prepare ImageStatistics object
    hist = IlluminationStatistics(
        config["original_images"]["bits"],
        config["original_images"]["channels"],
        config["illumination_correction"]["down_scale_factor"],
        config["illumination_correction"]["median_filter_size"],
        name=plateName
    )

    hist.expected = dset.numberOfRecords("all")

    # Run the intensity computation
    dset.scan(hist.processImage, frame="all")

    # Retrieve and store results
    stats = hist.computeStats()

    outfile = illum_stats_filename(config["compression"]["output_dir"], plateName)

    utils.check_path(outfile)

    with open(outfile, "wb") as output:
        pickle.dump(stats, output)

