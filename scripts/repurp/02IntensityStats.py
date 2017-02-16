################################################################################
## Script for computing intensity statistics on the repurp dataset
## Calculates intensity distributions, percentiles and mean image
## 11/22/2016. Broad Institute of MIT and Harvard
################################################################################

import argparse
import data.metadata as meta
import data.dataset as ds
import data.utils as utils
import data.pixels as px
import pandas as pd
import pickle as pickle

## Settings for this dataset
# Channels used in this dataset (order is important)
CHANNELS = ["DNA","ER","RNA","AGP","Mito"]
# Pixel depth
BITS = 16
# Parameters for illumination correction
DOWN_SCALE_FACTOR = 4    # Make images 4 times smaller for aggregation
MEDIAN_FILTER_SIZE = 50  # Create a filter of 50 pixels in diameter

## Generator of plates. Reads metadata and yields plates
def readPlates(metaFile):
    metadata = meta.Metadata(metaFile)
    plates = metadata.data["Metadata_Plate"].unique()
    utils.logger.info("Total plates: " + str(len(plates)))
    for i in range(len(plates)):
        plate = metadata.filterRecords(lambda df: (df.Metadata_Plate == plates[i]), copy=True)
        yield plate
    return

## Computation of intensity stats per plate
def intensityStats(args):
    plate, root, outDir = args
    plateName = plate.data["Metadata_Plate"].iloc[0]
    dataset = ds.Dataset(plate, "Treatment", CHANNELS, root)
    hist = px.ImageStatistics(BITS, CHANNELS, DOWN_SCALE_FACTOR, MEDIAN_FILTER_SIZE, name=plateName)
    hist.expected = dataset.numberOfRecords("all")
    dataset.scan(hist.processImage, frame="all")
    stats = hist.computeStats()
    outfile = outDir + plateName + ".pkl"
    with open(outfile,"wb") as output:
        pickle.dump(stats, output)
    return

## Main script
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata", help="Metadata csv file with paths to all images")
    parser.add_argument("root_dir", help="Path where the images directory is stored")
    parser.add_argument("output_dir", help="Directory to store the statistics")
    args = parser.parse_args()

    manager = utils.Parallel()
    manager.compute(intensityStats, readPlates(args.metadata), [args.root_dir, args.output_dir])

