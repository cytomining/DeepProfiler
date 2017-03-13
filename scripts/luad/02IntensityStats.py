################################################################################
## Script for computing intensity statistics on the LUAD dataset
## Calculates intensity distributions, percentiles and mean image
## 02/23/2017. Broad Institute of MIT and Harvard
################################################################################

import argparse
import pandas as pd
import pickle as pickle
import data.metadata as meta
import data.dataset as ds
import data.utils as utils
import data.image_statistics as ists
import scripts.processing as prs

## Settings for this dataset
CHANNELS = ["DNA","ER","RNA","AGP","Mito"]  # Channels used in this dataset (order is important)
BITS = 16 # Pixel depth
# Parameters for illumination correction
DOWN_SCALE_FACTOR = 4    # Make images 4 times smaller for aggregation
MEDIAN_FILTER_SIZE = 24  # Create a filter of 50 pixels in diameter

## Main script
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata", help="Metadata csv file with paths to all images")
    parser.add_argument("root_dir", help="Path where the images directory is stored")
    parser.add_argument("output_dir", help="Directory to store the statistics")
    args = parser.parse_args()

    params = {"channels": CHANNELS, "bits":BITS, "down_scale_factor":DOWN_SCALE_FACTOR, "median_filter_size":MEDIAN_FILTER_SIZE,
              "image_dir":args.root_dir, "output_dir":args.output_dir, "label_field":"Alleles"}
    process = utils.Parallel(params)
    process.compute(prs.intensityStats, meta.readPlates(args.metadata))

