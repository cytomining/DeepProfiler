################################################################################
## Script for compressing images in the LUAD dataset
## Applies illumination correction, stretches histogram and converts to png
## Does not rescale images. Preserves same dimensions
## 02/24/2017. Broad Institute of MIT and Harvard
################################################################################
import argparse
import pandas as pd
import pickle as pickle
import data.metadata as meta
import data.dataset as ds
import data.utils as utils
import data.compression as cmpr
import scripts.processing as prs

CHANNELS = ["RNA","ER","AGP","Mito","DNA"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata", help="Metadata csv file with paths to all images")
    parser.add_argument("root_dir", help="Path where the images directory is stored")
    parser.add_argument("stats_dir", help="Path where stats files are stored")
    parser.add_argument("output_dir", help="Directory to store the compressed images")
    args = parser.parse_args()

    params = {"images_dir":args.root_dir, "stats_dir":args.stats_dir, "output_dir":args.output_dir,
              "channels": CHANNELS, "source_format":"tif", "scaling_factor":1.0, "label_field":"Alleles" 
              "control_field":"Allele", "control_value":"17"}
    processor = utils.Parallel(params)
    processor.compute(prs.compressBatch, meta.readPlates(args.metadata))

