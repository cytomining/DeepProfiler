################################################################################
## Script for compressing images in the LUAD dataset
## Applies illumination correction, stretches histogram and converts to png
## Does not rescale images. Preserves same dimensions
## 02/24/2017. Broad Institute of MIT and Harvard
################################################################################
import argparse
import data.metadata as meta
import data.utils as utils
import scripts.processing as prs
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="JSON configuration file")
    args = parser.parse_args()

    params = json.load(open(args.config, "r"))
    processor = utils.Parallel(params)
    processor.compute(prs.compressBatch, meta.readPlates(params["metadata"]["filename"]))

