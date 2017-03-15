################################################################################
## Script for computing intensity statistics on the LUAD dataset
## Calculates intensity distributions, percentiles and mean image
## 02/23/2017. Broad Institute of MIT and Harvard
################################################################################

import argparse
import data.metadata as meta
import data.utils as utils
import scripts.processing as prs
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="JSON configuration file for data preparation")
    args = parser.parse_args()

    params = json.load(open(args.config))
    process = utils.Parallel(params)
    process.compute(prs.intensityStats, meta.readPlates(params["metadata"]["filename"]))

