################################################################################
## Script for compressing images in the LUAD dataset
## Applies illumination correction, stretches histogram and converts to png
## Does not rescale images. Preserves same dimensions
## 02/24/2017. Broad Institute of MIT and Harvard
################################################################################
import argparse
import data.metadata as meta
import data.dataset as ds
import data.utils as utils
import data.pixels as px
import pandas as pd
import pickle as pickle

CHANNELS = ["RNA","ER","AGP","Mito","DNA"]

def readMetadata(metaFile):
    metadata = meta.Metadata(metaFile)
    plates = metadata.data["Metadata_Plate"].unique()
    print("Total plates:",len(plates))
    #plate = metadata.filterRecords(lambda df: (df.Metadata_Plate == plates[0]) & (df.Metadata_Well == "a01"), copy=True)
    for i in range(len(plates)):
        plate = metadata.filterRecords(lambda df: (df.Metadata_Plate == plates[i]), copy=True)
        yield plate
    return

def compressBatch(args):
    plate, imgsDir, statsDir, outDir = args
    # Dataset parameters
    statsfile = statsDir + plate.data.iloc[0]["Metadata_Plate"] + ".pkl"
    stats = pickle.load( open(statsfile, "rb") )
    keyGen = lambda r: "{}/{}-{}".format(r["Metadata_Plate"], r["Metadata_Well"], r["Metadata_Site"])
    dataset = ds.Dataset(plate, "Allele", CHANNELS, imgsDir, keyGen)
    # Configure compression object
    compress = px.Compress(stats, CHANNELS, outDir)
    compress.setFormats(sourceFormat="tif", targetFormat="png")
    compress.setScalingFactor(1.0)
    compress.recomputePercentile(0.0001, side="lower")
    compress.recomputePercentile(0.9999, side="upper")
    compress.expected = dataset.numberOfRecords("all")
    # Allele 17 is EMPTY in the alleles.csv metadata file
    compress.setControlSamplesFilter(lambda x: x["Allele"]=="17")
    # Run compression and save results
    dataset.scan(compress.processImage, frame="all")
    new_stats = compress.getUpdatedStats()
    with open(statsfile,"wb") as output:
        pickle.dump(new_stats, output)
    return 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata", help="Metadata csv file with paths to all images")
    parser.add_argument("root_dir", help="Path where the images directory is stored")
    parser.add_argument("stats_dir", help="Path where stats files are stored")
    parser.add_argument("output_dir", help="Directory to store the compressed images")
    args = parser.parse_args()

    manager = utils.Parallel()
    manager.compute(compressBatch, readMetadata(args.metadata), [args.root_dir, args.stats_dir, args.output_dir])
    # TODO: Aggregate control distributions from all plates

