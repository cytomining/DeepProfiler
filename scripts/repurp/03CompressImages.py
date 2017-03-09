################################################################################
## Script for compressing images in the LINCS Cell Painting Pilot dataset
## Applies illumination correction, stretches histogram and converts to png
## Does not rescale images. Preserves same dimensions
## 02/27/2017. Broad Institute of MIT and Harvard
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
    for i in range(len(plates)):
        #plate = metadata.filterRecords(lambda df: (df.Metadata_Plate == plates[i]) & (df.Metadata_Well == "A01"), copy=True)
        plate = metadata.filterRecords(lambda df: (df.Metadata_Plate == plates[i]), copy=True)
        yield plate
    return

def compressBatch(args):
    plate, imgsDir, statsDir, outDir = args
    # Dataset parameters 
    statsfile = statsDir + plate.data.iloc[0]["Metadata_Plate"] + ".pkl"
    stats = pickle.load( open(statsfile, "rb") )
    keyGen = lambda x: x["Metadata_Plate"]+ "/" + x["Metadata_Well"] + "-" + x["Metadata_Site"]
    dataset = ds.Dataset(plate, "Treatment", CHANNELS, imgsDir, keyGen)
    # Configure compression object
    compress = px.Compress(stats, CHANNELS, outDir)
    compress.setFormats(sourceFormat="tiff", targetFormat="png")
    compress.setScalingFactor(0.5)
    compress.recomputePercentile(0.0001, side="lower")
    compress.recomputePercentile(0.9999, side="upper")
    compress.expected = dataset.numberOfRecords("all")
    # Treatment 0 is DMSO in the treatments.csv metadata file
    compress.setControlSamplesFilter(lambda x: x["Treatment"]=="0")
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

