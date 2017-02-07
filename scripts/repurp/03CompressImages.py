
import argparse
import data.metadata as meta
import data.dataset as ds
import data.utils as utils
import data.pixels as px
import pandas as pd
import cPickle as pickle

CHANNELS = ["RNA","ER","AGP","Mito","DNA"]

def readMetadata(metaFile):
    metadata = meta.Metadata(metaFile)
    plates = metadata.data["Metadata_Plate"].unique()
    print "Total plates:",len(plates)
    plate = metadata.filterRecords(lambda df: (df.Metadata_Plate == plates[0]) & (df.Metadata_Well == "A01"), copy=True)
    #plate = metadata.filterRecords(lambda df: df.Metadata_Plate == plates[0], copy=True)
    ## TODO: Iterate over plates and yield each
    return plate

def compressBatch(plate, imgsDir, statsDir, outDir):
    statsfile = statsDir + plate.data["Metadata_Plate"][0] + ".pkl"
    stats = pickle.load( open(statsfile, "r") )
    dataset = ds.Dataset(plate, "Treatment", CHANNELS, imgsDir)
    compress = px.Compress(stats, CHANNELS, outDir)
    compress.recomputePercentile(0.00005, side="lower")
    compress.recomputePercentile(0.99995, side="upper")
    compress.expected = dataset.numberOfRecords("all")
    dataset.scan(compress.processImage, frame="all")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata", help="Metadata csv file with paths to all images")
    parser.add_argument("root_dir", help="Path where the images directory is stored")
    parser.add_argument("stats_dir", help="Path where stats files are stored")
    parser.add_argument("output_dir", help="Directory to store the compressed images")
    args = parser.parse_args()

    plate = readMetadata(args.metadata)
    compressBatch(plate, args.root_dir, args.stats_dir, args.output_dir)
