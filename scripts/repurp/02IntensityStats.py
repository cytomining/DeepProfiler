
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
    #plate = metadata.filterRecords(lambda df: (df.Metadata_Plate == plates[0]) & (df.Metadata_Well == "A01"), copy=True)
    plate = metadata.filterRecords(lambda df: df.Metadata_Plate == plates[0], copy=True)
    ## TODO: Iterate over plates and yield each
    return plate

def computeMaxIntensity(plate, root, outDir):
    dataset = ds.Dataset(plate, "Treatment", CHANNELS, root)
    hist = px.Histogram(16, 5)
    hist.expected = dataset.numberOfRecords("all")
    dataset.scan(hist.processImage, frame="all")
    stats = hist.computeStats()
    outfile = outDir + plate.data["Metadata_Plate"][0] + ".pkl"
    with open(outfile,"wb") as output:
        pickle.dump(stats, output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata", help="Metadata csv file with paths to all images")
    parser.add_argument("root_dir", help="Path where the images directory is stored")
    parser.add_argument("output_dir", help="Directory to store the statistics")
    args = parser.parse_args()

    plate = readMetadata(args.metadata)
    computeMaxIntensity(plate, args.root_dir, args.output_dir) 
