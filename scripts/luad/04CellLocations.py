################################################################################
## Script for creating cell location files for the LUAD dataset 
## Cells have been identified by CellProfiler. Here we read the output and
## transform it into simple files that can be read efficiently for training
## 02/27/2017. Broad Institute of MIT and Harvard
################################################################################
import argparse
import data.metadata as meta
import data.dataset as ds
import data.utils as utils
import pandas as pd
import os

CHANNELS = ["RNA","ER","AGP","Mito","DNA"]

def readMetadata(metaFile):
    metadata = meta.Metadata(metaFile)
    plates = metadata.data["Metadata_Plate"].unique()
    print("Total plates:",len(plates))
    for i in range(len(plates)):
        #plate = metadata.filterRecords(lambda df: (df.Metadata_Plate == plates[0]) & (df.Metadata_Well == "a01"), copy=True)
        plate = metadata.filterRecords(lambda df: (df.Metadata_Plate == plates[i]), copy=True)
        yield plate
    return

def createCellFiles(args):
    plate, featuresDir, outDir = args
    os.system("mkdir -p " + outDir + plate.data.iloc[0]["Metadata_Plate"])
    iteration = 1
    for index, row in plate.data.iterrows():
        # Read cells file for each image
        features_file = row["Metadata_Plate"] + "/analysis/" + row["Metadata_Well"].upper() + "-" + row["Metadata_Site"] + "/Cells.csv"
        features = pd.read_csv(featuresDir + features_file)
        # Keep center coordinates only, remove NaNs, and transform to integers
        cell_locations = features[["Location_Center_X","Location_Center_Y"]].copy()
        cell_locations = cell_locations.dropna(axis=0)
        cell_locations["Location_Center_X"] = cell_locations["Location_Center_X"].astype(int)
        cell_locations["Location_Center_Y"] = cell_locations["Location_Center_Y"].astype(int)
        # Save the resulting data frame in the output directory
        cells_file = row["Metadata_Plate"] + "/" + row["Metadata_Well"] + "-" + row["Metadata_Site"] + ".csv"
        cell_locations.to_csv(outDir + cells_file, index=False)
        utils.printProgress(iteration, len(plate.data), prefix='Cell locations in plate')
        iteration += 1
    print("")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata", help="Metadata csv file with paths to all images")
    parser.add_argument("features_dir", help="Path where the features directory is found")
    parser.add_argument("output_dir", help="Directory to store the cell files")
    args = parser.parse_args()

    manager = utils.Parallel()
    manager.compute(createCellFiles, readMetadata(args.metadata), [args.features_dir, args.output_dir])

