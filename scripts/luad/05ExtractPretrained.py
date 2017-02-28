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

def readDataset(metaFile, images_dir):
    metadata = meta.Metadata(metaFile)
    dataset = ds.Dataset(metadata, "Allele", CHANNELS, images_dir)
    return dataset

def getCellLocations(image_record, cells_dir):
    return

def extractFeatures(dataset, cells_dir, output_dir):
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata", help="Metadata csv file with paths to all images")
    parser.add_argument("images_dir", help="Path where the images directory is found")
    parser.add_argument("cells_dir", help="Path where the cell locations directory is found")
    parser.add_argument("output_dir", help="Directory to store extracted feature files")
    args = parser.parse_args()

    images = readDataset(args.metadata, args.images_dir)
    extractFeatures(images, args.cells_dir, args.output_dir)

