################################################################################
## Script for learning a basic CNN from single cell images 
## Takes metadata and cell locations to read images and produce crops in a queue 
## Crops are consumed by a training routine.
## 02/28/2017. Broad Institute of MIT and Harvard
################################################################################
import argparse
import data.metadata as meta
import data.dataset as ds
import data.utils as utils
import pandas as pd
import os

CHANNELS = ["RNA","ER","AGP","Mito","DNA"]
SAMPLE_CELLS = 20       # Number of cells sampled per image
IMAGE_BATCH_SIZE = 20   # Number of images read to load cells

def readDataset(metaFile, images_dir):
    # Read metadata and split data in training and validation
    metadata = meta.Metadata(metaFile, dtype=None)
    trainingFilter = lambda df: df["Allele_Replicate"] <= 5
    validationFilter = lambda df: df["Allele_Replicate"] > 5
    metadata.splitMetadata(trainingFilter, validationFilter)
    # Create a dataset
    keyGen = lambda r: "{}/{}-{}".format(r["Metadata_Plate"], r["Metadata_Well"], r["Metadata_Site"])
    dataset = ds.Dataset(metadata, "Allele", CHANNELS, images_dir, keyGen)
    print(metadata.data.iloc[100])
    return dataset

def getCellLocations(cells_dir, image_key, random_sample=None):
    cells = pd.read_csv(os.path.join(cells_dir, image_key + ".csv"))
    if random_sample is not None:
        return cells.sample(random_sample)
    else:
        return cells

def learnCNN(dataset, cells_dir, output_dir):
    batch = dataset.getTrainBatch(IMAGE_BATCH_SIZE)
    batch["cells"] = [getCellLocations(cells_dir, x, SAMPLE_CELLS) for x in batch["keys"]]
    print(batch.keys())
    print(batch["keys"], batch["labels"])
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata", help="Metadata csv file with paths to all images")
    parser.add_argument("images_dir", help="Path where the images directory is found")
    parser.add_argument("cells_dir", help="Path where the cell locations directory is found")
    parser.add_argument("output_dir", help="Directory to store extracted feature files")
    args = parser.parse_args()

    images = readDataset(args.metadata, args.images_dir)
    learnCNN(images, args.cells_dir, args.output_dir)

