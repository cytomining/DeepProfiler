################################################################################
## Script for learning a basic CNN from single cell images 
## Takes metadata and cell locations to read images and produce crops in a queue 
## Crops are consumed by a training routine.
## 02/28/2017. Broad Institute of MIT and Harvard
################################################################################
import argparse
import data.metadata as meta
import data.dataset as ds
import learn.training as training

CHANNELS = ["RNA","ER","AGP","Mito","DNA"]
SAMPLE_CELLS = 20       # Number of cells sampled per image
IMAGE_BATCH_SIZE = 10   # Number of images read to load cells
BOX_SIZE = 256
IMAGE_WIDTH = 1080
IMAGE_HEIGHT = 1080

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata", help="Metadata csv file with paths to all images")
    parser.add_argument("images_dir", help="Path where the images directory is found")
    parser.add_argument("cells_dir", help="Path where the cell locations directory is found")
    parser.add_argument("output_dir", help="Directory to store extracted feature files")
    args = parser.parse_args()

    config = {'channels':CHANNELS, 'sample_cells':SAMPLE_CELLS, 'image_batch_size':IMAGE_BATCH_SIZE,
              'box_size':BOX_SIZE, 'image_width':IMAGE_WIDTH, 'image_height':IMAGE_HEIGHT,
              'fifo_queue_size':4096, 'random_queue_size':4096, 'cropping_workers':6, 'augmentation_workers':2,
              'minibatch_size':128, 'training_iterations':10}

    images = readDataset(args.metadata, args.images_dir)
    training.learnCNN(config, images, args.cells_dir, args.output_dir)

