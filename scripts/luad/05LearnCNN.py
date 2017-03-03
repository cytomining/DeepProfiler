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

import tensorflow as tf
import numpy as np

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

## BOX HANDLING CODE BEGINS

def getCellLocations(cells_dir, image_key, random_sample=None):
    cells = pd.read_csv(os.path.join(cells_dir, image_key + ".csv"))
    if random_sample is not None and random_sample < len(cells):
        return cells.sample(random_sample)
    else:
        return cells

def prepareBoxes(cellsBatch, imageLabels):
    all_boxes = []
    all_indices = []
    all_labels = []
    index = 0
    for cells in cellsBatch:
        boxes = np.zeros((len(cells), 4), np.float32)
        boxes[:,0] = cells["Location_Center_Y"] - BOX_SIZE/2
        boxes[:,1] = cells["Location_Center_X"] - BOX_SIZE/2
        boxes[:,2] = cells["Location_Center_Y"] + BOX_SIZE/2
        boxes[:,3] = cells["Location_Center_X"] + BOX_SIZE/2
        boxes[:,[0,2]] /= IMAGE_HEIGHT
        boxes[:,[1,3]] /= IMAGE_WIDTH
        box_ind = index * np.ones((len(cells)), np.int32)
        labels = imageLabels[index] * np.ones((len(cells)), np.int32)
        all_boxes.append(boxes)
        all_indices.append(box_ind)
        all_labels.append(labels)
        index += 1
    return np.concatenate(all_boxes), np.concatenate(all_indices), np.concatenate(all_labels)

def loadBatch(dataset, cells_dir):
    batch = dataset.getTrainBatch(IMAGE_BATCH_SIZE)
    batch["cells"] = [getCellLocations(cells_dir, x, SAMPLE_CELLS) for x in batch["keys"]]
    return batch

## BOX HANDLING CODE ENDS


## MODEL DEFINITION BEGINS
def crop(image_ph, boxes_ph, box_ind_ph):
    with tf.device('/cpu:0'):
        with tf.variable_scope("cropping"):
            crop_size_ph = tf.constant([BOX_SIZE, BOX_SIZE], name="crop_size")
        crops = tf.image.crop_and_resize(image_ph, boxes_ph, box_ind_ph, crop_size_ph)
    return crops

def augment(crop):
    # Data augmentation queue
    with tf.device('/cpu:0'):
        with tf.variable_scope("augmentation"):
            augmented = tf.image.random_flip_left_right(crop)
            angle = tf.random_uniform([1], minval=0, maxval=3, dtype=tf.int32)
            augmented = tf.image.rot90(augmented, angle[0])
    return augmented

## MODEL DEFINITION ENDS

def learnCNN(dataset, cells_dir, output_dir):
    # Inputs to the graph
    image_ph = tf.placeholder(tf.float32, shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, len(CHANNELS)], name="raw_images")
    boxes_ph = tf.placeholder(tf.float32, shape=[None, 4], name="cell_boxes")
    box_ind_ph = tf.placeholder(tf.int32, shape=[None], name="box_indicators")
    labels_ph = tf.placeholder(tf.int32, shape=[None], name="image_labels")

    # Outputs and queue of the cropping graph
    crop_op = crop(image_ph, boxes_ph, box_ind_ph)
    daug_queue = tf.FIFOQueue(4096, [tf.float32, tf.int32])
    daug_enqueue_op = daug_queue.enqueue_many([crop_op, labels_ph])
    labeled_crops = daug_queue.dequeue()

    # Outputs and queue of the data augmentation graph
    augmented_op = augment(labeled_crops[0])
    train_queue = tf.RandomShuffleQueue(4096, 20, [tf.float32, tf.int32], shapes=[(256, 256, 5), ()])
    train_enqueue_op = train_queue.enqueue([augmented_op, labeled_crops[1]])
    train_inputs = train_queue.dequeue_many(128)

    # Start session
    sess = tf.Session()
    coord = tf.train.Coordinator()
    import threading
    import time

    # Enqueuing thread for raw images
    def data_enqueue_thread():
        while not coord.should_stop():
            # Load images and cell boxes
            batch = loadBatch(dataset, cells_dir)
            images = np.reshape(batch["images"], (IMAGE_BATCH_SIZE,IMAGE_HEIGHT,IMAGE_WIDTH,len(CHANNELS) ))
            boxes, box_ind, labels = prepareBoxes(batch["cells"], batch["labels"])
            sess.run(daug_enqueue_op, {image_ph:images, boxes_ph:boxes, box_ind_ph:box_ind, labels_ph:labels})
            print("Images enqueued",images.shape, boxes.shape, labels.shape, box_ind.shape)
    for _ in range(6):
        threading.Thread(target=data_enqueue_thread).start()

    # Enqueuing thread for labeled crops
    def augm_enqueue_thread():
        while not coord.should_stop():
            sess.run(train_enqueue_op)
    for _ in range(2):
        threading.Thread(target=augm_enqueue_thread).start()

    # Main training loop
    while True:
        if coord.should_stop():
            break
        training_data = sess.run(train_inputs)
        print('Cells used for training',training_data[0].shape)
        time.sleep(0.5)
        
    coord.request_stop()
    coord.join(enq_threads1)
    coord.join(enq_threads2)
    print(batch.keys())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata", help="Metadata csv file with paths to all images")
    parser.add_argument("images_dir", help="Path where the images directory is found")
    parser.add_argument("cells_dir", help="Path where the cell locations directory is found")
    parser.add_argument("output_dir", help="Directory to store extracted feature files")
    args = parser.parse_args()

    images = readDataset(args.metadata, args.images_dir)
    learnCNN(images, args.cells_dir, args.output_dir)

