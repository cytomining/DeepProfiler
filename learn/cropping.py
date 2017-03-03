import tensorflow as tf
import numpy as np
import pandas as pd
import os

#################################################
## BOUNDING BOX HANDLING
#################################################

def getCellLocations(cells_dir, image_key, random_sample=None):
    cells = pd.read_csv(os.path.join(cells_dir, image_key + ".csv"))
    if random_sample is not None and random_sample < len(cells):
        return cells.sample(random_sample)
    else:
        return cells

def prepareBoxes(cellsBatch, imageLabels, config):
    all_boxes = []
    all_indices = []
    all_labels = []
    index = 0
    for cells in cellsBatch:
        boxes = np.zeros((len(cells), 4), np.float32)
        boxes[:,0] = cells["Location_Center_Y"] - config["box_size"]/2
        boxes[:,1] = cells["Location_Center_X"] - config["box_size"]/2
        boxes[:,2] = cells["Location_Center_Y"] + config["box_size"]/2
        boxes[:,3] = cells["Location_Center_X"] + config["box_size"]/2
        boxes[:,[0,2]] /= config["image_height"]
        boxes[:,[1,3]] /= config["image_width"]
        box_ind = index * np.ones((len(cells)), np.int32)
        labels = imageLabels[index] * np.ones((len(cells)), np.int32)
        all_boxes.append(boxes)
        all_indices.append(box_ind)
        all_labels.append(labels)
        index += 1
    return np.concatenate(all_boxes), np.concatenate(all_indices), np.concatenate(all_labels)

def loadBatch(dataset, cells_dir, config):
    batch = dataset.getTrainBatch(config["image_batch_size"])
    batch["cells"] = [getCellLocations(cells_dir, x, config["sample_cells"]) for x in batch["keys"]]
    return batch

#################################################
## CROPPING AND TRANSFORMATION OPERATIONS
#################################################

def crop(image_ph, boxes_ph, box_ind_ph, box_size):
    with tf.device('/cpu:0'):
        with tf.variable_scope("cropping"):
            crop_size_ph = tf.constant([box_size, box_size], name="crop_size")
        crops = tf.image.crop_and_resize(image_ph, boxes_ph, box_ind_ph, crop_size_ph)
    return crops

def augment(crop):
    with tf.device('/cpu:0'):
        with tf.variable_scope("augmentation"):
            augmented = tf.image.random_flip_left_right(crop)
            angle = tf.random_uniform([1], minval=0, maxval=3, dtype=tf.int32)
            augmented = tf.image.rot90(augmented, angle[0])
    return augmented


