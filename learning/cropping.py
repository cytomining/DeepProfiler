import tensorflow as tf
import numpy as np
import pandas as pd
import os

#################################################
## BOUNDING BOX HANDLING
#################################################

def getLocations(image_key, config):
    keys = image_key.split("/")
    locations_file = "{}/locations/{}-{}.csv".format(
        keys[0], 
        keys[1], 
        config["sampling"]["locations_field"]
    )
    locations = pd.read_csv(os.path.join(config["image_set"]["path"], locations_file))
    random_sample = config["sampling"]["locations"]
    if random_sample is not None and random_sample < len(locations):
        return locations.sample(random_sample)
    else:
        return locations

def prepareBoxes(locationsBatch, imageLabels, config):
    all_boxes = []
    all_indices = []
    all_labels = []
    index = 0
    y_key = config["sampling"]["locations_field"] + "_Location_Center_Y"
    x_key = config["sampling"]["locations_field"] + "_Location_Center_X"
    for locations in locationsBatch:
        boxes = np.zeros((len(locations), 4), np.float32)
        boxes[:,0] = locations[y_key] - config["sampling"]["box_size"]/2
        boxes[:,1] = locations[x_key] - config["sampling"]["box_size"]/2
        boxes[:,2] = locations[y_key] + config["sampling"]["box_size"]/2
        boxes[:,3] = locations[x_key] + config["sampling"]["box_size"]/2
        boxes[:,[0,2]] /= config["image_set"]["height"]
        boxes[:,[1,3]] /= config["image_set"]["width"]
        box_ind = index * np.ones((len(locations)), np.int32)
        labels = imageLabels[index] * np.ones((len(locations)), np.int32)
        all_boxes.append(boxes)
        all_indices.append(box_ind)
        all_labels.append(labels)
        index += 1
    return np.concatenate(all_boxes), np.concatenate(all_indices), np.concatenate(all_labels)

def loadBatch(dataset, config):
    batch = dataset.getTrainBatch(config["sampling"]["images"])
    batch["locations"] = [ getLocations(x, config) for x in batch["keys"] ]
    return batch

#################################################
## CROPPING AND TRANSFORMATION OPERATIONS
#################################################

def crop(image_ph, boxes_ph, box_ind_ph, box_size):
    with tf.variable_scope("cropping"):
        crop_size_ph = tf.constant([box_size, box_size], name="crop_size")
        crops = tf.image.crop_and_resize(image_ph, boxes_ph, box_ind_ph, crop_size_ph)
    return crops

def augment(crop):
    with tf.variable_scope("augmentation"):
        augmented = tf.image.random_flip_left_right(crop)
        angle = tf.random_uniform([1], minval=0, maxval=3, dtype=tf.int32)
        augmented = tf.image.rot90(augmented, angle[0])
    return augmented

def aument_multiple(crops, parallel=10):
    with tf.variable_scope("augmentation"):
        return tf.map_fn(augment, crops, parallel_iterations=parallel)
