import tensorflow as tf
import numpy as np
import pandas as pd
import os

PI = 3.1415926539

#################################################
## BOUNDING BOX HANDLING
#################################################

def getLocations(image_key, config, randomize=True):
    keys = image_key.split("/")
    locations_file = "{}/locations/{}-{}.csv".format(
        keys[0], 
        keys[1], 
        config["sampling"]["locations_field"]
    )
    locations = pd.read_csv(os.path.join(config["image_set"]["path"], locations_file))
    random_sample = config["sampling"]["locations"]
    if randomize and random_sample is not None and random_sample < len(locations):
        return locations.sample(random_sample)
    else:
        return locations


def prepareBoxes(batch, config):
    locationsBatch = batch["locations"]
    imageLabels = batch["labels"]
    images = batch["images"]
    all_boxes = []
    all_indices = []
    all_labels = []
    all_masks = []
    index = 0
    y_key = config["sampling"]["locations_field"] + "_Location_Center_Y"
    x_key = config["sampling"]["locations_field"] + "_Location_Center_X"
    for locations in locationsBatch:
        # Collect and normalize boxes between 0 and 1
        boxes = np.zeros((len(locations), 4), np.float32)
        boxes[:,0] = locations[y_key] - config["sampling"]["box_size"]/2
        boxes[:,1] = locations[x_key] - config["sampling"]["box_size"]/2
        boxes[:,2] = locations[y_key] + config["sampling"]["box_size"]/2
        boxes[:,3] = locations[x_key] + config["sampling"]["box_size"]/2
        boxes[:,[0,2]] /= config["image_set"]["height"]
        boxes[:,[1,3]] /= config["image_set"]["width"]
        # Create indicators for this set of boxes, belonging to the same image
        box_ind = index * np.ones((len(locations)), np.int32)
        # Propage the same label to all crops
        labels = imageLabels[index] * np.ones((len(locations)), np.int32)
        # Identify object mask for each crop
        masks = np.zeros(len(locations), np.int32)
        if config["image_set"]["mask_objects"]:
            i = 0
            for lkey in locations.index:
                y = int(locations.loc[lkey, y_key])
                x = int(locations.loc[lkey, x_key])
                masks[i] = int(images[index][y, x, -1])
                i += 1
        # Pile up the resulting variables
        all_boxes.append(boxes)
        all_indices.append(box_ind)
        all_labels.append(labels)
        all_masks.append(masks)
        index += 1
    result = (np.concatenate(all_boxes), 
              np.concatenate(all_indices), 
              np.concatenate(all_labels),
              np.concatenate(all_masks)
             )
    return result


def loadBatch(dataset, config):
    batch = dataset.getTrainBatch(config["sampling"]["images"])
    batch["locations"] = [ getLocations(x, config) for x in batch["keys"] ]
    return batch


#################################################
## CROPPING AND TRANSFORMATION OPERATIONS
#################################################

def crop(image_ph, boxes_ph, box_ind_ph, mask_ind_ph, box_size, mask_boxes=False):
    with tf.variable_scope("cropping"):
        crop_size_ph = tf.constant([box_size, box_size], name="crop_size")
        crops = tf.image.crop_and_resize(image_ph, boxes_ph, box_ind_ph, crop_size_ph)
        if mask_boxes:
            mask_ind = tf.expand_dims(tf.expand_dims(mask_ind_ph, -1), -1)
            mask_values = tf.ones_like(crops[:,:,:,-1], dtype=tf.float32) * tf.cast(mask_ind, dtype=tf.float32)
            masks = tf.to_float( tf.equal(crops[:,:,:,-1], mask_values) )
            crops = crops[:,:,:,0:-1] * tf.expand_dims(masks, -1)
    return crops


def augment(crop):
    with tf.variable_scope("augmentation"):
        augmented = tf.image.random_flip_left_right(crop)
        angle = tf.random_uniform([1], minval=0.0, maxval=2*PI, dtype=tf.float32)
        augmented = tf.contrib.image.rotate(augmented, angle[0], interpolation="BILINEAR")
        illum = tf.random_uniform([1], minval=-0.1, maxval=0.1, dtype=tf.float32)
        augmented = augmented + illum
    return augmented


def aument_multiple(crops, parallel=10):
    with tf.variable_scope("augmentation"):
        return tf.map_fn(augment, crops, parallel_iterations=parallel)

