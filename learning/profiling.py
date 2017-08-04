import os
import numpy as np
import pandas
import glob
import skimage.transform

import tensorflow as tf
from tensorflow.contrib import slim
# export PYTHONPATH=/home/jccaicedo/luad-playground/models/slim
from nets import inception
from datasets import dataset_utils

import learning.training
import learning.cropping
from dataset.utils import tic, toc


num_features = 1536
image_size = inception.inception_resnet_v2.default_image_size


def get_features(crops, sess, placeholder, endpoints, config):
    num_channels = len(config["image_set"]["channels"])
    everything = np.zeros(shape=(num_channels, len(crops), image_size, image_size, 3))
    features = np.zeros(shape=(num_channels, len(crops), num_features))
    s = tic()
    for i, crop in enumerate(crops):
        # Crops are rescaled to match the network's input size. Normalization is applied too
        crop = skimage.transform.resize(crop, (image_size, image_size), mode='reflect')
        crop = 2 * (crop - 0.5)
        # Each channel is converted to RGB
        for j in range(num_channels):
            channel = crop[:,:,j]
            channel = channel[:,:,np.newaxis]
            channel = np.tile(channel,(1,1,3))
            everything[j,i,:,:,:] = channel
    toc("Crop resizing and tiling", s)
    # Channels are forward to the network to get features
    s = tic()
    for i in range(num_channels):
        features[i,:,:] = sess.run(endpoints['PreLogitsFlatten'], feed_dict={placeholder:everything[i,:,:,:,:]})
    toc("Feature computation for all channels", s)
    return features


def profile(config, dset):
    # Setup pretrained model 
    placeholder = tf.placeholder(tf.float32, shape=(None, image_size, image_size, 3))
    url = config["profiling"]["url"]
    checkpoint = config["profiling"]["checkpoint"]
    if not os.path.isfile(checkpoint):
        dataset_utils.download_and_uncompress_tarball(url, os.path.dirname(checkpoint))
    with slim.arg_scope(inception.inception_resnet_v2_arg_scope()):
        _, endpoints = inception.inception_resnet_v2(placeholder, num_classes=1001, is_training=False)
    init_fn = slim.assign_from_checkpoint_fn(checkpoint, slim.get_model_variables())
   
    
    # Variables and cropping comp. graph
    num_channels = len(config["image_set"]["channels"])
    num_classes = dset.numberOfClasses()
    input_vars = learning.training.input_graph(config)
    images = input_vars["labeled_crops"][0]
    labels = tf.one_hot(input_vars["labeled_crops"][1], num_classes)
    
    # Session configuration
    configuration = tf.ConfigProto()
    configuration.gpu_options.allow_growth = True
    configuration.gpu_options.visible_device_list = config["profiling"]["gpu"]
   
    sess = tf.Session(config=configuration)
    init_fn(sess)


    def check(meta):
        output_file = config["profiling"]["output_dir"] + "/{}_{}_{}.npz"
        output_file = output_file.format( meta["Metadata_Plate"], meta["Metadata_Well"], meta["Metadata_Site"])

        # Check if features were computed before
        if os.path.isfile(output_file):
            print("Already done:", output_file)
            return False
        else:
            return True

    
    # Function to process a single image
    def extract_features(key, image_array, meta):
        output_file = config["profiling"]["output_dir"] + "/{}_{}_{}.npz"
        output_file = output_file.format( meta["Metadata_Plate"], meta["Metadata_Well"], meta["Metadata_Site"])

        # Prepare image and crop locations
        batch_size = config["training"]["minibatch"]
        image_key, image_names = dset.getImagePaths(meta)
        locations = [ learning.cropping.getLocations(image_key, config, randomize=False) ]
        if len(locations[0]) == 0:
            print("Empty locations set:", str(key))
            return
        # Pad last batch with empty locations
        pads = batch_size - len(locations[0]) % batch_size
        zero_pads = np.zeros(shape=(pads, 2), dtype=np.int32)
        pad_data = pandas.DataFrame(columns=locations[0].columns, data=zero_pads)
        locations[0] = pandas.concat((locations[0], pad_data))   

        # Prepare boxes, indices, labels and push the image to the queue
        labels_data = [ meta[config["training"]["label_field"]] ]
        boxes, box_ind, labels_data = learning.cropping.prepareBoxes(locations, labels_data, config)
        images_data = np.reshape(image_array, input_vars["shapes"]["batch"])

        sess.run(input_vars["enqueue_op"], {
                        input_vars["image_ph"]:images_data,
                        input_vars["boxes_ph"]:boxes,
                        input_vars["box_ind_ph"]:box_ind,
                        input_vars["labels_ph"]:labels_data
        })

        # Collect crops of from the queue
        items = sess.run(input_vars["queue"].size())
        data = np.zeros(shape=(num_channels, len(locations[0]), num_features))
        b = 0
        start = tic()
        while items >= batch_size:
            # Compute features in a batch of crops
            crops = sess.run(images)
            feats = get_features(crops, sess, placeholder, endpoints, config)
            data[:, b * batch_size:(b + 1) * batch_size, :] = feats
            items = sess.run(input_vars["queue"].size())
            b += 1

        # Save features
        np.savez_compressed(output_file, f=data[:, :-pads, :])
        toc(image_key, start)
        
    dset.scan(extract_features, frame="all", check=check)
    print("Profiling: done")
