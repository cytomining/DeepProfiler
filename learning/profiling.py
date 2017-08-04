import numpy as np
import tensorflow as tf

from datasets import dataset_utils

from tensorflow.contrib import slim

import learning.training
import learning.cropping

from dataset.utils import tic, toc

import numpy as np
import os
import tensorflow as tf

import skimage.transform

import pandas

import glob

from nets import inception

image_size = inception.inception_resnet_v2.default_image_size

num_features = 1536

def get_features(crops, sess, placeholder, endpoints, config):
    num_channels = len(config["image_set"]["channels"])
    everything = np.zeros(shape=(num_channels, len(crops), image_size, image_size, 3))
    features = np.zeros(shape=(num_channels, len(crops), num_features))
    for i, crop in enumerate(crops):
        crop = skimage.transform.resize(crop, (image_size, image_size), mode='reflect')
        crop = 2 * (crop - 0.5)
        for j in range(num_channels):
            channel = crop[:,:,j]
            channel = channel[:,:,np.newaxis]
            channel = np.tile(channel,(1,1,3))
            everything[j,i,:,:,:] = channel
    for i in range(num_channels):
        features[i,:,:] = sess.run(endpoints['PreLogitsFlatten'], feed_dict={placeholder:everything[i,:,:,:,:]})
    return features

def profile(config, dset):
    num_channels = len(config["image_set"]["channels"])
    url = config["profiling"]["url"]
    checkpoint = config["profiling"]["checkpoint"]
    if not os.path.isfile(checkpoint):
        dataset_utils.download_and_uncompress_tarball(url, os.path.dirname(checkpoint))
    
    configuration = tf.ConfigProto()
    configuration.gpu_options.allow_growth = True
    configuration.gpu_options.visible_device_list = config["profiling"]["gpu"]
    
    num_classes = dset.numberOfClasses()
    input_vars = learning.training.input_graph(config)
    images = input_vars["labeled_crops"][0]
    labels = tf.one_hot(input_vars["labeled_crops"][1], num_classes)
    
    placeholder = tf.placeholder(tf.float32, shape=(None, image_size, image_size, 3))
    
    with slim.arg_scope(inception.inception_resnet_v2_arg_scope()):
        _, endpoints = inception.inception_resnet_v2(placeholder, num_classes=1001, is_training=False)
    init_fn = slim.assign_from_checkpoint_fn(
        checkpoint,
        slim.get_model_variables())
    
    sess = tf.Session(config=configuration)
    init_fn(sess)
    
    def predict(key, image_array, meta):
        existing = glob.glob(os.path.join(config["profiling"]["out_dir"], str(meta["Metadata_Plate"])+"_"+meta["Metadata_Well"]+"_"+str(meta["Metadata_Site"])+"_*"+".csv"))
        if len(existing) > 0:
            print("Already done:", str(key))
            return
        batch_size = config["training"]["minibatch"]
        image_key, image_names = dset.getImagePaths(meta)
        locations = [ learning.cropping.getLocations(image_key, config, randomize=False) ]
        if len(locations[0]) == 0:
            print("Empty locations set:", str(key))
            return
        pads = batch_size - len(locations[0]) % batch_size
        pad_data = pandas.DataFrame(columns=locations[0].columns, data=np.zeros(shape=(pads, 2), dtype=np.int32))
        locations[0] = pandas.concat((locations[0], pad_data))   
        labels_data = [ meta[config["training"]["label_field"]] ]
        boxes, box_ind, labels_data = learning.cropping.prepareBoxes(locations, labels_data, config)
        images_data = np.reshape(image_array, input_vars["shapes"]["batch"])

        sess.run(input_vars["enqueue_op"], {
                        input_vars["image_ph"]:images_data,
                        input_vars["boxes_ph"]:boxes,
                        input_vars["box_ind_ph"]:box_ind,
                        input_vars["labels_ph"]:labels_data
        })
        items = sess.run(input_vars["queue"].size())
        data = np.zeros(shape=(num_channels, len(locations[0]), num_features))
        b = 0
        while items >= batch_size:
            crops = sess.run(images)
            time = tic()
            feats = get_features(crops, sess, placeholder, endpoints, config)
            toc("get_features", time)
            data[:,b*batch_size:(b+1)*batch_size,:] = feats
            b += 1
            items = sess.run(input_vars["queue"].size())
        for i in range(num_channels):
            filename = os.path.join(config["profiling"]["out_dir"], str(meta["Metadata_Plate"])+"_"+meta["Metadata_Well"]+"_"+str(meta["Metadata_Site"])+"_"+config["image_set"]["channels"][i]+".csv")
            csv = pandas.DataFrame(data=data[i,:-pads,:])
            csv.to_csv(filename, index=False)
            print("Wrote ", filename)
        print(" *",image_key, "done")
        
    dset.scan(predict, frame="all")
    print("Profiling: done")
