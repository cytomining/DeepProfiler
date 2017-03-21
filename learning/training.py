import threading
import time
from tqdm import tqdm

import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline

import dataset.utils
import learning.cropping
import learning.models


#################################################
## QUEUES FOR DATA LOADING
#################################################

def start_data_queues(config, dset, sess, coord):
    # Data shapes
    crop_shape = [(config["sampling"]["box_size"], config["sampling"]["box_size"], len(config["image_set"]["channels"])), ()]
    imgs_shape = [None, config["image_set"]["height"], config["image_set"]["width"], len(config["image_set"]["channels"])]
    batch_shape = (config["sampling"]["images"],config["image_set"]["height"],config["image_set"]["width"],len(config["image_set"]["channels"]))

    # Inputs to the load data queue
    image_ph = tf.placeholder(tf.float32, shape=imgs_shape, name="raw_images")
    boxes_ph = tf.placeholder(tf.float32, shape=[None, 4], name="cell_boxes")
    box_ind_ph = tf.placeholder(tf.int32, shape=[None], name="box_indicators")
    labels_ph = tf.placeholder(tf.int32, shape=[None], name="image_labels")

    with tf.device("/cpu:0"):
        # Outputs and queue of the cropping graph
        crop_op = learning.cropping.crop(image_ph, boxes_ph, box_ind_ph, config["sampling"]["box_size"])
        daug_queue = tf.FIFOQueue(
            config["queueing"]["fifo_queue_size"], 
            [tf.float32, tf.int32], 
            shapes=crop_shape
        )
        daug_enqueue_op = daug_queue.enqueue_many([crop_op, labels_ph])
        labeled_crops = daug_queue.dequeue_many(config["training"]["minibatch"])

        # Outputs and queue of the data augmentation graph
        train_queue = tf.RandomShuffleQueue(
            config["queueing"]["random_queue_size"], 
            config["queueing"]["min_size"], 
            [tf.float32, tf.int32], 
            shapes=crop_shape
        )
        augmented_op = learning.cropping.aument_multiple(labeled_crops[0], config["queueing"]["augmentation_workers"])
        train_enqueue_op = train_queue.enqueue_many([augmented_op, labeled_crops[1]])
        train_inputs = train_queue.dequeue() 

        # Enqueueing threads for raw images
        def data_enqueue_thread():
            while not coord.should_stop():
                # Load images and cell boxes
                batch = learning.cropping.loadBatch(dset, config)
                images = np.reshape(batch["images"], batch_shape)
                boxes, box_ind, labels = learning.cropping.prepareBoxes(batch["locations"], batch["labels"], config)
                sess.run(daug_enqueue_op, {image_ph:images, boxes_ph:boxes, box_ind_ph:box_ind, labels_ph:labels})

        load_threads = []
        for i in range(config["queueing"]["cropping_workers"]):
            lt = threading.Thread(target=data_enqueue_thread)
            load_threads.append(lt)
            lt.isDaemon()
            lt.start()

        # Enqueueing threads for augmented crops
        qr = tf.train.QueueRunner(train_queue, [train_enqueue_op]*config["queueing"]["augmentation_workers"])
        enqueue_threads = qr.create_threads(sess, coord=coord, start=True)

    return train_inputs, load_threads + enqueue_threads


#################################################
## MAIN TRAINING ROUTINE
#################################################

def learn_model(config, dset):

    # Start session
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.per_process_gpu_memory_fraction = 0.5
    sess = tf.Session(config=gpu_config)
    coord = tf.train.Coordinator()
    train_inputs, queue_threads = start_data_queues(config, dset, sess, coord)

    # Define data batches
    num_classes = dset.numberOfClasses()
    image_batch, label_batch = tf.train.shuffle_batch(
        [train_inputs[0], tf.one_hot(train_inputs[1], num_classes)],
        batch_size=config["training"]["minibatch"],
        num_threads=config["queueing"]["augmentation_workers"],
        capacity=config["queueing"]["random_queue_size"],
        min_after_dequeue=config["queueing"]["min_size"]
    )

    # Learning model
    box_shape = [None, config["sampling"]["box_size"], config["sampling"]["box_size"], len(config["image_set"]["channels"])]
    network = learning.models.create_vgg(image_batch, num_classes)
    train_ops, summary_writer = learning.models.create_trainer(network, label_batch, sess, config)

    # Main training loop
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    sess.run(tf.global_variables_initializer())
    for i in tqdm(range(config["training"]["iterations"]), desc="Training"):
        if coord.should_stop():
            break
        results = sess.run(train_ops)
        summary_writer.add_summary(results[-1], i)
    
    print("Complete. Closing session.")
    coord.request_stop()
    coord.join(queue_threads)
    sess.close()

