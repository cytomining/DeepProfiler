import threading
import time

import learning.cropping
import numpy as np
import tensorflow as tf


#################################################
## AUXILIARY FUNCTIONS
#################################################

def checkConfig(config):
    required_fields = []
    # Image geometry
    required_fields += ["image_height", "image_width", "box_size", "channels"]
    # Batch size
    required_fields += ["image_batch_size", "sample_cells", "minibatch_size"]
    # Queue size
    required_fields += ["fifo_queue_size", "random_queue_size"]
    # Number of workers
    required_fields += ["cropping_workers", "augmentation_workers"]
    # Learning
    required_fields += ["training_iterations"]
    # Field verification
    for field in required_fields:
        assert field in config.keys()
    return True

#################################################
## MAIN TRAINING ROUTINE
#################################################

def learnCNN(config, dataset):
    #checkConfig(config)
    # Inputs to the graph
    imgs_shape = [None, config["image_set"]["height"], config["image_set"]["width"], len(config["image_set"]["channels"])]
    image_ph = tf.placeholder(tf.float32, shape=imgs_shape, name="raw_images")
    boxes_ph = tf.placeholder(tf.float32, shape=[None, 4], name="cell_boxes")
    box_ind_ph = tf.placeholder(tf.int32, shape=[None], name="box_indicators")
    labels_ph = tf.placeholder(tf.int32, shape=[None], name="image_labels")

    # Outputs and queue of the cropping graph
    crop_op = learning.cropping.crop(image_ph, boxes_ph, box_ind_ph, config["sampling"]["box_size"])
    daug_queue = tf.FIFOQueue(config["queueing"]["fifo_queue_size"], [tf.float32, tf.int32])
    daug_enqueue_op = daug_queue.enqueue_many([crop_op, labels_ph])
    labeled_crops = daug_queue.dequeue()

    # Outputs and queue of the dataset augmentation graph
    augmented_op = learning.cropping.augment(labeled_crops[0])
    crop_shape = [(config["sampling"]["box_size"], config["sampling"]["box_size"], len(config["image_set"]["channels"])), ()]
    train_queue = tf.RandomShuffleQueue(config["queueing"]["random_queue_size"], config["sampling"]["locations"], [tf.float32, tf.int32], shapes=crop_shape)
    train_enqueue_op = train_queue.enqueue([augmented_op, labeled_crops[1]])
    train_inputs = train_queue.dequeue_many(config["training"]["minibatch"])

    # Start session
    sess = tf.Session()
    coord = tf.train.Coordinator()

    # Enqueuing thread for raw images
    batch_size = (config["sampling"]["images"],config["image_set"]["height"],config["image_set"]["width"],len(config["image_set"]["channels"]))
    def data_enqueue_thread():
        while not coord.should_stop():
            # Load images and cell boxes
            batch = learning.cropping.loadBatch(dataset, config["image_set"]["path"], config)
            images = np.reshape(batch["images"], batch_size)
            boxes, box_ind, labels = learning.cropping.prepareBoxes(batch["cells"], batch["labels"], config)
            sess.run(daug_enqueue_op, {image_ph:images, boxes_ph:boxes, box_ind_ph:box_ind, labels_ph:labels})
            print("Images enqueued",images.shape, boxes.shape, labels.shape, box_ind.shape)
    for _ in range(config["queueing"]["cropping_workers"]):
        threading.Thread(target=data_enqueue_thread).start()

    # Enqueuing thread for labeled crops
    def augm_enqueue_thread():
        while not coord.should_stop():
            sess.run(train_enqueue_op)
    for _ in range(config["queueing"]["augmentation_workers"]):
        threading.Thread(target=augm_enqueue_thread).start()

    # Main training loop
    for i in range(config["training"]["iterations"]):
        if coord.should_stop():
            break
        training_data = sess.run(train_inputs)
        print('Cells used for training',training_data[0].shape)
        time.sleep(0.5)
        
    coord.request_stop()

