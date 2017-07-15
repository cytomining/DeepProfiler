import threading
from tqdm import tqdm

import numpy as np
import tensorflow as tf

import dataset.utils
import learning.cropping
import learning.models


#################################################
## INPUT GRAPH DEFINITION
#################################################

def input_graph(config, labels=True):
    # Data shapes
    crop_shape = [
                   (
                     config["sampling"]["box_size"], 
                     config["sampling"]["box_size"], 
                     len(config["image_set"]["channels"])
                   ), 
                   ()
                 ]
    imgs_shape = [
                   None, 
                   config["image_set"]["height"], 
                   config["image_set"]["width"], 
                   len(config["image_set"]["channels"])
                 ]
    batch_shape = (
                    -1, #config["sampling"]["images"],
                    config["image_set"]["height"],
                    config["image_set"]["width"],
                    len(config["image_set"]["channels"])
                  )

    # Inputs to the load data queue
    image_ph = tf.placeholder(tf.float32, shape=imgs_shape, name="raw_images")
    boxes_ph = tf.placeholder(tf.float32, shape=[None, 4], name="cell_boxes")
    box_ind_ph = tf.placeholder(tf.int32, shape=[None], name="box_indicators")
    labels_ph = tf.placeholder(tf.int32, shape=[None], name="image_labels")

    with tf.device("/cpu:0"):
        # Outputs and queue of the cropping graph
        crop_op = learning.cropping.crop(
            image_ph, 
            boxes_ph, 
            box_ind_ph, 
            config["sampling"]["box_size"]
        )
        daug_queue = tf.FIFOQueue(
            config["queueing"]["fifo_queue_size"], 
            [tf.float32, tf.int32], 
            shapes=crop_shape
        )
        daug_enqueue_op = daug_queue.enqueue_many([crop_op, labels_ph])
        labeled_crops = daug_queue.dequeue_many(config["training"]["minibatch"])


    variables = {
        "image_ph":image_ph,
        "boxes_ph":boxes_ph,
        "box_ind_ph":box_ind_ph,
        "labels_ph":labels_ph,
        "labeled_crops":labeled_crops,
        "shapes": {
            "crops": crop_shape,
            "images": imgs_shape,
            "batch": batch_shape
        },
        "queue":daug_queue,
        "enqueue_op":daug_enqueue_op
    }
    return variables

#################################################
## AUGMENTATION GRAPH DEFINITION
#################################################

def augmentation_graph(config, input_vars, num_classes):

    # Outputs and queue of the data augmentation graph
    train_queue = tf.RandomShuffleQueue(
        config["queueing"]["random_queue_size"], 
        config["queueing"]["min_size"], 
        [tf.float32, tf.int32], 
        shapes=input_vars["shapes"]["crops"]
    )
    augmented_op = learning.cropping.aument_multiple(
        input_vars["labeled_crops"][0], 
        config["queueing"]["augmentation_workers"]
    )
    train_enqueue_op = train_queue.enqueue_many([
        augmented_op,
        input_vars["labeled_crops"][1]
    ])
    train_inputs = train_queue.dequeue() #_many(config["training"]["minibatch"]) 


    train_vars = {
        "image_batch":train_inputs[0],
        "label_batch":tf.one_hot(train_inputs[1], num_classes),
        "queue":train_queue,
        "enqueue_op":train_enqueue_op
    }

    return train_vars

#################################################
## START TRAINING QUEUES
#################################################

def training_queues(sess, dset, config, input_vars, train_vars):
    coord = tf.train.Coordinator()

    # Enqueueing threads for raw images
    def data_enqueue_thread():
        while not coord.should_stop():
            try:
                # Load images and cell boxes
                batch = learning.cropping.loadBatch(dset, config)
                images = np.reshape(batch["images"], input_vars["shapes"]["batch"])
                boxes, box_ind, labels = learning.cropping.prepareBoxes(batch["locations"], batch["labels"], config)
                sess.run(input_vars["enqueue_op"], {
                        input_vars["image_ph"]:images, 
                        input_vars["boxes_ph"]:boxes, 
                        input_vars["box_ind_ph"]:box_ind, 
                        input_vars["labels_ph"]:labels
                })
            except:
                print(".", end="", flush=True)
                return

    load_threads = []
    for i in range(config["queueing"]["cropping_workers"]):
        lt = threading.Thread(target=data_enqueue_thread)
        load_threads.append(lt)
        lt.isDaemon()
        lt.start()

    # Enqueueing threads for augmented crops
    qr = tf.train.QueueRunner(
           train_vars["queue"], 
           [ train_vars["enqueue_op"] ] * config["queueing"]["augmentation_workers"]
    )
    enqueue_threads = qr.create_threads(sess, coord=coord, start=True)

    return coord, load_threads + enqueue_threads


#################################################
## MAIN TRAINING ROUTINE
#################################################

def learn_model(config, dset):

    # Start session
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.per_process_gpu_memory_fraction = config["queueing"]["gpu_mem_fraction"]
    sess = tf.Session(config=gpu_config)

    # Define input data batches
    with tf.variable_scope("train_inputs"):
        num_classes = dset.numberOfClasses()
        input_vars = input_graph(config)
        train_vars = augmentation_graph(config, input_vars, num_classes)
        image_batch, label_batch = tf.train.shuffle_batch(
            [train_vars["image_batch"], train_vars["label_batch"]],
            batch_size=config["training"]["minibatch"],
            num_threads=config["queueing"]["augmentation_workers"],
            capacity=config["queueing"]["random_queue_size"],
            min_after_dequeue=config["queueing"]["min_size"]
        )
        for i in range(len(config["image_set"]["channels"])):
            tf.summary.image("channel-" + str(i + 1), 
                             tf.expand_dims(image_batch[:,:,:,i], -1), 
                             max_outputs=3, 
                             collections=None
                            )


    # Graph of learning model
    network = learning.models.create_resnet(image_batch, num_classes)
    with tf.variable_scope("trainer"):
        train_ops, summary_writer = learning.models.create_trainer(network, label_batch, sess, config)
    sess.run(tf.global_variables_initializer())

    # Model saver
    convnet_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="convnet")
    saver = tf.train.Saver(var_list=convnet_vars)
    output_file = config["training"]["output"] + "/model/weights.ckpt"

    # Start data threads
    coord, queue_threads = training_queues(sess, dset, config, input_vars, train_vars)
    tf.train.start_queue_runners(coord=coord, sess=sess)

    # Main training loop
    for i in tqdm(range(config["training"]["iterations"]), desc="Training"):
        if coord.should_stop():
            break
        results = sess.run(train_ops)
        if i % 100 == 0:
            summary_writer.add_summary(results[-1], i)
        if i % 10000 == 0:
            save_path = saver.save(sess, output_file, global_step=i)

    # Close session and stop threads
    print("Complete! Closing session.", end="", flush=True)
    coord.request_stop()
    sess.run(input_vars["queue"].close(cancel_pending_enqueues=True))
    sess.run(train_vars["queue"].close(cancel_pending_enqueues=True))
    coord.join(queue_threads)
    sess.close()
    print(" All set.")

