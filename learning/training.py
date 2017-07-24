import threading
from tqdm import tqdm

import numpy as np
import tensorflow as tf
import keras

import dataset.utils
import learning.cropping
import learning.models


#################################################
## INPUT GRAPH DEFINITION
#################################################

def input_graph(config, labels=True):
    # Identify number of channels
    mask_objects = config["image_set"]["mask_objects"]
    if mask_objects:
        img_channels = len(config["image_set"]["channels"]) + 1
    else:
        img_channels = len(config["image_set"]["channels"])
    crop_channels = len(config["image_set"]["channels"])

    # Identify image and box sizes
    box_size = config["sampling"]["box_size"]
    img_width = config["image_set"]["width"]
    img_height = config["image_set"]["height"]

    # Data shapes
    crop_shape = [(box_size, box_size, crop_channels), ()]
    imgs_shape = [None, img_height, img_width, img_channels]
    batch_shape = (-1, img_height, img_width, img_channels)

    # Inputs to the load data queue
    image_ph = tf.placeholder(tf.float32, shape=imgs_shape, name="raw_images")
    boxes_ph = tf.placeholder(tf.float32, shape=[None, 4], name="cell_boxes")
    box_ind_ph = tf.placeholder(tf.int32, shape=[None], name="box_indicators")
    labels_ph = tf.placeholder(tf.int32, shape=[None], name="image_labels")
    mask_ind_ph = tf.placeholder(tf.int32, shape=[None], name="mask_indicators")

    with tf.device("/cpu:0"):
        # Outputs and queue of the cropping graph
        crop_op = learning.cropping.crop(
            image_ph, 
            boxes_ph, 
            box_ind_ph, 
            mask_ind_ph,
            box_size,
            mask_objects
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
        "mask_ind_ph":mask_ind_ph,
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
                boxes, box_ind, labels, masks = learning.cropping.prepareBoxes(batch, config)
                sess.run(input_vars["enqueue_op"], {
                        input_vars["image_ph"]:images, 
                        input_vars["boxes_ph"]:boxes, 
                        input_vars["box_ind_ph"]:box_ind, 
                        input_vars["labels_ph"]:labels,
                        input_vars["mask_ind_ph"]:masks
                })
            except:
                #import traceback
                #traceback.print_exc()
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
    configuration = tf.ConfigProto()
    configuration.gpu_options.allow_growth = True
    configuration.gpu_options.visible_device_list = "0"
    session = tf.Session(config = configuration)
    keras.backend.set_session(session)

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
    merged_summary = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(config["training"]["output"], session.graph)
    '''
    with tf.variable_scope("trainer"):
        tf.summary.histogram("labels", tf.argmax(label_batch, axis=1))
        train_ops, summary_writer = learning.models.create_trainer(network, label_batch, sess, config)
    sess.run(tf.global_variables_initializer())
    '''

    # Start data threads
    coord, queue_threads = training_queues(session, dset, config, input_vars, train_vars)
    tf.train.start_queue_runners(coord=coord, sess=session)

    def batch_generator(sess, global_step=0):
        while True:
            if coord.should_stop():
                break
            im, lb, ms = sess.run([image_batch, label_batch, merged_summary])
            global_step += 1
            if global_step % 10 == 0: 
                summary_writer.add_summary(ms, global_step)

            yield (im, lb)

    # keras-resnet model
    output_file = config["training"]["output"] + "/checkpoint_{epoch:04d}.hdf5"
    callback_model_checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=output_file,
        save_weights_only=True,
        save_best_only=False
    )
    csv_output = config["training"]["output"] + "/log.csv"
    callback_csv = keras.callbacks.CSVLogger(filename=csv_output)
    callbacks = [callback_model_checkpoint, callback_csv]

    input_shape = (
        config["sampling"]["box_size"],      # height 
        config["sampling"]["box_size"],      # width
        len(config["image_set"]["channels"]) # channels
    )
    model = learning.models.create_keras_resnet(input_shape, num_classes)
    optimizer = keras.optimizers.Adam(lr=config["training"]["learning_rate"])
    model.compile(optimizer, "categorical_crossentropy", ["accuracy"])

    epochs = 100
    steps = config["training"]["iterations"] / epochs
    model.fit_generator(
        generator=batch_generator(session),
        steps_per_epoch=steps,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1 
    )

    # Close session and stop threads
    print("Complete! Closing session.", end="", flush=True)
    coord.request_stop()
    session.run(input_vars["queue"].close(cancel_pending_enqueues=True))
    session.run(train_vars["queue"].close(cancel_pending_enqueues=True))
    coord.join(queue_threads)
    session.close()
    print(" All set.")

