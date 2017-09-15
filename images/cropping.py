import threading

import numpy as np
import tensorflow as tf

import images.boxes
import learning.models

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

class CropGenerator(object):

    def __init__(self, config):
        self.config = config
        self.label_sources = []

    #################################################
    ## INPUT GRAPH DEFINITION
    #################################################

    def build_input_graph(self):
        # Identify number of channels
        mask_objects = self.config["image_set"]["mask_objects"]
        if mask_objects:
            img_channels = len(self.config["image_set"]["channels"]) + 1
        else:
            img_channels = len(self.config["image_set"]["channels"])
        crop_channels = len(self.config["image_set"]["channels"])

        # Identify image and box sizes
        box_size = self.config["sampling"]["box_size"]
        img_width = self.config["image_set"]["width"]
        img_height = self.config["image_set"]["height"]

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
            crop_op = crop(
                image_ph,
                boxes_ph,
                box_ind_ph,
                mask_ind_ph,
                box_size,
                mask_objects
            )
            daug_queue = tf.FIFOQueue(
                self.config["queueing"]["fifo_queue_size"],
                [tf.float32, tf.int32],
                shapes=crop_shape
            )
            daug_enqueue_op = daug_queue.enqueue_many([crop_op, labels_ph])
            labeled_crops = daug_queue.dequeue_many(self.config["training"]["minibatch"])


        self.input_variables = {
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

    #################################################
    ## AUGMENTATION GRAPH DEFINITION
    #################################################

    def build_augmentation_graph(self, num_classes):

        # Outputs and queue of the data augmentation graph
        train_queue = tf.RandomShuffleQueue(
            self.config["queueing"]["random_queue_size"],
            self.config["queueing"]["min_size"],
            [tf.float32, tf.int32],
            shapes=self.input_variables["shapes"]["crops"]
        )
        augmented_op = images.aumentations.aument_multiple(
            self.input_variables["labeled_crops"][0],
            self.config["queueing"]["augmentation_workers"]
        )
        train_enqueue_op = train_queue.enqueue_many([
            augmented_op,
            self.input_variables["labeled_crops"][1]
        ])
        train_inputs = train_queue.dequeue() #_many(config["training"]["minibatch"])

        self.train_variables = {
            "image_batch":train_inputs[0],
            "label_batch":tf.one_hot(train_inputs[1], num_classes),
            "queue":train_queue,
            "enqueue_op":train_enqueue_op
        }

    #################################################
    ## START TRAINING QUEUES
    #################################################

    def training_queues(self, sess, dset):
        coord = tf.train.Coordinator()

        # Enqueueing threads for raw images
        def data_enqueue_thread():
            while not coord.should_stop():
                try:
                    # Load images and cell boxes
                    batch = images.boxes.loadBatch(dset, self.config)
                    images = np.reshape(batch["images"], self.input_variables["shapes"]["batch"])
                    boxes, box_ind, labels, masks = images.boxes.prepareBoxes(batch, self.config)
                    sess.run(self.input_variables["enqueue_op"], {
                            self.input_variables["image_ph"]:images,
                            self.input_variables["boxes_ph"]:boxes,
                            self.input_variables["box_ind_ph"]:box_ind,
                            self.input_variables["labels_ph"]:labels,
                            self.input_variables["mask_ind_ph"]:masks
                    })
                except:
                    #import traceback
                    #traceback.print_exc()
                    print(".", end="", flush=True)
                    return

        load_threads = []
        for i in range(self.config["queueing"]["cropping_workers"]):
            lt = threading.Thread(target=data_enqueue_thread)
            load_threads.append(lt)
            lt.isDaemon()
            lt.start()

        # Enqueueing threads for augmented crops
        qr = tf.train.QueueRunner(
               self.train_variables["queue"],
               [ self.train_variables["enqueue_op"] ] * self.config["queueing"]["augmentation_workers"]
        )
        enqueue_threads = qr.create_threads(sess, coord=coord, start=True)

        return coord, load_threads + enqueue_threads

    def start(self, dset, session):
        # Define input data batches
        with tf.variable_scope("train_inputs"):
            num_classes = dset.numberOfClasses()
            self.build_input_graph()
            self.build_augmentation_graph(num_classes)
            self.image_batch, self.label_batch = tf.train.shuffle_batch(
                [self.train_variables["image_batch"], self.train_variables["label_batch"]],
                batch_size=self.config["training"]["minibatch"],
                num_threads=self.config["queueing"]["augmentation_workers"],
                capacity=self.config["queueing"]["random_queue_size"],
                min_after_dequeue=self.config["queueing"]["min_size"]
            )
            for i in range(len(self.config["image_set"]["channels"])):
                tf.summary.image("channel-" + str(i + 1),
                                 tf.expand_dims(self.image_batch[:, :, :, i], -1),
                                 max_outputs=3,
                                 collections=None
                                 )
        self.merged_summary = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter(self.config["training"]["output"], session.graph)
        '''
        with tf.variable_scope("trainer"):
            tf.summary.histogram("labels", tf.argmax(label_batch, axis=1))
            train_ops, summary_writer = learning.models.create_trainer(network, label_batch, sess, config)
        sess.run(tf.global_variables_initializer())
        '''

        # Start data threads
        self.coord, self.queue_threads = self.training_queues(session, dset)
        tf.train.start_queue_runners(coord=self.coord, sess=session)

    def generate(self, sess, global_step=0):
        while True:
            if self.coord.should_stop():
                break
            im, lb, ms = sess.run([self.image_batch, self.label_batch, self.merged_summary])
            global_step += 1
            if global_step % 10 == 0:
                self.summary_writer.add_summary(ms, global_step)

            yield (im, lb)

    def stop(self, session):
        self.coord.request_stop()
        session.run(self.input_variables["queue"].close(cancel_pending_enqueues=True))
        session.run(self.input_variables["queue"].close(cancel_pending_enqueues=True))
        self.coord.join(self.queue_threads)

