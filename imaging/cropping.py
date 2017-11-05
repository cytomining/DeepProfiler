import threading
import pandas as pd

import numpy as np
import tensorflow as tf

import imaging.boxes
import imaging.augmentations
import imaging.cropset

def crop_graph(image_ph, boxes_ph, box_ind_ph, mask_ind_ph, box_size, mask_boxes=False):
    with tf.variable_scope("cropping"):
        crop_size_ph = tf.constant([box_size, box_size], name="crop_size")
        crops = tf.image.crop_and_resize(image_ph, boxes_ph, box_ind_ph, crop_size_ph)
        if mask_boxes:
            mask_ind = tf.expand_dims(tf.expand_dims(mask_ind_ph, -1), -1)
            mask_values = tf.ones_like(crops[:,:,:,-1], dtype=tf.float32) * tf.cast(mask_ind, dtype=tf.float32)
            masks = tf.to_float( tf.equal(crops[:,:,:,-1], mask_values) )
            crops = crops[:,:,:,0:-1] * tf.expand_dims(masks, -1)
        #crops = tf.log( crops + 1 )
        max_intensities = tf.reduce_max( tf.reduce_max( crops, axis=1, keep_dims=True), axis=2, keep_dims=True) / 2.0
        crops = (crops - max_intensities) / max_intensities
    return crops

class CropGenerator(object):

    def __init__(self, config, dset):
        self.config = config
        self.dset = dset

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
        num_targets = len(self.dset.targets)
        crop_shape = [(box_size, box_size, crop_channels)] + [()]*num_targets
        imgs_shape = [None, img_height, img_width, img_channels]
        batch_shape = (-1, img_height, img_width, img_channels)

        # Inputs to the load data queue
        image_ph = tf.placeholder(tf.float32, shape=imgs_shape, name="raw_images")
        boxes_ph = tf.placeholder(tf.float32, shape=[None, 4], name="cell_boxes")
        box_ind_ph = tf.placeholder(tf.int32, shape=[None], name="box_indicators")
        mask_ind_ph = tf.placeholder(tf.int32, shape=[None], name="mask_indicators")
        targets_phs = {}
        for i in range(num_targets):
            tname = "target_" + str(i)
            tgt = self.dset.targets[i]
            targets_phs[tname] = tf.placeholder(tf.int32, shape=[None], name=tname)

        with tf.device("/cpu:0"):
            # Outputs and queue of the cropping graph
            crop_op = crop_graph(
                image_ph,
                boxes_ph,
                box_ind_ph,
                mask_ind_ph,
                box_size,
                mask_objects
            )
            daug_queue = tf.FIFOQueue(
                self.config["queueing"]["fifo_queue_size"],
                [tf.float32] + [tf.int32] * len(targets_phs),
                shapes=crop_shape
            )
            daug_enqueue_op = daug_queue.enqueue_many([crop_op] + [targets_phs[t] for t in targets_phs.keys()])
            labeled_crops = daug_queue.dequeue_many(self.config["training"]["minibatch"])

        self.input_variables = {
            "image_ph":image_ph,
            "boxes_ph":boxes_ph,
            "box_ind_ph":box_ind_ph,
            "targets_phs":targets_phs,
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

    def build_augmentation_graph(self):
        num_targets = len(self.dset.targets)

        # Outputs and queue of the data augmentation graph
        train_queue = tf.RandomShuffleQueue(
            self.config["queueing"]["random_queue_size"],
            self.config["queueing"]["min_size"],
            [tf.float32] + [tf.int32] * num_targets,
            shapes=self.input_variables["shapes"]["crops"]
        )
        augmented_op = imaging.augmentations.aument_multiple(
            self.input_variables["labeled_crops"][0],
            self.config["queueing"]["augmentation_workers"]
        )
        train_enqueue_op = train_queue.enqueue_many(
            [augmented_op] +
            self.input_variables["labeled_crops"][1:]
        )
        train_inputs = train_queue.dequeue() #_many(config["training"]["minibatch"])

        self.train_variables = {
            "image_batch":train_inputs[0],
            "queue":train_queue,
            "enqueue_op":train_enqueue_op
        }

        for i in range(num_targets):
            tname = "target_" + str(i)
            tgt = self.dset.targets[i]
            self.train_variables[tname] = tf.one_hot(train_inputs[i+1], tgt.shape[1])

    #################################################
    ## START TRAINING QUEUES
    #################################################

    def training_queues(self, sess):
        coord = tf.train.Coordinator()

        # Enqueueing threads for raw images
        def data_enqueue_thread():
            while not coord.should_stop():
                try:
                    # Load images and cell boxes
                    batch = imaging.boxes.loadBatch(self.dset, self.config)
                    images = np.reshape(batch["images"], self.input_variables["shapes"]["batch"])
                    boxes, box_ind, targets, masks = imaging.boxes.prepareBoxes(batch, self.config)
                    feed_dict = {
                            self.input_variables["image_ph"]:images,
                            self.input_variables["boxes_ph"]:boxes,
                            self.input_variables["box_ind_ph"]:box_ind,
                            self.input_variables["mask_ind_ph"]:masks
                    }
                    for i in range(len(targets)):
                        tname = "target_" + str(i)
                        feed_dict[self.input_variables["targets_phs"][tname]] = targets[i]

                    sess.run(self.input_variables["enqueue_op"], feed_dict)
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

    def start(self, session):
        # Define input data batches
        with tf.variable_scope("train_inputs"):
            self.build_input_graph()
            self.build_augmentation_graph()
            targets = [self.train_variables[t] for t in self.train_variables.keys() if t.startswith("target_")]
            batch = tf.train.shuffle_batch(
                [self.train_variables["image_batch"]] + targets,
                batch_size=self.config["training"]["minibatch"],
                num_threads=self.config["queueing"]["augmentation_workers"],
                capacity=self.config["queueing"]["random_queue_size"],
                min_after_dequeue=self.config["queueing"]["min_size"]
            )
            self.image_batch = batch[0]
            self.target_batch = batch[1:]

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
        self.coord, self.queue_threads = self.training_queues(session)
        tf.train.start_queue_runners(coord=self.coord, sess=session)

    def generate(self, sess, global_step=0):
        while True:
            if self.coord.should_stop():
                break
            data = sess.run([self.image_batch] + self.target_batch + [self.merged_summary])
            # Indices of data => [0] images, [1:-1] targets, [-1] summary
            ms = data[-1]
            global_step += 1
            if global_step % 10 == 0:
                self.summary_writer.add_summary(ms, global_step)

            yield (data[0], data[1:-1])

    def stop(self, session):
        self.coord.request_stop()
        session.run(self.input_variables["queue"].close(cancel_pending_enqueues=True))
        session.run(self.train_variables["queue"].close(cancel_pending_enqueues=True))
        self.coord.join(self.queue_threads)


#######################################################
## SUB CLASS TO GENERATE ALL CROPS IN A SINGLE IMAGE
#######################################################
# Useful for validation, predictions and profiling.
# Important differences to the above class:
# * No randomization is performed for crop generation
# * A batch of crops is padded with zeros at the end
#   for avoiding mixing of crops from different images.
# * Only one queue is used for loading a single image
#   and creating all crops in that image
# * No need to stop threads.
# * The generate method yields crops for a single image
# * The generator needs to be restarted for each image.
#########################################################

class SingleImageCropGenerator(CropGenerator):

    def __init__(self, config, dset):
        super().__init__(config, dset)


    def start(self, session):
        # Define input data batches
        with tf.variable_scope("train_inputs"):
            self.build_input_graph()


    def prepare_image(self, session, image_array, meta, sample_first_crops=False):
        num_targets = len(self.dset.targets)
        self.batch_size = self.config["validation"]["minibatch"]
        image_key, image_names, outlines = self.dset.getImagePaths(meta)

        batch = {"images": [], "locations": [], "targets": [[]]}
        batch["images"].append(image_array)
        batch["locations"].append(imaging.boxes.getLocations(image_key, self.config, randomize=False))
        for i in range(num_targets):
            tgt = self.dset.targets[i]
            batch["targets"][0].append(tgt.get_values(meta))

        if sample_first_crops and self.batch_size < len(batch["locations"][0]):
            batch["locations"][0] = batch["locations"][0].head(self.batch_size)

        remaining = len(batch["locations"][0]) % self.batch_size
        if remaining > 0:
            pads = self.batch_size - remaining
        else:
            pads = 0

        padding = pd.DataFrame(columns=batch["locations"][0].columns, data=np.zeros(shape=(pads, 2), dtype=np.int32))
        batch["locations"][0] = pd.concat((batch["locations"][0], padding), ignore_index=True)


        boxes, box_ind, targets, mask_ind = imaging.boxes.prepareBoxes(batch, self.config)
        batch["images"] = np.reshape(image_array, self.input_variables["shapes"]["batch"])

        feed_dict = {
            self.input_variables["image_ph"]: batch["images"],
            self.input_variables["boxes_ph"]: boxes,
            self.input_variables["box_ind_ph"]: box_ind,
            self.input_variables["mask_ind_ph"]: mask_ind
        }
        for i in range(num_targets):
            tname = "target_" + str(i)
            feed_dict[self.input_variables["targets_phs"][tname]] = targets[i]

        session.run(self.input_variables["enqueue_op"], feed_dict)

        total_crops = len(batch["locations"][0])
        return total_crops, pads


    def generate(self, session, global_step=0):
        items = session.run(self.input_variables["queue"].size())
        while items >= self.batch_size:
            batch = session.run(self.input_variables["labeled_crops"])
            yield batch
            items = session.run(self.input_variables["queue"].size())


#######################################################
## SUB CLASS TO GENERATE SETS OF CROPS FOR SEQUENCE LEARNING
#######################################################

class SetCropGenerator(CropGenerator):

    def __init__(self, config, dset):
        super().__init__(config, dset)


    def start(self, session):
        super().start(session)

        self.batch_size = self.config["training"]["minibatch"]
        self.target_sizes = []
        targets = [t for t in self.train_variables.keys() if t.startswith("target_")]
        targets.sort()
        for t in targets:
            self.target_sizes.append(self.train_variables[t].shape[0])

        self.set_manager = imaging.cropset.CropSet(
                   self.config["image_set"]["crop_set_length"],
                   self.config["queueing"]["random_queue_size"], 
                   self.input_variables["shapes"]["crops"],
                   self.target_sizes[0]
        )


    def generate(self, sess, global_step=0):
        while True:
            if self.coord.should_stop():
                break
            data = sess.run([self.image_batch] + self.target_batch + [self.merged_summary])
            # Indices of data => [0] images, [1:-1] targets, [-1] summary
            self.set_manager.add_crops(data[0], data[1]) #TODO: support for multiple targets
            while not self.set_manager.ready:
                data = sess.run([self.image_batch] + self.target_batch + [self.merged_summary])
                self.set_manager.add_crops(data[0], data[1])

            global_step += 1
            # TODO: Enable use of summaries
            #ms = data[-1]
            #if global_step % 10 == 0:
            #    self.summary_writer.add_summary(ms, global_step)

            batch = self.set_manager.batch(self.batch_size)

            yield (batch[0], batch[1]) # TODO: support for multiple targets


class SingleImageCropSetGenerator(SingleImageCropGenerator):

    def __init__(self, config, dset):
        super().__init__(config, dset)


    def start(self, session):
        super().start(session)


    def generate(self, session, global_step=0):
        reps = self.config["image_set"]["crop_set_length"]
        for batch in supper().generate(session):
            batch[0] = batch[0][:,np.newaxis,:,:,:]
            batch[0] = np.tile(batch[0], (1,reps,1,1,1))
            yield batch
