import gc
import threading
import time

import numpy as np
import skimage.exposure
import tensorflow as tf

import deepprofiler.dataset.utils
import deepprofiler.imaging.augmentations
import deepprofiler.imaging.boxes

tf.compat.v1.disable_v2_behavior()
tf.config.run_functions_eagerly(False)


def crop_graph(image_ph, boxes_ph, box_ind_ph, mask_ind_ph, box_size, mask_boxes=False, export_masks=False):
    with tf.compat.v1.variable_scope("cropping"):
        crop_size_ph = tf.constant([box_size, box_size], name="crop_size")
        crops = tf.image.crop_and_resize(image_ph, boxes_ph, box_ind_ph, crop_size_ph)
        if export_masks or mask_boxes:
            mask_ind = tf.expand_dims(tf.expand_dims(mask_ind_ph, -1), -1)
            mask_values = tf.ones_like(crops[:, :, :, -1], dtype=tf.float32) * tf.cast(mask_ind, dtype=tf.float32)
            masks = tf.compat.v1.to_float(tf.equal(crops[:, :, :, -1], mask_values))
        if mask_boxes:
            crops = crops[:, :, :, 0:-1] * tf.expand_dims(masks, -1)

        #mean = tf.math.reduce_mean(crops, axis=[1, 2], keepdims=True)
        #std = tf.math.reduce_std(crops, axis=[1, 2], keepdims=True)
        #crops = (crops - mean)/std
        mini = tf.math.reduce_min(crops, axis=[1, 2], keepdims=True)
        maxi = tf.math.reduce_max(crops, axis=[1, 2], keepdims=True)
        crops = (crops - mini) / maxi

        if export_masks:
            crops = tf.concat((crops[:, :, :, 0:-1], tf.expand_dims(masks, axis=-1)), axis=3)

    return crops


def unfold_channels(crop, mode=0):
    # Expected input image shape: (h, w, c)
    # Output image shape: (h, w * c)
    # Pixels are rescaled to the [0,255] interval with 8 bits encoding
    unfolded = np.reshape(np.moveaxis(crop, mode, 0), (crop.shape[mode], -1), order='F')
    unfolded = np.floor( 
        skimage.exposure.rescale_intensity(unfolded, in_range="image", out_range="uint8")
    ).astype(np.uint8)
    return unfolded


def fold_channels(crop, last_channel=-1):
    # Expected input image shape: (h, w * c), with h = w
    # Output image shape: (h, w, c), with h = w
    output = np.reshape(crop, (crop.shape[0], crop.shape[0], -1), order="F").astype(np.float)

    if last_channel < 0:
        # Keep all channels
        output = output[:, :, :]
    elif last_channel > 0 and last_channel <= output.shape[2]:
        # Drop last N channels
        output = output[:, :, 0:last_channel]
    elif last_channel == 0:
        # Use last channel as a binary mask
        output = output[:, :, 0:-1] * (output[:, :, -1:] / 255.)

    #for i in range(output.shape[-1]):
    #    mean = np.mean(output[:, :, i])
    #    std = np.std(output[:, :, i])
    #    output[:, :, i] = (output[:, :, i] - mean) / std
    return output / 255.


# TODO: implement abstract crop generator
class CropGenerator(object):

    def __init__(self, config, dset): 
        self.config = config
        self.dset = dset

    #################################################
    ## INPUT GRAPH DEFINITION
    #################################################

    def build_input_graph(self, export_masks=False):
        # Identify number of channels
        mask_objects = self.config["dataset"]["locations"]["mask_objects"]
        if mask_objects:
            img_channels = len(self.config["dataset"]["images"]["channels"]) + 1
        else:
            img_channels = len(self.config["dataset"]["images"]["channels"])

        if export_masks:
            crop_channels = len(self.config["dataset"]["images"]["channels"]) + 1
            mask_objects = False
        else:
            crop_channels = len(self.config["dataset"]["images"]["channels"])

        # Identify image and box sizes
        box_size = self.config["dataset"]["locations"]["box_size"]
        img_width = self.config["dataset"]["images"]["width"]
        img_height = self.config["dataset"]["images"]["height"]

        # Data shapes
        num_targets = len(self.dset.targets)
        crop_shape = [(box_size, box_size, crop_channels)] + [()]*num_targets
        #imgs_shape = [None, img_height, img_width, img_channels]
        imgs_shape = [None, None, None, img_channels]
        batch_shape = (-1, img_height, img_width, img_channels)

        # Inputs to cropping graph
        image_ph = tf.compat.v1.placeholder(tf.float32, shape=imgs_shape, name="raw_images")
        boxes_ph = tf.compat.v1.placeholder(tf.float32, shape=[None, 4], name="cell_boxes")
        box_ind_ph = tf.compat.v1.placeholder(tf.int32, shape=[None], name="box_indicators")
        mask_ind_ph = tf.compat.v1.placeholder(tf.int32, shape=[None], name="mask_indicators")
        targets_phs = {}
        for i in range(num_targets):
            tname = "target_" + str(i)
            tgt = self.dset.targets[i]
            targets_phs[tname] = tf.compat.v1.placeholder(tf.int32, shape=[None], name=tname)

        # Outputs and cache of the cropping graph
        crop_op = crop_graph(
            image_ph,
            boxes_ph,
            box_ind_ph,
            mask_ind_ph,
            box_size,
            mask_objects,
            export_masks
        )
        labeled_crops = tf.tuple([crop_op] + [targets_phs[t] for t in targets_phs.keys()])

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
        }

        # Training variables
        self.train_variables = {
                "image_batch": self.input_variables["labeled_crops"][0],
                "target_0": tf.one_hot(
                    self.input_variables["labeled_crops"][1], 
                    self.dset.targets[0].shape[1]
                )
        }


    #################################################
    ## START TRAINING QUEUES
    #################################################

    def training_queues(self, sess):
        coord = tf.train.Coordinator()
        lock = threading.Lock()
        self.exception_occurred = False

        # Enqueueing threads for raw images
        # vvv Start of thread function vvv
        def data_loading_thread():
            while not coord.should_stop():
                try:
                    # Load images and cell boxes
                    batch = self.dset.get_train_batch(lock)
                    if len(batch["images"]) == 0: continue
                    images = np.reshape(batch["images"], self.input_variables["shapes"]["batch"])
                    boxes, box_ind, targets, masks = deepprofiler.imaging.boxes.prepare_boxes(batch, self.config)

                    feed_dict = {
                            self.input_variables["image_ph"]:images,
                            self.input_variables["boxes_ph"]:boxes,
                            self.input_variables["box_ind_ph"]:box_ind,
                            self.input_variables["mask_ind_ph"]:masks
                    }
                    for i in range(len(targets)):
                        tname = "target_" + str(i)
                        feed_dict[self.input_variables["targets_phs"][tname]] = targets[i]

                    output = sess.run(self.train_variables, feed_dict)

                    # Find block in the pool to store data
                    lock.acquire()
                    first = self.pool_pointer
                    records = output["image_batch"].shape[0]

                    if self.pool_pointer + records < self.image_pool.shape[0]:
                        last = self.pool_pointer + records
                        self.pool_pointer += records
                    else:
                        last = self.image_pool.shape[0]
                        records = last - first
                        self.pool_pointer = 0
                        self.ready_to_sample = True

                    self.dset.cache_records += records

                    # Replace block (TODO:order of targets and keys may be wrong if more than one target is used)
                    self.image_pool[first:last,...] = output["image_batch"][0:records,...]
                    k = 0
                    for t in output.keys():
                        if t.startswith("target_"): 
                            self.label_pool[k][first:last,:] = output[t][0:records,:]
                            k += 1
                    lock.release()

                except:
                    import traceback
                    traceback.print_exc()
                    print(".", end="", flush=True)
                    self.exception_occurred = True
                    return

        # ^^^ End of thread function ^^^
        
        load_threads = []
        for i in range(self.config["train"]["sampling"]["workers"]):
            lt = threading.Thread(target=data_loading_thread)
            load_threads.append(lt)
            lt.isDaemon()
            lt.start()

        return coord, load_threads

    def start(self, session):
        # Define input data batches
        with tf.compat.v1.variable_scope("train_inputs"):
            self.build_input_graph()
            targets = [self.train_variables[t] for t in self.train_variables.keys() if t.startswith("target_")]

            self.image_pool = np.zeros(
                    [self.config["train"]["sampling"]["cache_size"]] + list(self.input_variables["shapes"]["crops"][0])
            )
            self.label_pool = [np.zeros([self.config["train"]["sampling"]["cache_size"], t.shape[1]]) for t in targets]
            self.pool_pointer = 0
            self.ready_to_sample = False
            print("Waiting for data", self.image_pool.shape, [l.shape for l in self.label_pool])

        self.merged_summary = tf.compat.v1.summary.merge_all()
        self.summary_writer = tf.compat.v1.summary.FileWriter(self.config["paths"]["summaries"], session.graph)

        # Start data threads
        self.coord, self.queue_threads = self.training_queues(session)
        tf.compat.v1.train.start_queue_runners(coord=self.coord, sess=session)

    def sample_batch(self, pool_index):
        while not self.ready_to_sample:
            time.sleep(2)
        np.random.shuffle(pool_index) 
        idx = pool_index[0:self.config["train"]["model"]["params"]["batch_size"]]
        # TODO: make outputs for all targets
        data = [self.image_pool[idx,...], self.label_pool[0][idx,:], 0]
        return data

    def generate(self, sess, global_step=0):
        pool_index = np.arange(self.image_pool.shape[0])
        while True:
            if self.coord.should_stop():
                break
            data = self.sample_batch(pool_index)
            # Indices of data => [0] images, [1:-1] targets, [-1] summary

            global_step += 1
            yield data[0], data[1:-1]

    def stop(self, session):
        self.coord.request_stop()
        self.coord.join(self.queue_threads)
        session.close()
        gc.collect()


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

    def start(self, session):
        # Define input data batches
        with tf.compat.v1.variable_scope("train_inputs"):
            self.config["train"]["model"]["params"]["batch_size"] = self.config["train"]["validation"]["batch_size"]
            self.build_input_graph()
            # Align cells by rotating nuclei
            #self.angles = tf.placeholder(tf.float32, shape=[None], name="nuclei_angles")
            #rotated_imgs = tf.contrib.image.rotate(self.input_variables["labeled_crops"][0], self.angles, interpolation="BILINEAR")
            #self.aligned_labeled = [rotated_imgs, self.input_variables["labeled_crops"][1]]

    def prepare_image(self, session, image_array, meta, sample_first_crops=False):

        num_targets = len(self.dset.targets)
        self.batch_size = self.config["train"]["validation"]["batch_size"]
        image_key, image_names, outlines = self.dset.get_image_paths(meta)

        batch = {"images": [], "locations": [], "targets": [[] for i in range(num_targets)]}
        batch["images"].append(image_array)
        batch["locations"].append(deepprofiler.imaging.boxes.get_locations(image_key, self.config, random_sample=None))
        for i in range(num_targets):
            tgt = self.dset.targets[i]
            batch["targets"][i].append(tgt.get_values(meta))

        if sample_first_crops and self.batch_size < len(batch["locations"][0]):
            batch["locations"][0] = batch["locations"][0].head(self.batch_size)

        has_orientation = "Orientation" in batch["locations"][0].columns
        boxes, box_ind, targets, mask_ind = deepprofiler.imaging.boxes.prepare_boxes(batch, self.config)
        #batch["images"] = np.reshape(image_array, self.input_variables["shapes"]["batch"])
        batch["images"] = image_array[np.newaxis, ...]
        feed_dict = {
            self.input_variables["image_ph"]: batch["images"],
            self.input_variables["boxes_ph"]: boxes,
            self.input_variables["box_ind_ph"]: box_ind,
            self.input_variables["mask_ind_ph"]: mask_ind
        }

        for i in range(num_targets):
            tname = "target_" + str(i)
            feed_dict[self.input_variables["targets_phs"][tname]] = targets[i]

        total_crops = len(batch["locations"][0])

        #if has_orientation:
        #    # Align cells by rotating to the major axis of nuclei
        #    feed_dict[self.angles] = (batch["locations"][0]["Orientation"]*deepprofiler.dataset.utils.PI)/180.
        #    output = session.run(self.aligned_labeled, feed_dict)
        #else:
        output = session.run(self.input_variables["labeled_crops"], feed_dict)
        output = {"image_batch": output[0], "target_0": output[1]}

        self.image_pool = output["image_batch"]
        num_classes = self.dset.targets[0].shape[1]
        self.label_pool = tf.compat.v1.keras.utils.to_categorical(output["target_0"], num_classes=num_classes)

        return batch["locations"][0] 


    def generate(self, session, global_step=0):
        yield [self.image_pool, self.label_pool]
