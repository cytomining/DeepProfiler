import gc
import threading
import time

import numpy as np
import skimage.exposure
import tensorflow as tf

import deepprofiler.dataset.utils
import deepprofiler.imaging.augmentations
import deepprofiler.imaging.boxes


# Eager crop+resize operation. Takes raw image tensors and bounding boxes and
# returns normalized crops. Works in TensorFlow 2 eager mode (no graph/session).
def crop_graph(images, boxes, box_ind, mask_ind, box_size, mask_boxes=False, export_masks=False):
    crop_size = tf.constant([box_size, box_size], name="crop_size")
    images = tf.convert_to_tensor(images, dtype=tf.float32)
    boxes = tf.convert_to_tensor(boxes, dtype=tf.float32)
    box_ind = tf.convert_to_tensor(box_ind, dtype=tf.int32)
    crops = tf.image.crop_and_resize(images, boxes, box_ind, crop_size)
    if export_masks or mask_boxes:
        mask_ind = tf.convert_to_tensor(mask_ind, dtype=tf.int32)
        mask_ind = tf.expand_dims(tf.expand_dims(mask_ind, -1), -1)
        mask_values = tf.ones_like(crops[:, :, :, -1], dtype=tf.float32) * tf.cast(mask_ind, dtype=tf.float32)
        masks = tf.cast(tf.equal(crops[:, :, :, -1], mask_values), dtype=tf.float32)
    if mask_boxes:
        crops = crops[:, :, :, 0:-1] * tf.expand_dims(masks, -1)

    mini = tf.math.reduce_min(crops, axis=[1, 2], keepdims=True)
    maxi = tf.math.reduce_max(crops, axis=[1, 2], keepdims=True)
    crops = (crops - mini) / (maxi - mini + tf.keras.backend.epsilon())

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
    output = np.reshape(crop, (crop.shape[0], crop.shape[0], -1), order="F").astype(np.float64)

    if last_channel < 0:
        # Keep all channels
        output = output[:, :, :]
    elif last_channel > 0 and last_channel <= output.shape[2]:
        # Drop last N channels
        output = output[:, :, 0:last_channel]
    elif last_channel == 0:
        # Use last channel as a binary mask
        output = output[:, :, 0:-1] * (output[:, :, -1:] / 255.)

    return output / 255.


# TODO: implement abstract crop generator
class CropGenerator(object):

    def __init__(self, config, dset):
        self.config = config
        self.dset = dset

    #################################################
    ## INPUT CONFIGURATION
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
        crop_shape = [(box_size, box_size, crop_channels)] + [()] * num_targets
        imgs_shape = [None, img_height, img_width, img_channels]
        batch_shape = (-1, img_height, img_width, img_channels)

        # No placeholders/graph in eager mode: just record the parameters that the
        # eager cropping operation and the data pool allocation need.
        self.export_masks = export_masks
        self.mask_objects = mask_objects
        self.box_size = box_size
        self.num_targets = num_targets
        self.input_variables = {
            "box_size": box_size,
            "num_targets": num_targets,
            "mask_objects": mask_objects,
            "export_masks": export_masks,
            "shapes": {
                "crops": crop_shape,
                "images": imgs_shape,
                "batch": batch_shape,
            },
        }

    def crop_batch(self, batch):
        # Build bounding boxes and labels, then crop eagerly.
        boxes, box_ind, targets, masks = deepprofiler.imaging.boxes.prepare_boxes(batch, self.config)
        images = np.reshape(batch["images"], self.input_variables["shapes"]["batch"])
        crops = crop_graph(
            images, boxes, box_ind, masks, self.box_size, self.mask_objects, self.export_masks
        ).numpy()
        labels = []
        for i in range(len(targets)):
            depth = self.dset.targets[i].shape[1]
            labels.append(tf.one_hot(targets[i], depth).numpy())
        return crops, labels

    #################################################
    ## START TRAINING LOADERS
    #################################################

    def training_queues(self):
        coord = tf.train.Coordinator()
        lock = threading.Lock()
        self.exception_occurred = False

        # Enqueueing threads for raw images
        def data_loading_thread():
            while not coord.should_stop():
                try:
                    # Load images and cell boxes
                    batch = self.dset.get_train_batch(lock)
                    if len(batch["images"]) == 0:
                        continue
                    crops, labels = self.crop_batch(batch)

                    # Find block in the pool to store data
                    lock.acquire()
                    first = self.pool_pointer
                    records = crops.shape[0]

                    if self.pool_pointer + records < self.image_pool.shape[0]:
                        last = self.pool_pointer + records
                        self.pool_pointer += records
                    else:
                        last = self.image_pool.shape[0]
                        records = last - first
                        self.pool_pointer = 0
                        self.ready_to_sample = True

                    self.dset.cache_records += records

                    # Replace block in the pool
                    self.image_pool[first:last, ...] = crops[0:records, ...]
                    for k in range(len(labels)):
                        self.label_pool[k][first:last, :] = labels[k][0:records, :]
                    lock.release()

                except Exception:
                    import traceback
                    traceback.print_exc()
                    print(".", end="", flush=True)
                    self.exception_occurred = True
                    return

        load_threads = []
        for i in range(self.config["train"]["sampling"]["workers"]):
            lt = threading.Thread(target=data_loading_thread)
            load_threads.append(lt)
            lt.daemon = True
            lt.start()

        return coord, load_threads

    def start(self, session=None):
        # Define input data parameters and pools
        self.build_input_graph()

        self.image_pool = np.zeros(
            [self.config["train"]["sampling"]["cache_size"]] + list(self.input_variables["shapes"]["crops"][0])
        )
        self.label_pool = [
            np.zeros([self.config["train"]["sampling"]["cache_size"], t.shape[1]]) for t in self.dset.targets
        ]
        self.pool_pointer = 0
        self.ready_to_sample = False
        print("Waiting for data", self.image_pool.shape, [l.shape for l in self.label_pool])

        # Start data threads
        self.coord, self.queue_threads = self.training_queues()

    def sample_batch(self, pool_index):
        while not self.ready_to_sample:
            time.sleep(2)
        np.random.shuffle(pool_index)
        idx = pool_index[0:self.config["train"]["model"]["params"]["batch_size"]]
        # TODO: make outputs for all targets
        data = [self.image_pool[idx, ...], self.label_pool[0][idx, :], 0]
        return data

    def generate(self, session=None, global_step=0):
        pool_index = np.arange(self.image_pool.shape[0])
        while True:
            if self.coord.should_stop():
                break
            data = self.sample_batch(pool_index)
            # Indices of data => [0] images, [1:-1] targets, [-1] summary
            global_step += 1
            yield data[0], data[1:-1]

    def generator(self, session=None, global_step=0):
        # Yields (crops, labels) tuples ready for Keras model.fit
        pool_index = np.arange(self.image_pool.shape[0])
        while True:
            if self.coord.should_stop():
                break
            data = self.sample_batch(pool_index)
            global_step += 1
            yield data[0], data[1]

    def stop(self, session=None):
        self.coord.request_stop()
        self.coord.join(self.queue_threads)
        gc.collect()


#######################################################
## SUB CLASS TO GENERATE ALL CROPS IN A SINGLE IMAGE
#######################################################
# Useful for validation, predictions and profiling.
# Important differences to the above class:
# * No randomization is performed for crop generation
# * Only one image is loaded at a time and all crops in
#   that image are created in a single batch.
# * The generate method yields crops for a single image
# * The generator needs to be restarted for each image.
#########################################################

class SingleImageCropGenerator(CropGenerator):

    def start(self, session=None):
        self.config["train"]["model"]["params"]["batch_size"] = self.config["train"]["validation"]["batch_size"]
        self.build_input_graph()

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

        boxes, box_ind, targets, mask_ind = deepprofiler.imaging.boxes.prepare_boxes(batch, self.config)
        images = image_array[np.newaxis, ...]

        # check that all boxes overlap the image
        ymins = boxes[:, [0, 2]].min(axis=1)
        ymaxs = boxes[:, [0, 2]].max(axis=1)
        xmins = boxes[:, [1, 3]].min(axis=1)
        xmaxs = boxes[:, [1, 3]].max(axis=1)
        if (np.any(ymins > 1) or np.any(xmins > 1) or
                np.any(ymaxs < 0) or np.any(ymaxs < 0)):
            print("WARNING: Some cell boxes are entirely outside the image")

        crops = crop_graph(
            images, boxes, box_ind, mask_ind, self.box_size, self.mask_objects, self.export_masks
        ).numpy()

        self.image_pool = crops
        num_classes = self.dset.targets[0].shape[1]
        self.label_pool = tf.keras.utils.to_categorical(targets[0], num_classes=num_classes)

        return batch["locations"][0]

    def generate(self, session=None, global_step=0):
        yield [self.image_pool, self.label_pool]
