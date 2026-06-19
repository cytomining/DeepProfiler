"""Single-cell crop extraction from full field-of-view microscopy images.

DeepProfiler operates on full FOV images. Cell locations (centroids) come from
an external segmentation tool such as CellProfiler and are stored in per-image
CSV files (see :mod:`deepprofiler.imaging.boxes`). This module uses those
centroids to cut fixed-size patches out of the full image and feed them to the
feature extraction model.

Cropping is implemented as a TF1 graph operation (``tf.image.crop_and_resize``)
so that it runs efficiently on GPU alongside the model. Each crop is
independently min-max normalised to [0, 1] within the graph.
"""

import numpy as np
import tensorflow as tf

import deepprofiler.dataset.utils
import deepprofiler.imaging.boxes

tf.compat.v1.disable_v2_behavior()
tf.config.run_functions_eagerly(False)


def crop_graph(image_ph, boxes_ph, box_ind_ph, mask_ind_ph, box_size, mask_boxes=False, export_masks=False):
    """Build the TF1 subgraph that extracts and normalises cell crops.

    Uses ``tf.image.crop_and_resize`` to cut a ``box_size × box_size`` patch
    around each cell centroid via bilinear interpolation. Crops that extend
    beyond the image boundary are zero-padded automatically.

    Each crop is normalised independently to [0, 1] using its own per-crop
    min and max, so absolute pixel intensities do not affect the embedding.

    Args:
        image_ph: Float32 placeholder of shape ``(batch, H, W, C[+1])``.
            The optional extra channel holds object masks when
            ``mask_objects`` is enabled in config.
        boxes_ph: Float32 placeholder of shape ``(num_crops, 4)`` with
            normalised ``[y1, x1, y2, x2]`` coordinates in ``[0, 1]``.
        box_ind_ph: Int32 placeholder of shape ``(num_crops,)`` mapping each
            box to its image in the batch.
        mask_ind_ph: Int32 placeholder of shape ``(num_crops,)`` with the
            object mask label for each cell. Only used when masking is active.
        box_size: Integer side length of the output crops in pixels.
        mask_boxes: If True, multiply each crop by a binary mask derived from
            the last channel, zeroing out pixels that don't belong to the
            target cell. The mask channel is then dropped from the output.
        export_masks: If True, append the binary mask as the final output
            channel instead of dropping it.

    Returns:
        Float32 tensor of shape ``(num_crops, box_size, box_size, C)``
        with values normalised to [0, 1].
    """
    with tf.compat.v1.variable_scope("cropping"):
        crop_size_ph = tf.constant([box_size, box_size], name="crop_size")
        crops = tf.image.crop_and_resize(image_ph, boxes_ph, box_ind_ph, crop_size_ph)
        if export_masks or mask_boxes:
            mask_ind = tf.expand_dims(tf.expand_dims(mask_ind_ph, -1), -1)
            mask_values = tf.ones_like(crops[:, :, :, -1], dtype=tf.float32) * tf.cast(mask_ind, dtype=tf.float32)
            masks = tf.compat.v1.to_float(tf.equal(crops[:, :, :, -1], mask_values))
        if mask_boxes:
            crops = crops[:, :, :, 0:-1] * tf.expand_dims(masks, -1)

        mini = tf.math.reduce_min(crops, axis=[1, 2], keepdims=True)
        maxi = tf.math.reduce_max(crops, axis=[1, 2], keepdims=True)
        crops = (crops - mini) / (maxi - mini + tf.keras.backend.epsilon())

        if export_masks:
            crops = tf.concat((crops[:, :, :, 0:-1], tf.expand_dims(masks, axis=-1)), axis=3)

    return crops


class CropGenerator(object):
    """Base class that builds the TF1 crop-and-resize input graph.

    Manages the TF1 placeholders and ops needed to extract cell crops from
    a batch of full FOV images. The graph is built once via
    :meth:`build_input_graph` and reused across images.

    :class:`SingleImageCropGenerator` extends this for the profiling use case
    (one image at a time, all cells, no randomisation).
    """

    def __init__(self, config, dset):
        """
        Args:
            config: Experiment configuration dict.
            dset: :class:`~deepprofiler.dataset.image_dataset.ImageDataset`
                with at least one target.
        """
        self.config = config
        self.dset = dset

    def build_input_graph(self, export_masks=False):
        """Construct TF1 placeholders and the crop-and-resize op.

        Reads image dimensions and channel count from config. Creates
        placeholders for the raw image batch, normalised bounding boxes,
        box-to-image indicators, object mask labels, and per-target labels.
        Runs :func:`crop_graph` and stores all tensors in
        ``self.input_variables``.

        After this call, ``self.input_variables["labeled_crops"]`` is a
        ``tf.tuple`` of ``[crops_tensor, target_0_tensor, ...]`` that can be
        evaluated in a TF1 session.

        Args:
            export_masks: Pass True to append the binary object mask as an
                extra output channel (forwarded to :func:`crop_graph`).
        """
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

        box_size = self.config["dataset"]["locations"]["box_size"]
        img_width = self.config["dataset"]["images"]["width"]
        img_height = self.config["dataset"]["images"]["height"]

        num_targets = len(self.dset.targets)
        crop_shape = [(box_size, box_size, crop_channels)] + [()]*num_targets
        imgs_shape = [None, img_height, img_width, img_channels]
        batch_shape = (-1, img_height, img_width, img_channels)

        image_ph = tf.compat.v1.placeholder(tf.float32, shape=imgs_shape, name="raw_images")
        boxes_ph = tf.compat.v1.placeholder(tf.float32, shape=[None, 4], name="cell_boxes")
        box_ind_ph = tf.compat.v1.placeholder(tf.int32, shape=[None], name="box_indicators")
        mask_ind_ph = tf.compat.v1.placeholder(tf.int32, shape=[None], name="mask_indicators")
        targets_phs = {}
        for i in range(num_targets):
            tname = "target_" + str(i)
            targets_phs[tname] = tf.compat.v1.placeholder(tf.int32, shape=[None], name=tname)

        crop_op = crop_graph(
            image_ph, boxes_ph, box_ind_ph, mask_ind_ph,
            box_size, mask_objects, export_masks
        )
        labeled_crops = tf.tuple([crop_op] + [targets_phs[t] for t in targets_phs.keys()])

        self.input_variables = {
            "image_ph": image_ph,
            "boxes_ph": boxes_ph,
            "box_ind_ph": box_ind_ph,
            "targets_phs": targets_phs,
            "mask_ind_ph": mask_ind_ph,
            "labeled_crops": labeled_crops,
            "shapes": {
                "crops": crop_shape,
                "images": imgs_shape,
                "batch": batch_shape,
            },
        }

        self.train_variables = {
            "image_batch": self.input_variables["labeled_crops"][0],
            "target_0": tf.one_hot(
                self.input_variables["labeled_crops"][1],
                self.dset.targets[0].shape[1]
            )
        }


class SingleImageCropGenerator(CropGenerator):
    """Crop generator for profiling: processes one full FOV image at a time.

    Reads all cell centroids for an image from the locations CSV, draws
    bounding boxes, and runs the TF1 crop graph in a single session call.
    The resulting crop batch is stored in ``self.image_pool`` and yielded
    once by :meth:`generate` for the feature extractor to consume.

    No randomisation is applied — every cell in the image is cropped in the
    order it appears in the locations file.
    """

    def start(self, session):
        """Build the input graph, ready to process images.

        Must be called once before the first :meth:`prepare_image` call.
        Sets ``batch_size`` from ``config["train"]["validation"]["batch_size"]``.

        Args:
            session: Active TF1 session.
        """
        with tf.compat.v1.variable_scope("train_inputs"):
            self.config["train"]["model"]["params"]["batch_size"] = self.config["train"]["validation"]["batch_size"]
            self.build_input_graph()

    def prepare_image(self, session, image_array, meta, sample_first_crops=False):
        """Extract all cell crops from one full FOV image.

        Reads cell centroids from the locations CSV for this image, converts
        them to normalised bounding boxes, and runs the TF1 crop graph to
        produce a batch of ``(box_size, box_size, C)`` patches. Results are
        stored in ``self.image_pool`` for :meth:`generate` to yield.

        Args:
            session: Active TF1 session.
            image_array: numpy array of shape ``(H, W, C)`` — the full FOV
                image with all channels stacked.
            meta: Metadata row (Pandas Series) for this image, used to look
                up the locations file and target label.
            sample_first_crops: If True and the number of cells exceeds
                ``batch_size``, only the first ``batch_size`` cells are used.

        Returns:
            DataFrame of cell locations (rows correspond to crops in
            ``self.image_pool``).
        """
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
        batch["images"] = image_array[np.newaxis, ...]
        feed_dict = {
            self.input_variables["image_ph"]: batch["images"],
            self.input_variables["boxes_ph"]: boxes,
            self.input_variables["box_ind_ph"]: box_ind,
            self.input_variables["mask_ind_ph"]: mask_ind
        }

        ymins = boxes[:, [0, 2]].min(axis=1)
        ymaxs = boxes[:, [0, 2]].max(axis=1)
        xmins = boxes[:, [1, 3]].min(axis=1)
        xmaxs = boxes[:, [1, 3]].max(axis=1)
        if (np.any(ymins > 1) or np.any(xmins > 1) or
                np.any(ymaxs < 0) or np.any(ymaxs < 0)):
            print("WARNING: Some cell boxes are entirely outside the image")

        for i in range(num_targets):
            tname = "target_" + str(i)
            feed_dict[self.input_variables["targets_phs"][tname]] = targets[i]

        output = session.run(self.input_variables["labeled_crops"], feed_dict)
        output = {"image_batch": output[0], "target_0": output[1]}

        self.image_pool = output["image_batch"]
        num_classes = self.dset.targets[0].shape[1]
        self.label_pool = tf.compat.v1.keras.utils.to_categorical(output["target_0"], num_classes=num_classes)

        return batch["locations"][0]

    def generate(self, session, global_step=0):
        """Yield the crop batch produced by the last :meth:`prepare_image` call.

        Returns a single ``[crops, labels]`` pair where ``crops`` is
        ``self.image_pool`` (shape ``(num_cells, box_size, box_size, C)``) and
        ``labels`` is the one-hot encoded target array.
        """
        yield [self.image_pool, self.label_pool]
