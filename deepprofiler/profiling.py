import os

import efficientnet.tfkeras as efn
import numpy as np
import tensorflow as tf

from deepprofiler.dataset.utils import tic, toc
from deepprofiler.imaging.cropping import SingleImageCropGenerator

tf.compat.v1.disable_v2_behavior()
tf.config.run_functions_eagerly(False)

_EFFICIENTNET_MODELS = {
    0: efn.EfficientNetB0,
    1: efn.EfficientNetB1,
    2: efn.EfficientNetB2,
    3: efn.EfficientNetB3,
    4: efn.EfficientNetB4,
    5: efn.EfficientNetB5,
    6: efn.EfficientNetB6,
    7: efn.EfficientNetB7,
}


def build_model(config):
    """Build the EfficientNet feature extraction model from config.

    In standard profiling mode the model is built with the number of channels
    specified in the dataset config (e.g. 5 for Cell Painting) and no ImageNet
    weights.  The final layers are GlobalAveragePooling2D (named ``pool5``) and
    a softmax Dense head (named ``ClassProb``).  Weights are loaded separately
    via :meth:`Profile.configure`.

    If ``config["profile"]["use_pretrained_input_size"]`` is set, builds a
    3-channel ImageNet-pretrained model at that spatial resolution instead —
    useful for fine-tuning experiments.
    """
    num = config["train"]["model"]["params"]["conv_blocks"]
    assert num in _EFFICIENTNET_MODELS, f"{num} not in supported EfficientNet variants: {list(_EFFICIENTNET_MODELS)}"

    if config["profile"].get("use_pretrained_input_size"):
        size = config["profile"]["use_pretrained_input_size"]
        inp = tf.keras.layers.Input((size, size, 3), name="input")
        return _EFFICIENTNET_MODELS[num](input_tensor=inp, include_top=True, weights="imagenet")

    h = w = config["dataset"]["locations"]["box_size"]
    c = len(config["dataset"]["images"]["channels"])
    inp = tf.keras.layers.Input((h, w, c))
    base = _EFFICIENTNET_MODELS[num](input_tensor=inp, include_top=False, weights=None)
    features = tf.keras.layers.GlobalAveragePooling2D(name="pool5")(base.layers[-1].output)
    y = tf.keras.layers.Dense(config["num_classes"], activation="softmax", name="ClassProb")(features)
    return tf.keras.Model(inputs=inp, outputs=[y])


class Profile(object):
    """Extract per-cell deep learning features from microscopy images.

    This class implements the full profiling pipeline:

    1. Build an EfficientNet model matching the dataset channel count.
    2. Load pre-trained weights from a checkpoint (e.g. Cell Painting CNN v1).
    3. For each image, crop patches around cell centroids and run them through
       the network, collecting the activations at a named intermediate layer.
    4. Write per-image ``.npz`` files containing a ``features`` array of shape
       ``(num_cells, feature_dim)``.

    Typical usage::

        dset = deepprofiler.dataset.image_dataset.read_dataset(config)
        deepprofiler.profiling.profile(config, dset)

    Or step-by-step (e.g. in tests)::

        prof = Profile(config, dset)
        prof.configure()
        prof.extract_features(key, image_array, meta)
    """

    def __init__(self, config, dset):
        """Initialise the profiler and build the model graph.

        Args:
            config: Experiment configuration dict.  Must include
                ``dataset``, ``train``, ``profile``, ``paths``, and
                ``num_classes`` keys.
            dset: :class:`~deepprofiler.dataset.image_dataset.ImageDataset`
                with at least one target added via ``add_target``.
        """
        self.config = config
        self.dset = dset
        self.num_channels = len(self.config["dataset"]["images"]["channels"])

        self.config["num_classes"] = self.dset.targets[0].shape[1]
        self.feature_model = build_model(self.config)
        self.profile_crop_generator = SingleImageCropGenerator(config, dset)

    def configure(self):
        """Start the crop generator and load checkpoint weights.

        Loads weights from ``config["paths"]["checkpoints"] / config["profile"]["checkpoint"]``.
        If the checkpoint head has a different number of classes (e.g. loading
        Cell Painting CNN v1 with a custom class count), the classifier layer is
        renamed and weights are matched by layer name instead.

        After loading, builds ``self.feat_extractor``: a sub-model whose output
        is the activation of ``config["profile"]["feature_layer"]`` (e.g.
        ``"pool5"`` for the 1280-d GlobalAveragePooling2D embedding).
        """
        self.profile_crop_generator.start(tf.compat.v1.keras.backend.get_session())

        if self.config["profile"]["checkpoint"] != "None":
            checkpoint = self.config["paths"]["checkpoints"] + "/" + self.config["profile"]["checkpoint"]
            try:
                self.feature_model.load_weights(checkpoint)
            except ValueError:
                print("Loading weights without classifier (different number of classes)")
                self.feature_model.layers[-1]._name = "classifier"
                self.feature_model.load_weights(checkpoint, by_name=True)

        self.feature_model.summary()
        self.feat_extractor = tf.compat.v1.keras.Model(
            self.feature_model.inputs,
            self.feature_model.get_layer(self.config["profile"]["feature_layer"]).output
        )
        print("Extracting output from layer:", self.config["profile"]["feature_layer"])

    def check(self, meta):
        """Return True if this image still needs to be profiled.

        Skips images whose output ``.npz`` already exists, enabling resumable
        profiling runs.

        Args:
            meta: A row from the metadata DataFrame (Pandas Series or dict-like)
                with ``Metadata_Plate``, ``Metadata_Well``, and ``Metadata_Site``.
        """
        output_file = self.config["paths"]["features"] + "/{}/{}/{}.npz"
        output_file = output_file.format(meta["Metadata_Plate"], meta["Metadata_Well"], meta["Metadata_Site"])
        if os.path.isfile(output_file):
            print("Already done:", output_file)
            return False
        return True

    def extract_features(self, key, image_array, meta):
        """Extract and save features for a single image.

        Crops cell patches from ``image_array`` using the pre-loaded crop
        generator, runs them through ``self.feat_extractor``, and writes the
        result to a compressed ``.npz`` file.

        Output path: ``{paths.features}/{Plate}/{Well}/{Site}.npz``

        The ``.npz`` file contains:
        - ``features``: float array of shape ``(num_cells, feature_dim)``
        - ``metadata``: dict of metadata fields plus ``Metadata_Model``
        - ``locations``: cell centroid coordinates from the locations CSV

        Args:
            key: Image key (index into the dataset); may be ``None`` when
                called directly.
            image_array: numpy array of shape ``(H, W, C)`` where C matches
                the configured channel count.
            meta: Metadata row (Pandas Series) for this image.
        """
        start = tic()
        output_file = self.config["paths"]["features"] + "/{}/{}/{}.npz"
        output_file = output_file.format(meta["Metadata_Plate"], meta["Metadata_Well"], meta["Metadata_Site"])
        os.makedirs(self.config["paths"]["features"] + "/{}/{}".format(meta["Metadata_Plate"], meta["Metadata_Well"]), exist_ok=True)

        batch_size = self.config["profile"]["batch_size"]
        image_key, image_names, outlines = self.dset.get_image_paths(meta)
        crop_locations = self.profile_crop_generator.prepare_image(
            tf.compat.v1.keras.backend.get_session(),
            image_array,
            meta,
            False
        )
        total_crops = len(crop_locations)
        if total_crops == 0:
            print("No cells to profile:", output_file)
            return

        if (self.config["dataset"]["images"]["width"] != image_array.shape[1] or
                self.config["dataset"]["images"]["height"] != image_array.shape[0]):
            config_shape = (self.config["dataset"]["images"]["width"], self.config["dataset"]["images"]["height"])
            im_shape = (image_array.shape[1], image_array.shape[0])
            raise ValueError("Loaded image shape WxH " + str(im_shape) +
                             " != configured image shape WxH " + str(config_shape))

        crops = next(self.profile_crop_generator.generate(tf.compat.v1.keras.backend.get_session()))[0]
        feats = self.feat_extractor.predict(crops, batch_size=batch_size)

        while len(feats.shape) > 2:
            feats = np.mean(feats, axis=1)

        key_values = {k: meta[k] for k in meta.keys()}
        key_values["Metadata_Model"] = "efficientnet"
        np.savez_compressed(output_file, features=feats, metadata=key_values, locations=crop_locations)
        toc(image_key + " (" + str(total_crops) + " cells)", start)


def profile(config, dset):
    p = Profile(config, dset)
    p.configure()
    dset.scan(p.extract_features, frame="all", check=p.check)
    print("Profiling: done")
