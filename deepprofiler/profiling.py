import importlib
import os

import efficientnet.tfkeras as efn
import numpy as np
import tensorflow as tf

from deepprofiler.dataset.utils import tic, toc

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
    num = config["train"]["model"]["params"]["conv_blocks"]
    assert num in _EFFICIENTNET_MODELS, f"{num} not in supported EfficientNet variants: {list(_EFFICIENTNET_MODELS)}"

    if "use_pretrained_input_size" in config["profile"]:
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

    def __init__(self, config, dset):
        self.config = config
        self.dset = dset
        self.num_channels = len(self.config["dataset"]["images"]["channels"])

        crop_gen_module = importlib.import_module(
            "deepprofiler.crop_generators.{}".format(config["train"]["model"]["crop_generator"])
        )
        self.profile_crop_generator = crop_gen_module.SingleImageGeneratorClass

        self.config["num_classes"] = self.dset.targets[0].shape[1]
        self.feature_model = build_model(self.config)
        self.profile_crop_generator = self.profile_crop_generator(config, dset)

    def configure(self):
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
        output_file = self.config["paths"]["features"] + "/{}/{}/{}.npz"
        output_file = output_file.format(meta["Metadata_Plate"], meta["Metadata_Well"], meta["Metadata_Site"])
        if os.path.isfile(output_file):
            print("Already done:", output_file)
            return False
        return True

    def extract_features(self, key, image_array, meta):
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

        repeats = self.config["train"]["model"]["crop_generator"] in [
            "repeat_channel_crop_generator", "individual_channel_cropgen"
        ]

        crops = next(self.profile_crop_generator.generate(tf.compat.v1.keras.backend.get_session()))[0]
        feats = self.feat_extractor.predict(crops, batch_size=batch_size)

        while len(feats.shape) > 2:
            feats = np.mean(feats, axis=1)

        if repeats:
            feats = np.reshape(feats, (self.num_channels, total_crops, -1))
            feats = np.concatenate(feats, axis=-1)

        key_values = {k: meta[k] for k in meta.keys()}
        key_values["Metadata_Model"] = "efficientnet"
        np.savez_compressed(output_file, features=feats, metadata=key_values, locations=crop_locations)
        toc(image_key + " (" + str(total_crops) + " cells)", start)


def profile(config, dset):
    p = Profile(config, dset)
    p.configure()
    dset.scan(p.extract_features, frame="all", check=p.check)
    print("Profiling: done")
