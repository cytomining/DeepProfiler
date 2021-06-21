from comet_ml import Experiment

import tensorflow as tf

from deepprofiler.learning.model import DeepProfilerModel
from deepprofiler.imaging.augmentations import AugmentationLayer


tf.compat.v1.disable_v2_behavior()


##################################################
# Basic convolutional network with alternating
# convolutions and max pooling
##################################################


def define_model(config, dset, is_training):
    # Define input layer
    input_shape = (
        config["dataset"]["locations"]["box_size"],  # height
        config["dataset"]["locations"]["box_size"],  # width
        len(config["dataset"]["images"]["channels"])  # channels
    )
    input_image = tf.compat.v1.keras.layers.Input(input_shape)

    if config["train"]["model"]["params"]["conv_blocks"] < 1:
        raise ValueError("At least 1 convolutional block is required.")

    # Add convolutional blocks based on number specified in config, with increasing number of filters
    x = input_image
    if is_training:
        x = AugmentationLayer()(x)
    for i in range(config["train"]["model"]["params"]["conv_blocks"]):
        x = tf.compat.v1.keras.layers.Conv2D(8 * 2 ** i, (3, 3), padding="same")(x)
        x = tf.compat.v1.keras.layers.BatchNormalization()(x)
        x = tf.compat.v1.keras.layers.Activation("relu")(x)
        x = tf.compat.v1.keras.layers.MaxPooling2D((2, 2), padding="same")(x)
    x = tf.compat.v1.keras.layers.Flatten()(x)
    features = tf.compat.v1.keras.layers.Dense(config["train"]["model"]["params"]["feature_dim"], activation="relu", name="features")(x)

    # Create an output embedding for each target
    class_outputs = []
    i = 0
    for t in dset.targets:
        y = tf.compat.v1.keras.layers.Dense(t.shape[1], activation="softmax", name=t.field_name)(features)
        class_outputs.append(y)
        i += 1

    # Define model
    model = tf.compat.v1.keras.Model(input_image, class_outputs)
    optimizer = tf.compat.v1.keras.optimizers.Adam(learning_rate=config["train"]["model"]["params"]["learning_rate"])
    loss = "categorical_crossentropy"
    return model, optimizer, loss


class ModelClass(DeepProfilerModel):
    def __init__(self, config, dset, generator, val_generator, is_training):
        super(ModelClass, self).__init__(config, dset, generator, val_generator, is_training)
        self.feature_model, self.optimizer, self.loss = define_model(config, dset, is_training)
