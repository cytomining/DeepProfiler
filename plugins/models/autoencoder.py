from comet_ml import Experiment

import tensorflow as tf

from deepprofiler.learning.model import DeepProfilerModel

tf.compat.v1.disable_v2_behavior()

##################################################
# Convolutional autoencoder with alternating
# convolutions and max pooling
##################################################


def define_model(config, dset):
    # Define input layer
    input_shape = (
        config["dataset"]["locations"]["box_size"],  # height
        config["dataset"]["locations"]["box_size"],  # width
        len(config["dataset"]["images"]["channels"])  # channels
    )
    input_image = tf.compat.v1.keras.layers.Input(input_shape)

    if config["train"]["model"]["params"]["conv_blocks"] < 1:
        raise ValueError("At least 1 convolutional block is required.")

    # Add convolutional blocks to encoder based on number specified in config, with increasing number of filters
    x = input_image
    for i in range(config["train"]["model"]["params"]["conv_blocks"]):
        x = tf.compat.v1.keras.layers.Conv2D(8 * 2 ** i, (3, 3), padding="same")(x)
        x = tf.compat.v1.keras.layers.BatchNormalization()(x)
        x = tf.compat.v1.keras.layers.Activation("relu")(x)
        x = tf.compat.v1.keras.layers.MaxPooling2D((2, 2))(x)
    conv_shape = x.shape[1:]
    x = tf.compat.v1.keras.layers.Flatten()(x)
    flattened_shape = x.shape[1:]
    encoded = tf.compat.v1.keras.layers.Dense(config["train"]["model"]["params"]["feature_dim"], name="encoded")(x)
    encoded_shape = encoded.shape[1:]
    encoder = tf.compat.v1.keras.Model(input_image, encoded)

    # Build decoder
    decoder_input = tf.compat.v1.keras.layers.Input(encoded_shape)
    decoder_layers = []
    decoder_layers.append(tf.compat.v1.keras.layers.Dense(flattened_shape[0], input_shape=encoded_shape))
    decoder_layers.append(tf.compat.v1.keras.layers.Reshape(conv_shape))
    for i in reversed(range(config["train"]["model"]["params"]["conv_blocks"])):
        decoder_layers.extend([
            tf.compat.v1.keras.layers.Conv2DTranspose(8 * 2 ** i, (3, 3), padding="same"),
            tf.compat.v1.keras.layers.BatchNormalization(),
            tf.compat.v1.keras.layers.Activation("relu"),
            tf.compat.v1.keras.layers.UpSampling2D((2, 2))
        ])
    decoder_layers.append(tf.compat.v1.keras.layers.Conv2DTranspose(len(config["dataset"]["images"]["channels"]), (3, 3), activation="sigmoid", padding="same"))
    decoder = tf.compat.v1.keras.Sequential(decoder_layers, name="decoded")
    decoded = decoder(encoded)
    decoder = tf.compat.v1.keras.Model(decoder_input, decoder(decoder_input))

    # Define autoencoder
    autoencoder = tf.compat.v1.keras.Model(input_image, decoded)
    optimizer = tf.compat.v1.keras.optimizers.Adam(learning_rate=config["train"]["model"]["params"]["learning_rate"])
    loss = "mse"

    return autoencoder, encoder, decoder, optimizer, loss


class ModelClass(DeepProfilerModel):
    def __init__(self, config, dset, generator, val_generator, is_training):
        super(ModelClass, self).__init__(config, dset, generator, val_generator, is_training)
        self.feature_model, self.encoder, self.decoder, self.optimizer, self.loss = define_model(config, dset)
