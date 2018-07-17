from comet_ml import Experiment

import keras
from keras.layers import *
from keras.models import Model, Sequential
from keras.optimizers import Adam

from deepprofiler.learning.model import DeepProfilerModel


##################################################
# Convolutional autoencoder with alternating
# convolutions and max pooling
##################################################


def define_model(config, dset):
    # Define input layer
    input_shape = (
        config["sampling"]["box_size"],  # height
        config["sampling"]["box_size"],  # width
        len(config["image_set"]["channels"])  # channels
    )
    input_image = keras.layers.Input(input_shape)

    if config['model']['conv_blocks'] < 1:
        raise ValueError("At least 1 convolutional block is required.")

    # Add convolutional blocks to encoder based on number specified in config, with increasing number of filters
    x = input_image
    for i in range(config['model']['conv_blocks']):
        x = Conv2D(8 * 2 ** i, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2))(x)
    encoded = x
    encoded_shape = encoded._keras_shape[1:]
    encoder = Model(input_image, encoded)

    # Build decoder
    decoder_input = Input(encoded_shape)
    decoder_layers = []
    for i in reversed(range(config['model']['conv_blocks'])):
        if i == config['model']['conv_blocks'] - 1:
            decoder_layers.extend([
                Conv2DTranspose(8 * 2 ** i, (3, 3), activation='relu', padding='same', input_shape=encoded_shape),
                UpSampling2D((2, 2))
            ])
        else:
            decoder_layers.extend([
                Conv2DTranspose(8 * 2 ** i, (3, 3), activation='relu', padding='same'),
                UpSampling2D((2, 2))
            ])
    decoder_layers.append(Conv2DTranspose(len(config["image_set"]["channels"]), (3, 3), activation='sigmoid', padding='same'))
    decoder = Sequential(decoder_layers, name='decoded')
    decoded = decoder(encoded)
    decoder = Model(decoder_input, decoder(decoder_input))

    # Define autoencoder
    autoencoder = Model(input_image, decoded)
    optimizer=Adam(lr=config["model"]["params"]['learning_rate'])
    loss='mse'

    return autoencoder, encoder, decoder, optimizer, loss


class ModelClass(DeepProfilerModel):
    def __init__(self, config, dset, generator, val_generator):
        super(ModelClass, self).__init__(config, dset, generator, val_generator)
        self.model, self.encoder, self.decoder, self.optimizer, self.loss = define_model(config, dset)
