import keras
from keras.layers import *
from keras.models import Model, Sequential
from keras.optimizers import Adam

from deepprofiler.learning.model import DeepProfilerModel


def define_model(config, dset):
    input_shape = (
        config["sampling"]["box_size"],  # height
        config["sampling"]["box_size"],  # width
        len(config["image_set"]["channels"])  # channels
    )
    input_image = keras.layers.Input(input_shape)

    x = Conv2D(8, (3, 3), activation='relu', padding='same')(input_image)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same', name='encoded')(x)
    encoded_shape = encoded._keras_shape[1:]
    encoder = Model(input_image, encoded)

    # x = Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(encoded)
    # x = UpSampling2D((2, 2))(x)
    # x = Conv2DTranspose(16, (3, 3), activation='relu', padding='same')(x)
    # x = UpSampling2D((2, 2))(x)
    # x = Conv2DTranspose(8, (3, 3), activation='relu')(x)
    # x = UpSampling2D((2, 2))(x)
    # decoded = Conv2DTranspose(len(config["image_set"]["channels"]), (3, 3), activation='sigmoid', padding='same')(x)
    decoder_input = Input(encoded_shape)
    decoder = Sequential([
        Conv2DTranspose(32, (3, 3), activation='relu', padding='same'),
        UpSampling2D((2, 2)),
        Conv2DTranspose(16, (3, 3), activation='relu', padding='same'),
        UpSampling2D((2, 2)),
        Conv2DTranspose(8, (3, 3), activation='relu', padding='same'),
        UpSampling2D((2, 2)),
        Conv2DTranspose(len(config["image_set"]["channels"]), (3, 3), activation='sigmoid', padding='same')
    ], name='decoded')
    decoded = decoder(encoded)
    decoder = Model(decoder_input, decoder(decoder_input))

    autoencoder = Model(input_image, decoded)
    autoencoder.compile(optimizer=Adam(lr=config['training']['learning_rate']), loss='mse')

    return autoencoder, encoder, decoder


class ModelClass(DeepProfilerModel):
    def __init__(self, config, dset, generator):
        super(ModelClass, self).__init__(config, dset, generator)
        self.model, self.encoder, self.decoder = define_model(config, dset)
