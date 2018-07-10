import keras
from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam


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
    # features = Flatten(name='features')(encoded)

    x = Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2DTranspose(16, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2DTranspose(8, (3, 3), activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2DTranspose(len(config["image_set"]["channels"]), (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_image, decoded)
    autoencoder.compile(optimizer=Adam(lr=config['training']['learning_rate']), loss='mse')
    return autoencoder
