import keras
from keras import backend as K
from keras import objectives
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
    x = MaxPooling2D((2, 2), padding='same')(x)
    encoded_shape = x._keras_shape[1:]
    x = Flatten()(x)

    z_mean = Dense(config['model']['latent_dim'], name='z_mean')(x)
    z_log_sigma = Dense(config['model']['latent_dim'], name='z_log_sigma')(x)

    def sampling(args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(shape=(config['training']['minibatch'], config['model']['latent_dim']),
                                  mean=0., stddev=config['model']['epsilon_std'])
        return z_mean + K.exp(z_log_sigma) * epsilon

    z = Lambda(sampling, output_shape=(config['model']['latent_dim'],), name='z')([z_mean, z_log_sigma])
    encoder = Model(input_image, z_mean)

    decoder_input = Input((config['model']['latent_dim'],))
    decoder = Sequential([
        Reshape(encoded_shape),
        Conv2DTranspose(32, (3, 3), activation='relu', padding='same'),
        UpSampling2D((2, 2)),
        Conv2DTranspose(16, (3, 3), activation='relu', padding='same'),
        UpSampling2D((2, 2)),
        Conv2DTranspose(8, (3, 3), activation='relu', padding='same'),
        UpSampling2D((2, 2)),
        Conv2DTranspose(len(config["image_set"]["channels"]), (3, 3), activation='sigmoid', padding='same')
    ], name='decoded')
    decoded = decoder(z)
    generator = Model(decoder_input, decoder(decoder_input))

    vae = Model(input_image, decoded)

    def vae_loss(x, x_decoded_mean):
        x = K.flatten(x)
        x_decoded_mean = K.flatten(x_decoded_mean)
        xent_loss = objectives.binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)
        return xent_loss + kl_loss

    vae.compile(optimizer=Adam(lr=config['training']['learning_rate']), loss=vae_loss)

    return vae, encoder, generator


class ModelClass(DeepProfilerModel):
    def __init__(self, config, dset, generator):
        super(ModelClass, self).__init__(config, dset, generator)
        self.model, self.encoder, self.generator = define_model(config, dset)
