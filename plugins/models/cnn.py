from comet_ml import Experiment

import keras
from keras.layers import *
from keras.models import Model, Sequential
from keras.optimizers import Adam

from deepprofiler.learning.model import DeepProfilerModel


##################################################
# Basic convolutional network with alternating
# convolutions and max pooling
##################################################


def define_model(config, dset, metrics):
    # Define input layer
    input_shape = (
        config["sampling"]["box_size"],  # height
        config["sampling"]["box_size"],  # width
        len(config["image_set"]["channels"])  # channels
    )
    input_image = keras.layers.Input(input_shape)

    if config['model']['conv_blocks'] < 1:
        raise ValueError("At least 1 convolutional block is required.")

    # Add convolutional blocks based on number specified in config, with increasing number of filters
    x = input_image
    for i in range(config['model']['conv_blocks']):
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
    x = Flatten()(x)
    features = Dense(config['model']['feature_dim'], activation='relu')(x)

    # Create an output embedding for each target
    class_outputs = []
    i = 0
    for t in dset.targets:
        y = keras.layers.Dense(t.shape[1], activation="softmax", name=t.field_name)(features)
        class_outputs.append(y)
        i += 1

    # Define model
    model = Model(input_image, class_outputs)
    optimizer=Adam(lr=config["model"]["params"]['learning_rate'])
    loss='categorical_crossentropy'
    return model, optimizer, loss


class ModelClass(DeepProfilerModel):
    def __init__(self, config, dset, generator):
        super(ModelClass, self).__init__(config, dset, generator)
        self.model, self.optimizer, self.loss = define_model(config, dset)
