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
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Flatten()(x)
    features = Dense(config['model']['feature_dim'], activation='relu')(x)

    class_outputs = []
    i = 0
    for t in dset.targets:
        y = keras.layers.Dense(t.shape[1], activation="softmax", name=t.field_name)(features)
        class_outputs.append(y)
        i += 1

    model = Model(input_image, class_outputs)
    model.compile(optimizer=Adam(lr=config['training']['learning_rate']), loss='categorical_crossentropy')
    return model


class ModelClass(DeepProfilerModel):
    def __init__(self, config, dset, generator):
        super(ModelClass, self).__init__(config, dset, generator)
        self.model = define_model(config, dset)
