from comet_ml import Experiment

import keras
import keras_resnet
import keras_resnet.models
import tensorflow as tf

from deepprofiler.learning.model import DeepProfilerModel


##################################################
# ResNet architecture as defined in "Deep Residual
# Learning for Image Recognition" by Kaiming He,
# Xiangyu Zhang, Shaoqing Ren, Jian Sun
# https://arxiv.org/abs/1512.03385
##################################################

def define_model(config, dset):

    # 1. Create ResNet architecture to extract features
    input_shape = (
        config["sampling"]["box_size"],  # height
        config["sampling"]["box_size"],  # width
        len(config["image_set"]["channels"])  # channels
    )
    input_image = keras.layers.Input(input_shape)

    model = keras_resnet.models.ResNet18(input_image, include_top=False)#, freeze_bn=not is_training)
    features = keras.layers.GlobalAveragePooling2D(name="pool5")(model.layers[-1].output)
    #features = keras.layers.core.Dropout(0.5)(features)

    # TODO: factorize the multi-target output model

    # 2. Create an output embedding for each target
    class_outputs = []

    i = 0
    for t in dset.targets:
        y = keras.layers.Dense(t.shape[1], activation="softmax", name=t.field_name)(features)
        class_outputs.append(y)
        i += 1

    # 3. Define the loss function
    loss_func = "categorical_crossentropy"

    # 4. Create and compile model
    model = keras.models.Model(inputs=input_image, outputs=class_outputs)
    optimizer = keras.optimizers.Adam(lr=config["model"]["params"]["learning_rate"])

    return model, optimizer, loss_func


class ModelClass(DeepProfilerModel):
    def __init__(self, config, dset, generator):
        super(ModelClass, self).__init__(config, dset, generator)
        self.model, self.optimizer, self.loss = define_model(config, dset)

