import tensorflow as tf
import keras as keras
import keras.applications 
import numpy
import os
import warnings

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
        config["train"]["sampling"]["box_size"],  # height
        config["train"]["sampling"]["box_size"],  # width
        len(config["dataset"]["images"][
            "channels"])  # channels
    )
    input_image = keras.layers.Input(input_shape)
    model = keras.applications.ResNet50(include_top=False, weights=None, input_tensor=input_image)
    features = keras.layers.GlobalAveragePooling2D(name="pool5")(model.layers[-1].output)

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
    ## Added weight decay following tricks reported in:
    ## https://github.com/keras-team/keras/issues/2717
    regularizer = keras.regularizers.l2(0.00001)
    for layer in model.layers:
        if hasattr(layer, "kernel_regularizer"):
            setattr(layer, "kernel_regularizer", regularizer)
    model = keras.models.model_from_json(model.to_json())
    optimizer = keras.optimizers.SGD(lr=config["train"]["model"]["params"]["learning_rate"], momentum=0.9, nesterov=True)
    # optimizer = keras.optimizers.Adam(lr=config["train"]["model"]["params"]["learning_rate"])

    return model, optimizer, loss_func


class ModelClass(DeepProfilerModel):
    def __init__(self, config, dset, generator, val_generator):
        super(ModelClass, self).__init__(config, dset, generator, val_generator)
        self.feature_model, self.optimizer, self.loss = define_model(config, dset)


    def copy_pretrained_weights(self):
        base_model = keras.applications.ResNet50(weights='imagenet', include_top=False)
        # => Transfer all weights except conv1.1
        for i in range(3,len(base_model.layers)):
            if len(base_model.layers[i].weights) > 0:
                self.feature_model.layers[i].set_weights(base_model.layers[i].get_weights())
        
        # => Replicate filters of first layer as needed
        weights = base_model.layers[2].get_weights()
        available_channels = weights[0].shape[2]
        target_shape = self.feature_model.layers[2].weights[0].shape
        new_weights = numpy.zeros(target_shape)
        for i in range(new_weights.shape[2]):
            j = i%available_channels
            new_weights[:,:,i,:] = weights[0][:,:,j,:]
            self.feature_model.layers[2].set_weights([new_weights, weights[1]])
        print("Network initialized with pretrained ImageNet weights")



