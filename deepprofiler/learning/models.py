import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim.nets

slim = tf.contrib.slim
resnet_v2 = slim.nets.resnet_v2
resnet_utils = slim.nets.resnet_utils

import keras_resnet.models
import keras
from keras.metrics import top_k_categorical_accuracy

def make_regularizer(transforms, reg_lambda):
    loss = 0
    for i in range(len(transforms)):
        for j in range(i+1, len(transforms)):
            loss += reg_lambda * tf.reduce_sum(tf.abs(tf.matmul(transforms[i], transforms[j], transpose_a=True, transpose_b=False)))
    return loss


def create_keras_resnet(input_shape, targets, top_k=5, learning_rate=0.001, embed_dims=256, reg_lambda=10, is_training=True):

    def custom_top_k(y_true, y_pred):
        return top_k_categorical_accuracy(y_true, y_pred, k=top_k)

    embed_dims = [256, 256]
    # 1. Create ResNet architecture to extract features
    input_image = keras.layers.Input(input_shape)

    model = keras_resnet.models.ResNet18(input_image, include_top=False)#, freeze_bn=not is_training)
    features = keras.layers.GlobalAveragePooling2D(name="pool5")(model.layers[-1].output)
    #features = keras.layers.core.Dropout(0.5)(features)

    # TODO: factorize the multi-target output model

    # 2. Create an output embedding for each target
    class_outputs = []

    i = 0
    for t in targets:
        #e = keras.layers.Dense(embed_dims[i], activation=None, name=t.field_name + "_embed", use_bias=False)(features)
        #e = keras.layers.normalization.BatchNormalization()(e)
        #e = keras.layers.core.Dropout(0.5)(e)
        #y = keras.layers.Dense(t.shape[1], activation="softmax", name=t.field_name)(e)
        y = keras.layers.Dense(t.shape[1], activation="softmax", name=t.field_name)(features)
        class_outputs.append(y)
        i += 1

    # 3. Define the regularized loss function
    transforms = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if v.name.find("_embed") != -1]
    if len(transforms) > 1:
        regularizer = make_regularizer(transforms, reg_lambda)
        def regularized_loss(y_true, y_pred):
            loss = keras.losses.categorical_crossentropy(y_true, y_pred) + regularizer
            return loss
        loss_func = ["categorical_crossentropy"]*(len(transforms)-1) + [regularized_loss]
    else:
        loss_func = ["categorical_crossentropy"]

    # 4. Create and compile model
    model = keras.models.Model(inputs=input_image, outputs=class_outputs)
    print(model.summary())
    print([t.shape for t in transforms])
    optimizer = keras.optimizers.Adam(lr=learning_rate)
    model.compile(optimizer, loss_func, metrics=[keras.metrics.categorical_accuracy,custom_top_k])

    return model


def create_recurrent_keras_resnet(input_shape, targets, top_k=5, learning_rate=0.001, embed_dims=256, reg_lambda=10, is_training=True):

    def custom_top_k(y_true, y_pred):
        return top_k_categorical_accuracy(y_true, y_pred, k=top_k)

    classes = targets[0].shape[1] # TODO: support for multiple targets
    input_set = keras.layers.Input(input_shape)
    net = keras_resnet.models.TimeDistributedResNet18(input_set, include_top=False)#, freeze_bn=not is_training)
    features = keras.layers.TimeDistributed(keras.layers.GlobalAveragePooling2D(), name="pool5")(net.output[-1])
    memory = keras.layers.GRU(embed_dims, return_sequences=False, stateful=False)(features)
    classifier = keras.layers.Dense(classes, activation="softmax", name="Allele")(memory)
    model = keras.models.Model(input_set, classifier)

    print(model.summary())
    optimizer = keras.optimizers.Adam(lr=learning_rate)
    model.compile(optimizer, loss="categorical_crossentropy", metrics=[keras.metrics.categorical_accuracy,custom_top_k])
    return model


def create_keras_vgg(input_shape, targets, top_k=5, learning_rate=0.001, embed_dims=256, reg_lambda=10, is_training=True):
    
    def custom_top_k(y_true, y_pred):
        return top_k_categorical_accuracy(y_true, y_pred, k=top_k)
    
    embed_dims = [256, 256]
    # 1. Create ResNet architecture to extract features
    input_image = keras.layers.Input(input_shape)

    # Block 1
    x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(input_image)
    x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    features = keras.layers.GlobalAveragePooling2D(name="pool5")(x)

    # TODO: factorize the multi-task loss for all models (repeated code)
    # 2. Create an output embedding for each target
    class_outputs = []

    i = 0
    for t in targets:
        e = keras.layers.Dense(embed_dims[i], activation=None, name=t.field_name + "_embed", use_bias=False)(features)
        e = keras.layers.normalization.BatchNormalization()(e)
        e = keras.layers.core.Dropout(0.5)(e)
        y = keras.layers.Dense(t.shape[1], activation="softmax", name=t.field_name)(e)
        class_outputs.append(y)
        i += 1

    # 3. Define the regularized loss function
    transforms = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if v.name.find("_embed") != -1]
    if len(transforms) > 1:
        regularizer = make_regularizer(transforms, reg_lambda)
        def regularized_loss(y_true, y_pred):
            loss = keras.losses.categorical_crossentropy(y_true, y_pred) + regularizer
            return loss
        loss_func = ["categorical_crossentropy"]*(len(transforms)-1) + [regularized_loss]
    else:
        loss_func = ["categorical_crossentropy"]*len(transforms)

    # 4. Create and compile model
    model = keras.models.Model(inputs=input_image, outputs=class_outputs)
    print(model.summary())
    print([t.shape for t in transforms])
    optimizer = keras.optimizers.Adam(lr=learning_rate)
    model.compile(optimizer, loss_func, metrics=[keras.metrics.categorical_accuracy,custom_top_k])

    return model
