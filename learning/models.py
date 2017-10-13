import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim.nets

slim = tf.contrib.slim
resnet_v2 = slim.nets.resnet_v2
resnet_utils = slim.nets.resnet_utils

import keras_resnet.models
import keras


def make_regularizer(transforms, reg_lambda):
    loss = 0
    for i in range(len(transforms)):
        for j in range(i+1, len(transforms)):
            loss += reg_lambda * tf.reduce_sum(tf.abs(tf.matmul(transforms[i], transforms[j], transpose_a=True, transpose_b=False)))
    return loss


def create_keras_resnet(input_shape, targets, learning_rate=0.001, embed_dims=256, reg_lambda=10):
    embed_dims = [384, 128]
    # 1. Create ResNet architecture to extract features
    input_image = keras.layers.Input(input_shape)
    model = keras_resnet.models.ResNet18(input_image, include_top=False)
    features = keras.layers.GlobalAveragePooling2D(name="pool5")(model.layers[-1].output)
    #features = keras.layers.core.Dropout(0.5)(features)

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
    model.compile(optimizer, loss_func, ["categorical_accuracy"])

    return model


def create_resnet(inputs, num_classes, is_training=True):
    net = resnet_v2.resnet_v2_50(inputs, num_classes, scope="convnet", is_training=is_training)
    return tf.reshape(net[1]["convnet/logits"], (-1, num_classes))


def create_trainer(net, labels, sess, config):
    # labels are assumed to be one_hot encoded
    # Loss and optimizer
    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=net)
    convnet_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "convnet")
    optimizer = tf.train.AdamOptimizer(config["training"]["learning_rate"])
    train_op = optimizer.minimize(loss, var_list=convnet_vars)
    # Accuracy
    predictions = tf.nn.softmax(net)
    correct_prediction = tf.equal(tf.argmax(labels,1), tf.argmax(predictions,1))
    train_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    top_5_acc = tf.reduce_mean( tf.to_float( tf.nn.in_top_k(
                        predictions=predictions, 
                        targets=tf.argmax(labels, 1), 
                        k=5
                )))
    # Summaries
    tf.summary.scalar("training_loss", loss)
    tf.summary.scalar("training_accuracy", train_accuracy)
    tf.summary.scalar("training_top_5_acc", top_5_acc)
    merged_summary = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(config["training"]["output"] + "/model/", sess.graph)
    # Return 2 objects: An array with training ops and a summary writter object
    gt = tf.argmax(labels,1)
    pr = tf.argmax(predictions,1)
    ops = [train_op, loss, train_accuracy, top_5_acc, gt, pr, merged_summary]
    return ops, train_writer


def create_validator(net, labels, sess, config):
    #loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=net)
    loss = tf.reduce_max(net)
    # labels are assumed to be one_hot encoded
    # Accuracy
    predictions = tf.nn.softmax(net)
    correct_prediction = tf.equal(tf.argmax(labels,1), tf.argmax(predictions,1))
    val_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    top_5_acc = tf.reduce_mean( tf.to_float( tf.nn.in_top_k(
                        predictions=predictions, 
                        targets=tf.argmax(labels, 1), 
                        k=5
                )))
    # Summaries
    tf.summary.scalar("validation_accuracy", val_accuracy)
    tf.summary.scalar("validation_top_5_acc", top_5_acc)
    #tf.summary.histogram("logits", net)
    merged_summary = tf.summary.merge_all()
    val_writer = tf.summary.FileWriter(config["training"]["output"] + "/model/", sess.graph)
    # Return 2 objects: Array with validation ops and summary writter
    gt = tf.argmax(labels,1)
    pr = tf.argmax(predictions,1)
    ops = [loss, val_accuracy, top_5_acc, gt, pr, merged_summary]
    return ops, val_writer

