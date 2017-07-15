import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim.nets

slim = tf.contrib.slim
resnet_v2 = slim.nets.resnet_v2

def create_resnet(inputs, num_classes):
    ## The following produces a feature vector of 1x1x2048
    net = resnet_v2.resnet_v2_50(inputs, num_classes, scope="convnet")
    return tf.reshape(net[1]["convnet/logits"], (-1, num_classes))


def vgg_module(x, filters, sizes, layers, scope_name):
    with tf.variable_scope(scope_name):
        net = slim.stack(x, slim.conv2d, [(filters, sizes)]*layers, scope="conv")
        net = slim.max_pool2d(net, [2, 2], scope="pool")
        net = slim.batch_norm(net, scope="batch_norm")
    return net


def create_vgg(images, num_classes):
    # Assumes input of 128x128, produces 4096 features 
    with tf.variable_scope("convnet"):
        net = vgg_module(images, 32, [3, 3], 2, "scale1")
        net = vgg_module(net, 32, [3, 3], 2, "scale2")
        net = vgg_module(net, 64, [3, 3], 2, "scale3")
        net = vgg_module(net, 128, [3, 3], 2, "scale4")
        net = vgg_module(net, 256, [3, 3], 2, "scale5")
        net = slim.conv2d(net, 64, [1, 1], scope="fmap")
        net = slim.flatten(net, scope="features")
        net = slim.fully_connected(net, num_classes, scope="logits")
    return net


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
    ops = [train_op, train_accuracy, top_5_acc, merged_summary]
    return ops, train_writer


def create_validator(net, labels, sess, config):
    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=net)
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
    tf.summary.histogram("logits", net)
    merged_summary = tf.summary.merge_all()
    val_writer = tf.summary.FileWriter(config["training"]["output"] + "/model/", sess.graph)
    # Return 2 objects: Array with validation ops and summary writter
    gt = tf.argmax(labels,1)
    pr = tf.reduce_max(predictions,1)
    ops = [loss, val_accuracy, top_5_acc, gt, pr, merged_summary]
    return ops, val_writer

