import tensorflow as tf
import numpy as np
slim = tf.contrib.slim


def vgg_module(x, filters, sizes, layers, scope_name):
    with tf.variable_scope(scope_name):
        net = slim.stack(x, slim.conv2d, [(filters, sizes)]*layers, scope="conv")
        net = slim.max_pool2d(net, [2, 2], scope="pool")
        net = slim.batch_norm(net, scope="batch_norm")
    return net


def create_vgg(images, num_classes):
    with tf.variable_scope("convnet"):
        net = vgg_module(images, 32, [3, 3], 2, "scale1")
        net = vgg_module(net, 32, [3, 3], 2, "scale2")
        net = vgg_module(net, 64, [3, 3], 2, "scale3")
        net = vgg_module(net, 128, [3, 3], 2, "scale4")
        net = slim.conv2d(net, 32, [1, 1], scope="fmap")
        net = slim.flatten(net, scope="features")
        net = slim.fully_connected(net, num_classes, scope="logits")
    return net


def create_trainer(net, labels, sess, config):
    # Loss and optimizer
    loss = tf.losses.mean_squared_error(labels=labels, predictions=net)
    convnet_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "convnet")
    optimizer = tf.train.AdamOptimizer(config["training"]["learning_rate"])
    train_op = optimizer.minimize(loss, var_list=convnet_vars)
    # Summaries
    tf.summary.scalar("training_loss", loss)
    merged_summary = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(config["training"]["output"] + "/log", sess.graph)
    # Return 2 objects: An array with training ops and a summary writter object
    ops = [train_op, merged_summary]
    return ops, train_writer

