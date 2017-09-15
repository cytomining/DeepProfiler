import tensorflow as tf

PI = 3.1415926539

#################################################
## CROPPING AND TRANSFORMATION OPERATIONS
#################################################

def augment(crop):
    with tf.variable_scope("augmentation"):
        augmented = tf.image.random_flip_left_right(crop)
        #angle = tf.random_uniform([1], minval=0, maxval=3, dtype=tf.int32)
        #augmented = tf.image.rot90(augmented, angle[0])
        angle = tf.random_uniform([1], minval=0.0, maxval=2*PI, dtype=tf.float32)
        augmented = tf.contrib.image.rotate(augmented, angle[0], interpolation="BILINEAR")
        illum = tf.random_uniform([1], minval=-0.1, maxval=0.1, dtype=tf.float32)
        augmented = augmented + illum
    return augmented


def aument_multiple(crops, parallel=10):
    with tf.variable_scope("augmentation"):
        return tf.map_fn(augment, crops, parallel_iterations=parallel)

