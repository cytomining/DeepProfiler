import tensorflow as tf

PI = 3.1415926539

#################################################
## CROPPING AND TRANSFORMATION OPERATIONS
#################################################

def augment(crop):
    with tf.variable_scope("augmentation"):
        # Vertical and horizontal shifts:
        # Offsets is the amount of pixels to move the crop, sides is the direction of movement 
        offsets = tf.random_uniform([2], minval=0, maxval=int(crop.shape[0].value*0.3), dtype=tf.int32)
        sides = tf.random_uniform([2], minval=0, maxval=2, dtype=tf.int32)

        # Extract crop window
        row_start = offsets[0] * sides[0]
        row_length = crop.shape[0].value - offsets[0]
        col_start = offsets[1] * sides[1]
        col_length = crop.shape[1].value - offsets[1]
        sub_crop = tf.slice(crop, [row_start, col_start, 0], [row_length, col_length, crop.shape[2].value])

        # Pad sides with zeros
        upper_pads = offsets[0] * (1 - sides[0])
        lower_pads = offsets[0] * sides[0]
        left_pads = offsets[1] * (1 - sides[1])
        right_pads = offsets[1] * sides[1]
        moved_crop = tf.pad(sub_crop, [[upper_pads, lower_pads], [left_pads, right_pads], [0, 0]], "CONSTANT")

        # Horizontal flips
        augmented = tf.image.random_flip_left_right(moved_crop)

        # 90 degree rotations
        #angle = tf.random_uniform([1], minval=0, maxval=4, dtype=tf.int32)
        #augmented = tf.image.rot90(augmented, angle[0])

        # 360 degree rotations
        angle = tf.random_uniform([1], minval=0.0, maxval=2*PI, dtype=tf.float32)
        augmented = tf.contrib.image.rotate(augmented, angle[0], interpolation="BILINEAR")

        # Illumination changes
        #illum_s = tf.random_uniform([1], minval=0.8, maxval=1.2, dtype=tf.float32)
        #illum_t = tf.random_uniform([1], minval=-0.2, maxval=0.2, dtype=tf.float32)
        #augmented = augmented * illum_s
        #augmented = augmented + illum_t

    return augmented


def aument_multiple(crops, parallel=10):
    with tf.variable_scope("augmentation"):
        return tf.map_fn(augment, crops, parallel_iterations=parallel)

