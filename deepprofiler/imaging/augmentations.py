import tensorflow as tf

#################################################
# CROPPING AND TRANSFORMATION OPERATIONS
#################################################


def random_illumination(image):
    # Make channels independent images
    if isinstance(image.shape[-1], int):
        numchn = image.shape[-1]
    else:
        numchn = image.shape[-1].value

    source = tf.transpose(image, [2, 1, 0])
    source = tf.expand_dims(source, -1)
    source = tf.image.grayscale_to_rgb(source)
    
    # Apply illumination augmentations
    bright = tf.random.uniform([numchn], minval=-0.1, maxval=0.1, dtype=tf.float32)
    channels = [tf.image.adjust_brightness(source[s,...], bright[s]) for s in range(numchn)]
    contrast = tf.random.uniform([numchn], minval=0.8, maxval=1.2, dtype=tf.float32)
    channels = [tf.image.adjust_contrast(channels[s], contrast[s]) for s in range(numchn)]
    result = tf.concat([tf.expand_dims(t, 0) for t in channels], axis=0)
    
    # Recover multi-channel image
    result = tf.image.rgb_to_grayscale(result)
    result = tf.transpose(result[:, :, :, 0], [2, 1, 0])
    return result


def random_flips(image):
    # Horizontal flips
    augmented = tf.image.random_flip_left_right(image)

    # 90 degree rotations
    angle = tf.random.uniform([1], minval=0, maxval=4, dtype=tf.int32)
    augmented = tf.image.rot90(augmented, angle[0])
    
    return augmented


def random_crop(image):
    if isinstance(image.shape[-1], int):
        w, h, c = image.shape[0], image.shape[1], image.shape[-1]
    else:
        w, h, c = image.shape[0].value, image.shape[1].value, image.shape[-1].value
    if tf.less(tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32), tf.cast(0.5, tf.float32)):
        size = tf.random.uniform([1], minval=int(w * 0.8), maxval=w, dtype=tf.int32)
        image = tf.image.random_crop(image, [size[0], size[0], c])
        return tf.image.resize(image, (w, h))
    else:
        return image


def augment(image):
    augm = random_crop(image)
    augm = random_flips(augm)
    augm = random_illumination(augm)
    return augm


def augment_multiple(crops, parallel=None):
    return tf.map_fn(augment, crops, parallel_iterations=parallel, dtype=tf.float32)


# A layer for GPU accelerated augmentations
class AugmentationLayer(tf.compat.v1.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AugmentationLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        return

    def call(self, input_tensor):
        training = tf.compat.v1.keras.backend.learning_phase()
        if training:
            input_tensor = augment_multiple(input_tensor)
            return input_tensor
        else:
            return input_tensor


class AugmentationLayerV2(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AugmentationLayerV2, self).__init__(**kwargs)

    def build(self, input_shape):
        return

    def call(self, input_tensor, training=None):
        if training:
            return augment_multiple(input_tensor)
        else:
            return input_tensor
