import tensorflow as tf


@tf.custom_gradient
def grad_reverse(x):
    forward = tf.identity(x)

    def custom_grad(backward, alpha=-1.0):
        return alpha*backward
    return forward, custom_grad


class GradientReversal(tf.compat.v1.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(GradientReversal, self).__init__(**kwargs)

    def build(self, input_shape):
        return

    def call(self, input_tensor):
        return grad_reverse(input_tensor)
