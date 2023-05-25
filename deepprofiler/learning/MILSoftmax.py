# Code adopted from https://github.com/utayao/Atten_Deep_MIL/blob/master/utl/custom_layers.py
import tensorflow as tf


class MILSoftmax(tf.compat.v1.keras.layers.Layer):
    def __init__(self, output_dim, kernel_initializer='glorot_uniform', bias_initializer='zeros',
                 kernel_regularizer=None, bias_regularizer=None, use_bias=True, **kwargs):
        self.output_dim = output_dim

        self.kernel_initializer = tf.compat.v1.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.compat.v1.keras.initializers.get(bias_initializer)
        self.kernel_regularizer = tf.compat.v1.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.compat.v1.keras.regularizers.get(bias_regularizer)

        self.use_bias = use_bias
        super(MILSoftmax, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        input_dim = input_shape[2]

        self.kernel = self.add_weight(shape=(input_dim, self.output_dim),
                                        initializer=self.kernel_initializer,
                                        name='kernel',
                                        regularizer=self.kernel_regularizer)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.output_dim,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer)
        else:
            self.bias = None

        self.input_built = True

    def call(self, x, mask=None):
        _, n, d = x.shape
        x = tf.compat.v1.keras.backend.sum(x, axis=1, keepdims=True)
        x = tf.compat.v1.squeeze(x, axis=1)
        print('x after sum', x.shape)
        # compute instance-level score
        x = tf.compat.v1.keras.backend.dot(x, self.kernel)
        print('x after dot', x.shape)
        if self.use_bias:
            x = tf.compat.v1.keras.backend.bias_add(x, self.bias)

        out = tf.compat.v1.keras.backend.softmax(x)
        return out

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        assert len(shape) == 3
        shape[2] = self.output_dim
        return tuple(shape)

    def get_config(self):
        config = {
            'output_dim': self.output_dim,
            'use_bias': self.use_bias
        }
        base_config = super(MILSoftmax, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
