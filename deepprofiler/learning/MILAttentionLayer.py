# Code adopted from https://github.com/utayao/Atten_Deep_MIL/blob/master/utl/custom_layers.py
import tensorflow as tf


class MILAttentionLayer(tf.compat.v1.keras.layers.Layer):
    """Implementation of the attention-based Deep MIL layer.

    Args:
      weight_params_dim: Positive Integer. Dimension of the weight matrix.
      kernel_initializer: Initializer for the `kernel` matrix.
      kernel_regularizer: Regularizer function applied to the `kernel` matrix.
      use_gated: Boolean, whether or not to use the gated mechanism.

    Returns:
      List of 2D tensors with BAG_SIZE length.
      The tensors are the attention scores after softmax with shape `(batch_size, 1)`.
    """

    def __init__(self, weight_params_dim=256, output_dim=1, kernel_initializer='glorot_uniform',
                 kernel_regularizer=None, use_bias=True, use_gated=False, **kwargs):
        self.weight_params_dim = weight_params_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.use_gated = use_gated

        self.v_init = tf.compat.v1.keras.initializers.get(kernel_initializer)
        self.w_init = tf.compat.v1.keras.initializers.get(kernel_initializer)
        self.u_init = tf.compat.v1.keras.initializers.get(kernel_initializer)

        self.v_regularizer = tf.compat.v1.keras.regularizers.get(kernel_regularizer)
        self.w_regularizer = tf.compat.v1.keras.regularizers.get(kernel_regularizer)
        self.u_regularizer = tf.compat.v1.keras.regularizers.get(kernel_regularizer)

        super(MILAttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        assert len(input_shape) == 2
        print(input_shape)
        input_dim = input_shape[1]

        self.V = self.add_weight(shape=(input_dim, self.weight_params_dim),
                                      initializer=self.v_init,
                                      name='v',
                                      regularizer=self.v_regularizer,
                                      trainable=True)

        self.w = self.add_weight(shape=(self.weight_params_dim, 1),
                                    initializer=self.w_init,
                                    name='w',
                                    regularizer=self.w_regularizer,
                                    trainable=True)

        if self.use_gated:
            self.U = self.add_weight(shape=(input_dim, self.weight_params_dim),
                                     initializer=self.u_init,
                                     name='U',
                                     regularizer=self.u_regularizer,
                                     trainable=True)
        else:
            self.U = None

        self.input_built = True

    def call(self, x, mask=None):
        n, d = x.shape
        ori_x = x
        # do Vhk^T
        x = tf.math.tanh(tf.tensordot(x, self.V, axes=1)) # (2,64)

        if self.use_gated:
            gate_x = tf.math.sigmoid(tf.tensordot(ori_x, self.U, axes=1))
            ac_x = x * gate_x
        else:
            ac_x = x

        # do w^T x
        soft_x = tf.tensordot(ac_x, self.w, axes=1)  # (2,64) * (64, 1) = (2,1)
        alpha = tf.math.softmax(tf.transpose(soft_x)) # (2,1)
        alpha = tf.transpose(alpha)
        print('Attention output shape', alpha.shape)
        return alpha

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        assert len(shape) == 2
        shape[1] = self.output_dim
        return tuple(shape)

    def get_config(self):
        cfg = super().get_config()
        return cfg
