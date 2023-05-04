# Code from https://keras.io/examples/vision/attention_mil_classification/

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

    def __init__(
        self,
        weight_params_dim=256,
        kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        use_gated=True,
        **kwargs
    ):

        super().__init__(**kwargs)

        self.weight_params_dim = weight_params_dim
        self.use_gated = use_gated

        self.kernel_initializer = tf.compat.v1.keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = tf.compat.v1.keras.regularizers.get(kernel_regularizer)

        self.v_init = self.kernel_initializer
        self.w_init = self.kernel_initializer
        self.u_init = self.kernel_initializer

        self.v_regularizer = self.kernel_regularizer
        self.w_regularizer = self.kernel_regularizer
        self.u_regularizer = self.kernel_regularizer

    def build(self, input_shape):

        # Input shape.
        # List of 2D tensors with shape: (batch_size, input_dim).
        input_dim = input_shape[1]

        self.v_weight_params = self.add_weight(
            shape=(input_dim, self.weight_params_dim),
            initializer=self.v_init,
            name="v",
            regularizer=self.v_regularizer,
            trainable=True,
        )

        self.w_weight_params = self.add_weight(
            shape=(self.weight_params_dim, 1),
            initializer=self.w_init,
            name="w",
            regularizer=self.w_regularizer,
            trainable=True,
        )

        if self.use_gated:
            self.u_weight_params = self.add_weight(
                shape=(input_dim, self.weight_params_dim),
                initializer=self.u_init,
                name="u",
                regularizer=self.u_regularizer,
                trainable=True,
            )
        else:
            self.u_weight_params = None

        self.input_built = True

    def call(self, inputs):

        n, d = inputs.shape
        ori_x = inputs
        # do Vhk^T
        x = tf.math.tanh(tf.tensordot(inputs, self.v_weight_params, axes=1))  # (2,64)

        if self.use_gated:
            gate_x = tf.math.sigmoid(tf.tensordot(ori_x, self.u_weight_params, axes=1))
            ac_x = x * gate_x
        else:
            ac_x = inputs

        # do w^T x
        soft_x = tf.tensordot(ac_x, self.w_weight_params, axes=1)  # (2,64) * (64, 1) = (2,1)
        alpha = tf.math.softmax(tf.transpose(soft_x))  # (2,1)
        alpha = tf.transpose(alpha)
        return alpha

        # Assigning variables from the number of inputs.
        #instances = [self.compute_attention_scores(instance) for instance in inputs]

        # Apply softmax over instances such that the output summation is equal to 1.
        #alpha = tf.math.softmax(instances, axis=0)

        #return [alpha[i] for i in range(alpha.shape[0])]

    # def compute_attention_scores(self, instance):
    #
    #     # Reserve in-case "gated mechanism" used.
    #     original_instance = instance
    #
    #     # tanh(v*h_k^T)
    #     instance = tf.math.tanh(tf.tensordot(instance, self.v_weight_params, axes=1))
    #
    #     # for learning non-linear relations efficiently.
    #     if self.use_gated:
    #
    #         instance = instance * tf.math.sigmoid(
    #             tf.tensordot(original_instance, self.u_weight_params, axes=1)
    #         )
    #
    #     # w^T*(tanh(v*h_k^T)) / w^T*(tanh(v*h_k^T)*sigmoid(u*h_k^T))
    #     return tf.tensordot(instance, self.w_weight_params, axes=1)

    def get_config(self):
        cfg = super().get_config()
        return cfg
