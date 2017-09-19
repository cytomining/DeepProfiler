import tensorflow as tf


# TODO: Move metrics to some other module
class Metrics(object):

    def __init__(self, k=2):
        self.correct = 0.0
        self.in_top_k = 0.0
        self.counts = 0.0
        self.with_k = k

    def update(self, corr, top_k, counts):
        self.correct += corr
        self.in_top_k += top_k
        self.counts += counts

    def print(self, with_k):
        message = "Acc: {:0.4f} Top-{}: {:0.4f} Samples: {:0.0f}"
        acc = self.correct/self.counts
        top_k = self.in_top_k / self.counts
        print(message.format(acc, with_k, top_k, self.counts))

    def configure_ops(self, batch_size, num_classes):
        self.true_labels = tf.placeholder(tf.int32, shape=(batch_size))
        self.predictions = tf.placeholder(tf.float32, shape=(batch_size, num_classes))
        one_hot_true_labels = tf.one_hot(self.true_labels, num_classes)

        self.correct_op = tf.reduce_sum(
            tf.to_float(
                tf.equal(
                    tf.argmax(one_hot_true_labels, 1),
                    tf.argmax(self.predictions, 1)
                )
            )
        )
        self.in_top_k_op = tf.reduce_sum(
            tf.to_float(
                tf.nn.in_top_k(
                    predictions=self.predictions,
                    targets=tf.argmax(one_hot_true_labels, 1),
                    k=self.with_k
                )
            )
        )
