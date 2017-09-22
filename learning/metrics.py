import tensorflow as tf


# TODO: Move metrics to some other module
class Metrics(object):

    def __init__(self, k=2, name=""):
        self.correct = 0.0
        self.in_top_k = 0.0
        self.counts = 0.0
        self.with_k = k
        self.name = name
        self.cmatrix = None

    def update(self, values, counts):
        self.correct += values[0]
        self.in_top_k += values[1]
        self.counts += counts
        if self.cmatrix is None: 
            self.cmatrix = values[2]
        else:
            self.cmatrix += values[2]

    def result_string(self):
        print(self.cmatrix)
        message = self.name + "=[Acc: {:0.4f} Top-{}: {:0.4f} Samples: {:0.0f}]"
        acc = self.correct/self.counts
        top_k = self.in_top_k / self.counts
        return message.format(acc, self.with_k, top_k, self.counts)

    def configure_ops(self, batch_size, label_dict):
        self.label_dict = label_dict
        num_classes = len(label_dict)
        with tf.name_scope(self.name + "_metric"):
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
            self.confussion_matrix = tf.contrib.metrics.confusion_matrix(
                self.true_labels,
                tf.argmax(self.predictions, 1),
                num_classes=num_classes
            )

    def get_ops(self):
        return [self.correct_op, self.in_top_k_op, self.confussion_matrix]

    def set_inputs(self, batch_labels, batch_predictions):
        # Transform labels to class indices
        for i in range(len(batch_labels)):
            batch_labels[i] = self.label_dict[batch_labels[i]]
        feed_dict={
                     self.true_labels:batch_labels,
                     self.predictions:batch_predictions
                  }
        return feed_dict

