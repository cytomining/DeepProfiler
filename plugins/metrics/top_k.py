from keras.metrics import top_k_categorical_accuracy

from deepprofiler.learning.metric import Metric

class MetricClass(Metric):

    def metric(self, y_true, y_pred):
        return top_k_categorical_accuracy(y_true, y_pred, k=self.config['validation']['top_k'])