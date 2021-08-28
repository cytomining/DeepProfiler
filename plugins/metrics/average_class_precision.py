import tensorflow as tf
from deepprofiler.learning.metric import Metric

tf.compat.v1.disable_v2_behavior()

class MetricClass(Metric):

    def create_metric(self):
        def metric_func(y_true, y_pred):
            return self.metric(y_true, y_pred)
        metric_func.__name__ = "average_class_precision"
        self.f = metric_func
        
    def metric(self, y_true, y_pred):
        result = 0
        single_class_prec = [tf.keras.metrics.Precision(class_id=cls) for cls in range(self.config["num_classes"])]
        for cls_prec in single_class_prec:
            result += cls_prec(y_true, y_pred)
        return result / len(single_class_prec) 
