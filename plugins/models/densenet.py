import tensorflow as tf

from plugins.models import resnet

tf.compat.v1.disable_v2_behavior()

##################################################
# DenseNet architecture as in "Densely Connected 
# Convolutional Networks" by Gao Huang, Zhuang Liu, 
# Laurens van der Maaten, Kilian Q. Weinberger
# https://arxiv.org/pdf/1608.06993.pdf
##################################################

class ModelClass(resnet.ModelClass):
    def __init__(self, config, dset, generator, val_generator, is_training):
        super(ModelClass, self).__init__(config, dset, generator, val_generator, is_training)
        self.feature_model, self.optimizer, self.loss = super().define_model(config, dset)


    def get_supported_models(self):
        return {
            121: tf.compat.v1.keras.applications.DenseNet121,
            169: tf.compat.v1.keras.applications.DenseNet169,
            201: tf.compat.v1.keras.applications.DenseNet201
        }

