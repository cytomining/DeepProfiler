import inspect

import tensorflow as tf

from plugins.models import resnet


##################################################
# DenseNet architecture as in "Densely Connected
# Convolutional Networks" by Gao Huang, Zhuang Liu,
# Laurens van der Maaten, Kilian Q. Weinberger
# https://arxiv.org/pdf/1608.06993.pdf
##################################################


class DenseNetMixin(object):
    def get_supported_models(self):
        return {
            121: tf.keras.applications.DenseNet121,
            169: tf.keras.applications.DenseNet169,
            201: tf.keras.applications.DenseNet201
        }


class ModelClass(DenseNetMixin, resnet.ModelClass):
    pass


class ModelClassV2(DenseNetMixin, resnet.ModelClassV2):
    pass


def model_factory(config, dset, crop_generator, val_crop_generator, is_training):
    if inspect.currentframe().f_back.f_code.co_name == 'learn_model_v2':
        return ModelClassV2(config, dset, crop_generator, val_crop_generator, is_training)
    else:
        return ModelClass(config, dset, crop_generator, val_crop_generator, is_training)
