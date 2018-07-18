from comet_ml import Experiment
from keras.applications import inception_resnet_v2
from keras.layers import Input
from keras.optimizers import Adam

from deepprofiler.learning.model import DeepProfilerModel


def define_model(config, dset):
    # Load InceptionResnetV2 architecture
    if config["model"]["pretrained"]:
        weights = "imagenet"
    else:
        weights = None
    model = inception_resnet_v2.InceptionResNetV2(
        weights=weights,
        input_tensor=Input((
            config["sampling"]["box_size"],  # height
            config["sampling"]["box_size"],  # width
            len(config["image_set"]["channels"])  # channels
        ), name='input'),
        classes=dset.targets[0].shape[1]  # TODO: support multiple targets
    )

    # Optimizer and loss
    optimizer = Adam(lr=config['model']['params']['learning_rate'])
    loss = 'categorical_crossentropy'

    return model, optimizer, loss


class ModelClass(DeepProfilerModel):
    def __init__(self, config, dset, generator, val_generator):
        super(ModelClass, self).__init__(config, dset, generator, val_generator)
        self.model, self.optimizer, self.loss = define_model(config, dset)
