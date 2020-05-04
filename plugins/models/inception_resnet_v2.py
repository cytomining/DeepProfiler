from comet_ml import Experiment
from keras.applications import inception_resnet_v2
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam
import tensorflow as tf

from deepprofiler.learning.model import DeepProfilerModel


def define_model(config, dset):
   
    # Load InceptionResnetV2 base architecture
    if config["profile"]["use_pretrained_input_size"]:
        input_tensor = Input((299, 299, 3), name="input")
        model = inception_resnet_v2.InceptionResNetV2(
            include_top=True,
            input_tensor=input_tensor,
            weights='imagenet',
            pooling="avg"
        )
        model.summary()
    else:
        input_tensor = Input((
            config["dataset"]["locations"]["box_size"],  # height
            config["dataset"]["locations"]["box_size"],  # width
            len(config["dataset"]["images"]["channels"])  # channels
        ), name="input")
        base = inception_resnet_v2.InceptionResNetV2(
            include_top=False,
            weights=None,
            input_tensor=input_tensor,
            pooling="avg",
            classes=dset.targets[0].shape[1]
        )
        # Create output embedding for each target
        class_outputs = []
        i = 0
        for t in dset.targets:
            y = Dense(t.shape[1], activation="softmax", name=t.field_name)(base.output)
            class_outputs.append(y)
            i += 1
        # Define model
        model = Model(input_tensor, class_outputs)

    # Define optimizer and loss
    optimizer = Adam(lr=config["train"]["model"]["params"]["learning_rate"])
    loss = "categorical_crossentropy"

    return model, optimizer, loss


class ModelClass(DeepProfilerModel):
    def __init__(self, config, dset, generator, val_generator):
        super(ModelClass, self).__init__(config, dset, generator, val_generator)
        self.feature_model, self.optimizer, self.loss = define_model(config, dset)
