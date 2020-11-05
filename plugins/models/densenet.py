import keras
import keras.applications

from deepprofiler.imaging.augmentations import AugmentationLayer
from deepprofiler.learning.model import DeepProfilerModel

##################################################
# DenseNet architecture as in "Densely Connected 
# Convolutional Networks" by Gao Huang, Zhuang Liu, 
# Laurens van der Maaten, Kilian Q. Weinberger
# https://arxiv.org/pdf/1608.06993.pdf
##################################################

class ModelClass(DeepProfilerModel):
    def __init__(self, config, dset, generator, val_generator, is_training):
        super(ModelClass, self).__init__(config, dset, generator, val_generator, is_training)
        self.feature_model, self.optimizer, self.loss = self.define_model(config, dset)


    def get_supported_models(self):
        return {
            121: keras.applications.DenseNet121,
            169: keras.applications.DenseNet169,
            201: keras.applications.DenseNet201
        }

    # Load a supported model
    def get_model(self, config, input_image=None, weights=None, pooling=None, include_top=False):
        supported_models = self.get_supported_models()
        SM = "ResNet supported models: " + ",".join([str(x) for x in supported_models.keys()])
        num_layers = config["train"]["model"]["params"]["conv_blocks"]
        error_msg = str(num_layers) + " conv_blocks not in " + SM
        assert num_layers in supported_models.keys(), error_msg
        if self.is_training and weights is None:
            input_image = AugmentationLayer()(input_image)
        if pooling is not None:
            model = supported_models[num_layers](input_tensor=input_image, pooling=pooling, include_top=include_top,
                                                     weights=weights)
        else:
            model = supported_models[num_layers](input_tensor=input_image, include_top=include_top, weights=weights)
        return model


    def define_model(self, config, dset):
        supported_models = self.get_supported_models()
        SM = "DenseNet supported models: " + ",".join([str(x) for x in supported_models.keys()])
        # 1. Create ResNet architecture to extract features
        input_shape = (
            config["train"]["sampling"]["box_size"],  # height
            config["train"]["sampling"]["box_size"],  # width
            len(config["dataset"]["images"]["channels"])  # channels
        )
        input_image = keras.layers.Input(input_shape)

        num_layers = config["train"]["model"]["params"]["conv_blocks"]
        error_msg = str(num_layers) + " conv_blocks not in " + SM
        assert num_layers in supported_models.keys(), error_msg

        pooling = config["train"]["model"]["params"]["pooling"]
        model = supported_models[num_layers](input_tensor=input_image,
                                            include_top=False,
                                            pooling=pooling,
                                            weights=None,
                                            input_shape=input_shape,
                                            classes=dset.targets[0].shape[1])

        features = keras.layers.GlobalAveragePooling2D(name="pool5")(model.layers[-2].output)

        # 2. Create an output embedding for each target
        class_outputs = []

        i = 0
        for t in dset.targets:
            y = keras.layers.Dense(t.shape[1], activation="softmax", name=t.field_name)(features)
            class_outputs.append(y)
            i += 1

        # 3. Define the loss function
        loss_func = "categorical_crossentropy"

        # 4. Create and compile model
        model = keras.models.Model(input_image, class_outputs)
        model = keras.models.model_from_json(model.to_json())
        optimizer = keras.optimizers.SGD(lr=config["train"]["model"]["params"]["learning_rate"], momentum=0.9, nesterov=True)

        return model, optimizer, loss_func

    # TODO: copy_pretrained_weights
