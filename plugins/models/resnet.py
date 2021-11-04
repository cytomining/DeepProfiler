import numpy as np
import tensorflow as tf
import inspect

from deepprofiler.learning.model import DeepProfilerModel
from deepprofiler.learning.tf2train import DeepProfilerModelV2
from deepprofiler.imaging.augmentations import AugmentationLayer
from deepprofiler.imaging.augmentations import AugmentationLayerV2


##################################################
# ResNet architecture as defined in "Identity Mappings
# in Deep Residual Networks" by Kaiming He,
# Xiangyu Zhang, Shaoqing Ren, Jian Sun
# https://arxiv.org/abs/1603.05027
##################################################


def model_factory(config, dset, crop_generator, val_crop_generator, is_training):
    if inspect.currentframe().f_back.f_code.co_name == 'learn_model_v2':
        tf.compat.v1.enable_v2_behavior()
        tf.config.run_functions_eagerly(True)
        augmentation_base = AugmentationLayerV2()
        return createModelClass(DeepProfilerModelV2, config, dset, crop_generator,
                                val_crop_generator, is_training, augmentation_base)
    else:
        augmentation_base = AugmentationLayer()
        return createModelClass(DeepProfilerModel, config, dset, crop_generator,
                                val_crop_generator, is_training, augmentation_base)


def createModelClass(base, config, dset, crop_generator, val_crop_generator, is_training, augmentation_base):
    class ModelClass(base):
        def __init__(self, config, dset, crop_generator, val_crop_generator, is_training):
            super(ModelClass, self).__init__(config, dset, crop_generator, val_crop_generator, is_training)
            self.feature_model, self.optimizer, self.loss = self.define_model(config, dset)

        ## Define supported models
        def get_supported_models(self):
            return {
                50: tf.compat.v1.keras.applications.resnet_v2.ResNet50V2,
                101: tf.compat.v1.keras.applications.resnet_v2.ResNet101V2,
                152: tf.compat.v1.keras.applications.resnet_v2.ResNet152V2,
            }

        ## Load a supported model
        def get_model(self, config, input_image=None, weights=None, pooling=None, include_top=False):
            supported_models = self.get_supported_models()
            SM = "ResNet supported models: " + ",".join([str(x) for x in supported_models.keys()])
            num_layers = config["train"]["model"]["params"]["conv_blocks"]
            error_msg = str(num_layers) + " conv_blocks not in " + SM
            assert num_layers in supported_models.keys(), error_msg
            if self.is_training and weights is None and self.config["train"]['model'].get('augmentations') is True:
                input_image = AugmentationLayer()(input_image)
            if pooling is not None:
                model = supported_models[num_layers](input_tensor=input_image, pooling=pooling, include_top=include_top,
                                                     weights=weights)
            else:
                model = supported_models[num_layers](input_tensor=input_image, include_top=include_top, weights=weights)
            return model

        ## Model definition
        def define_model(self, config, dset):
            # 1. Create ResNet architecture to extract features
            loss_func = tf.compat.v1.keras.losses.CategoricalCrossentropy(label_smoothing=
                                                                          self.config["train"]["model"]["params"][
                                                                              "label_smoothing"])
            optimizer = tf.compat.v1.keras.optimizers.SGD(learning_rate=config["train"]["model"]["params"]["learning_rate"],
                                                          momentum=0.9, nesterov=True)
            if "use_pretrained_input_size" in config["profile"].keys() and self.is_training is False:
                input_tensor = tf.compat.v1.keras.layers.Input(
                    (config["profile"]["use_pretrained_input_size"], config["profile"]["use_pretrained_input_size"], 3),
                    name="input")
                model = self.get_model(
                    config,
                    input_image=input_tensor,
                    weights='imagenet',
                    pooling="avg",
                    include_top=True
                )
                model.summary()

            elif self.is_training is True or "use_pretrained_input_size" not in config["profile"].keys():
                input_shape = (
                    config["dataset"]["locations"]["box_size"],  # height
                    config["dataset"]["locations"]["box_size"],  # width
                    len(config["dataset"]["images"][
                            "channels"])  # channels
                )
                input_image = tf.compat.v1.keras.layers.Input(input_shape)
                model = self.get_model(config, input_image=input_image)
                features = tf.compat.v1.keras.layers.GlobalAveragePooling2D(name="pool5")(model.layers[-1].output)

                # 2. Create an output embedding for each target
                class_outputs = []

                y = tf.compat.v1.keras.layers.Dense(self.config["num_classes"], activation="softmax", name="ClassProb")(
                    features)
                class_outputs.append(y)
                # 4. Create and compile model
                model = tf.compat.v1.keras.models.Model(inputs=input_image, outputs=class_outputs)

                ## Added weight decay following tricks reported in:
                ## https://github.com/keras-team/keras/issues/2717
                regularizer = tf.compat.v1.keras.regularizers.l2(0.00001)
                for layer in model.layers:
                    if hasattr(layer, "kernel_regularizer"):
                        setattr(layer, "kernel_regularizer", regularizer)

                if self.config["train"]["model"].get("augmentations") is True:
                    model = tf.compat.v1.keras.models.model_from_json(
                        model.to_json(),
                        {'AugmentationLayer': augmentation_base}
                    )
                else:
                    model = tf.compat.v1.keras.models.model_from_json(model.to_json())


            return model, optimizer, loss_func


        ## Support for ImageNet initialization
        def copy_pretrained_weights(self):
            base_model = self.get_model(self.config, weights="imagenet")
            lshift = self.feature_model.layers[1].name == 'augmentation_layer_1'  # Shift one layer to accommodate the AugmentationLayer

            # => Transfer all weights except conv1.1
            total_layers = len(base_model.layers)
            for i in range(3, total_layers):
                if len(base_model.layers[i].weights) > 0:
                    print("Setting pre-trained weights: {:.2f}%".format((i/total_layers)*100), end="\r")
                    self.feature_model.layers[i + lshift].set_weights(base_model.layers[i].get_weights())

            # => Replicate filters of first layer as needed
            weights = base_model.layers[2].get_weights()
            available_channels = weights[0].shape[2]
            target_shape = self.feature_model.layers[2 + lshift].weights[0].shape
            new_weights = np.zeros(target_shape)

            for i in range(new_weights.shape[2]):
                j = i % available_channels
                new_weights[:, :, i, :] = weights[0][:, :, j, :]

            weights_array = [new_weights]
            if len(weights) > 1:
                weights_array += weights[1:]

            self.feature_model.layers[2 + lshift].set_weights(weights_array)
            print("Network initialized with pretrained ImageNet weights")

    return ModelClass(config, dset, crop_generator, val_crop_generator, is_training)
