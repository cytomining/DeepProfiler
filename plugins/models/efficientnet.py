import tensorflow as tf
import numpy
import inspect
import efficientnet.tfkeras as efn

from deepprofiler.imaging.augmentations import AugmentationLayer
from deepprofiler.imaging.augmentations import AugmentationLayerV2
from deepprofiler.learning.model import DeepProfilerModel
from deepprofiler.learning.tf2train import DeepProfilerModelV2


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
                0: efn.EfficientNetB0,
                1: efn.EfficientNetB1,
                2: efn.EfficientNetB2,
                3: efn.EfficientNetB3,
                4: efn.EfficientNetB4,
                5: efn.EfficientNetB5,
                6: efn.EfficientNetB6,
                7: efn.EfficientNetB7,
            }

        def get_model(self, config, input_image=None, weights=None, include_top=False):
            supported_models = self.get_supported_models()
            SM = "EfficientNet supported models: " + ",".join([str(x) for x in supported_models.keys()])
            num_layers = config["train"]["model"]["params"]["conv_blocks"]
            error_msg = str(num_layers) + " conv_blocks not in " + SM
            assert num_layers in supported_models.keys(), error_msg

            # Rescaling images in full_image mode
            if self.is_training and config["dataset"]["locations"]["mode"] == "full_image" and input_image is not None:
                input_view = input_image
                crop_size = (config["dataset"]["locations"]["box_size"], config["dataset"]["locations"]["box_size"])
                boxes = tf.compat.v1.keras.layers.Input(shape=(4,))
                box_ind = tf.compat.v1.keras.layers.Input(shape=(), dtype="int32")
                input_image = tf.image.crop_and_resize(input_view, boxes, box_ind, crop_size)

            # Adding augmentations
            if self.is_training and weights is None and self.config["train"]['model'].get('augmentations') is True:
                input_image = AugmentationLayer()(input_image)

            model = supported_models[num_layers](
                input_tensor=input_image,
                include_top=include_top,
                weights=weights
            )

            # Enable multiple inputs if in full_image mode
            if self.is_training and config["dataset"]["locations"]["mode"] == "full_image" and input_image is not None:
                model = tf.compat.v1.keras.Model(inputs=[input_view, boxes, box_ind], outputs=model.output)

            return model

        def define_model(self, config, dset):
            supported_models = self.get_supported_models()
            SM = "EfficientNet supported models: " + ",".join([str(x) for x in supported_models.keys()])
            num_layers = config["train"]["model"]["params"]["conv_blocks"]
            error_msg = str(num_layers) + " conv_blocks not in " + SM
            assert num_layers in supported_models.keys(), error_msg
            # Set session

            optimizer = tf.compat.v1.keras.optimizers.SGD(lr=config["train"]["model"]["params"]["learning_rate"],
                                                                momentum=0.9, nesterov=True)
            loss_func = tf.compat.v1.keras.losses.CategoricalCrossentropy(label_smoothing=
                                                                self.config["train"]["model"]["params"]["label_smoothing"])

            if not self.is_training and "use_pretrained_input_size" in config["profile"].keys():
                input_tensor = tf.compat.v1.keras.layers.Input(
                    (config["profile"]["use_pretrained_input_size"], config["profile"]["use_pretrained_input_size"], 3),
                    name="input")
                model = self.get_model(config, input_image=input_tensor, weights='imagenet', include_top=True)
            elif self.is_training or "use_pretrained_input_size" not in config["profile"].keys():
                if self.is_training and config["dataset"]["locations"]["mode"] == "full_image":
                    width = height = config["dataset"]["locations"]["view_size"]
                else:
                    width = height = config["dataset"]["locations"]["box_size"]

                input_shape = (height, width, len(config["dataset"]["images"]["channels"]))
                input_image = tf.compat.v1.keras.layers.Input(input_shape)

                    
                model = self.get_model(config, input_image=input_image)
                features = tf.compat.v1.keras.layers.GlobalAveragePooling2D(name="pool5")(model.layers[-1].output)
                # 2. Create an output embedding for each target
                class_outputs = []

                y = tf.compat.v1.keras.layers.Dense(self.config["num_classes"], activation="softmax", name="ClassProb")(features)
                class_outputs.append(y)

                # 4. Create and compile model
                model = tf.compat.v1.keras.models.Model(inputs=model.input, outputs=class_outputs)

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

        def copy_pretrained_weights(self):
            base_model = self.get_model(self.config, weights="imagenet")
            # Shift layers to accommodate crop_and_resize + AugmentationLayer
            layers = self.feature_model.layers
            lshift = 0
            if layers[1].name == 'augmentation_layer_1' or layers[4].name == "augmentation_layer_1":
                lshift += 1
            if layers[1].name == 'input_2' and layers[2].name == 'input_3':
                lshift += 3

            # => Transfer all weights except conv1.1
            total_layers = len(base_model.layers)
            for i in range(2, total_layers):
                if len(base_model.layers[i].weights) > 0:
                    print("Setting pre-trained weights: {:.2f}%".format((i / total_layers) * 100), end="\r")
                    self.feature_model.layers[i + lshift].set_weights(base_model.layers[i].get_weights())

            # => Replicate filters of first layer as needed
            weights = base_model.layers[1].get_weights()
            available_channels = weights[0].shape[2]
            target_shape = self.feature_model.layers[1 + lshift].weights[0].shape
            new_weights = numpy.zeros(target_shape)

            for i in range(new_weights.shape[2]):
                j = i % available_channels
                new_weights[:, :, i, :] = weights[0][:, :, j, :]

            weights_array = [new_weights]
            if len(weights) > 1:
                weights_array += weights[1:]

            self.feature_model.layers[1 + lshift].set_weights(weights_array)
            print("Network initialized with pretrained ImageNet weights")

    return ModelClass(config, dset, crop_generator, val_crop_generator, is_training)
