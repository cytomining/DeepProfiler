import tensorflow as tf

from deepprofiler.learning.model import DeepProfilerModel
from deepprofiler.imaging.augmentations import AugmentationLayer

tf.compat.v1.disable_v2_behavior()


class ModelClass(DeepProfilerModel):
    def __init__(self, config, dset, generator, val_generator, is_training):
        super(ModelClass, self).__init__(config, dset, generator, val_generator, is_training)
        self.feature_model, self.optimizer, self.loss = self.define_model(config, dset)

    ## Define supported models
    def get_supported_models(self):
        return {
            0: tf.compat.v1.keras.applications.EfficientNetB0,
            1: tf.compat.v1.keras.applications.EfficientNetB1,
            2: tf.compat.v1.keras.applications.EfficientNetB2,
            3: tf.compat.v1.keras.applications.EfficientNetB3,
            4: tf.compat.v1.keras.applications.EfficientNetB4,
            5: tf.compat.v1.keras.applications.EfficientNetB5,
            6: tf.compat.v1.keras.applications.EfficientNetB6,
            7: tf.compat.v1.keras.applications.EfficientNetB7,
        }

    def get_model(self, config, input_image=None, weights=None, include_top=False):
        supported_models = self.get_supported_models()
        SM = "EfficientNet supported models: " + ",".join([str(x) for x in supported_models.keys()])
        num_layers = config["train"]["model"]["params"]["conv_blocks"]
        error_msg = str(num_layers) + " conv_blocks not in " + SM
        assert num_layers in supported_models.keys(), error_msg

        if self.is_training and weights is None:
            input_image = AugmentationLayer()(input_image)

        model = supported_models[num_layers](
            input_tensor=input_image,
            include_top=include_top,
            weights=weights
        )
        return model

    def define_model(self, config, dset):
        supported_models = self.get_supported_models()
        SM = "EfficientNet supported models: " + ",".join([str(x) for x in supported_models.keys()])
        num_layers = config["train"]["model"]["params"]["conv_blocks"]
        error_msg = str(num_layers) + " conv_blocks not in " + SM
        assert num_layers in supported_models.keys(), error_msg
        # Set session

        optimizer = tf.compat.v1.keras.optimizers.SGD(lr=config["train"]["model"]["params"]["learning_rate"], momentum=0.9,
                                         nesterov=True)
        loss_func = "categorical_crossentropy"
        if self.is_training is False and "use_pretrained_input_size" in config["profile"].keys():
            input_tensor = tf.compat.v1.keras.layers.Input(
                (config["profile"]["use_pretrained_input_size"], config["profile"]["use_pretrained_input_size"], 3),
                name="input")
            model = self.get_model(config, input_image=input_tensor, weights='imagenet', include_top=True)
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

            i = 0
            for t in dset.targets:
                y = tf.compat.v1.keras.layers.Dense(t.shape[1], activation="softmax", name=t.field_name)(features)
                class_outputs.append(y)
                i += 1

            # 4. Create and compile model
            model = tf.compat.v1.keras.models.Model(inputs=input_image, outputs=class_outputs)


            ## Added weight decay following tricks reported in:
            ## https://github.com/keras-team/keras/issues/2717
            regularizer = tf.compat.v1.keras.regularizers.l2(0.00001)
            for layer in model.layers:
                if hasattr(layer, "kernel_regularizer"):
                    setattr(layer, "kernel_regularizer", regularizer)

            model = tf.compat.v1.keras.models.model_from_json(
                model.to_json(),
                {'AugmentationLayer': AugmentationLayer}
            )

        return model, optimizer, loss_func

    def copy_pretrained_weights(self):
        base_model = self.get_model(self.config, weights="imagenet")
        lshift = self.is_training  # Shift one layer to accommodate the AugmentationLayer

        # => Transfer all weights except conv1.1
        total_layers = len(base_model.layers)
        for i in range(3, total_layers):
            if len(base_model.layers[i].weights) > 0:
                print("Setting pre-trained weights: {:.2f}%".format((i / total_layers) * 100), end="\r")
                self.feature_model.layers[i + lshift].set_weights(base_model.layers[i].get_weights())

        print("Network initialized with pretrained ImageNet weights")
