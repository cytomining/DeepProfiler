import numpy
import keras
import efficientnet.keras as efn

from deepprofiler.learning.model import DeepProfilerModel

class ModelClass(DeepProfilerModel):
    def __init__(self, config, dset, generator, val_generator):
        super(ModelClass, self).__init__(config, dset, generator, val_generator)
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

    def get_model(self, config, input_image=None, weights=None):
        supported_models = self.get_supported_models()
        SM = "EfficientNet supported models: " + ",".join([str(x) for x in supported_models.keys()])
        num_layers = config["train"]["model"]["params"]["conv_blocks"]
        error_msg = str(num_layers) + " conv_blocks not in " + SM
        assert num_layers in supported_models.keys(), error_msg

        model = supported_models[num_layers](input_tensor=input_image, include_top=False, weights=weights)
        return model

    def define_model(self, config, dset):
        # Set session
        input_shape = (
            config["dataset"]["locations"]["box_size"],  # height
            config["dataset"]["locations"]["box_size"],  # width
            len(config["dataset"]["images"]["channels"])  # channels
        )
        input_image = keras.layers.Input(input_shape)
        model = self.get_model(config, input_image=input_image)
        features = keras.layers.GlobalAveragePooling2D(name="pool5")(model.layers[-1].output)

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
        model = keras.models.Model(inputs=input_image, outputs=class_outputs)

        model = keras.models.model_from_json(model.to_json())
        optimizer = keras.optimizers.SGD(lr=config["train"]["model"]["params"]["learning_rate"], momentum=0.9, nesterov=True)

        return model, optimizer, loss_func

    ## Support for ImageNet initialization
    def copy_pretrained_weights(self):
        base_model = self.get_model(self.config, weights="imagenet")
        # => Transfer all weights except conv1.1
        total_layers = len(base_model.layers)
        for i in range(3, total_layers):
            if len(base_model.layers[i].weights) > 0:
                print("Setting pre-trained weights: {:.2f}%".format((i / total_layers) * 100), end="\r")
                self.feature_model.layers[i].set_weights(base_model.layers[i].get_weights())

        # => Replicate filters of first layer as needed
        weights = base_model.layers[2].get_weights()
        available_channels = weights[0].shape[2]
        target_shape = self.feature_model.layers[2].weights[0].shape
        new_weights = numpy.zeros(target_shape)
        for i in range(new_weights.shape[2]):
            j = i % available_channels
            new_weights[:, :, i, :] = weights[0][:, :, j, :]
            weights_array = [new_weights]
            if len(weights) > 1:
                weights_array += weights[1:]
            self.feature_model.layers[2].set_weights(weights_array)
        print("Network initialized with pretrained ImageNet weights")