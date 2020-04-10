import keras 
import keras.applications.resnet_v2
import numpy

from deepprofiler.learning.model import DeepProfilerModel

##################################################
# ResNet architecture as defined in "Identity Mappings 
# in Deep Residual Networks" by Kaiming He,
# Xiangyu Zhang, Shaoqing Ren, Jian Sun
# https://arxiv.org/abs/1603.05027
##################################################


class ModelClass(DeepProfilerModel):
    def __init__(self, config, dset, generator, val_generator):
        super(ModelClass, self).__init__(config, dset, generator, val_generator)
        self.feature_model, self.optimizer, self.loss = self.define_model(config, dset)


    ## Define supported models
    def get_supported_models(self):
        return {
            50: keras.applications.resnet_v2.ResNet50V2,
            101: keras.applications.resnet_v2.ResNet101V2,
            152: keras.applications.resnet_v2.ResNet152V2,
        }
 

    ## Load a supported model
    def get_model(self, config, input_image=None, weights=None):
        supported_models = self.get_supported_models()
        SM = "ResNet supported models: " + ",".join([str(x) for x in supported_models.keys()])
        num_layers = config["train"]["model"]["params"]["conv_blocks"]
        error_msg = str(num_layers) + " conv_blocks not in " + SM
        assert num_layers in supported_models.keys(), error_msg
        
        model = supported_models[num_layers](input_tensor=input_image, include_top=False, weights=weights)
        return model
 

    ## Model definition
    def define_model(self, config, dset):
        # 1. Create ResNet architecture to extract features
        input_shape = (
            config["train"]["sampling"]["box_size"],  # height
            config["train"]["sampling"]["box_size"],  # width
            len(config["dataset"]["images"][
                "channels"])  # channels
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
        ## Added weight decay following tricks reported in:
        ## https://github.com/keras-team/keras/issues/2717
        regularizer = keras.regularizers.l2(0.00001)
        for layer in model.layers:
            if hasattr(layer, "kernel_regularizer"):
                setattr(layer, "kernel_regularizer", regularizer)
        model = keras.models.model_from_json(model.to_json())
        optimizer = keras.optimizers.SGD(lr=config["train"]["model"]["params"]["learning_rate"], momentum=0.9, nesterov=True)

        return model, optimizer, loss_func



    ## Support for ImageNet initialization
    def copy_pretrained_weights(self):
        base_model = self.get_model(self.config, weights="imagenet")
        # => Transfer all weights except conv1.1
        for i in range(3,len(base_model.layers)):
            if len(base_model.layers[i].weights) > 0:
                print("Setting weights for layer",i, end="\r")
                self.feature_model.layers[i].set_weights(base_model.layers[i].get_weights())
        
        # => Replicate filters of first layer as needed
        weights = base_model.layers[2].get_weights()
        available_channels = weights[0].shape[2]
        target_shape = self.feature_model.layers[2].weights[0].shape
        new_weights = numpy.zeros(target_shape)
        for i in range(new_weights.shape[2]):
            j = i%available_channels
            new_weights[:,:,i,:] = weights[0][:,:,j,:]
            weights_array = [new_weights]
            if len(weights) > 1: 
                weights_array += weights[1:]
            self.feature_model.layers[2].set_weights(weights_array)
        print("Network initialized with pretrained ImageNet weights")


