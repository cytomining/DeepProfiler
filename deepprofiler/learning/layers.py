import tensorflow as tf
import keras
import re
import os
import numpy as np
import pandas as pd
import skimage.io
from tqdm import tqdm

import deepprofiler.imaging.cropping



## Parse a model and replace or inser layers.
## Taken from: 
## https://stackoverflow.com/questions/49492255/how-to-replace-or-insert-intermediate-layer-in-keras-model

def insert_layer_nonseq(model, layer_regex, insert_layer_factory,
                        insert_layer_name=None, position='after'):

    # Auxiliary dictionary to describe the network graph
    network_dict = {'input_layers_of': {}, 'new_output_tensor_of': {}}

    # Set the input layers of each layer
    for layer in model.layers:
        for node in layer._outbound_nodes:
            layer_name = node.outbound_layer.name
            if layer_name not in network_dict['input_layers_of']:
                network_dict['input_layers_of'].update(
                        {layer_name: [layer.name]})
            else:
                network_dict['input_layers_of'][layer_name].append(layer.name)

    # Set the output tensor of the input layer
    network_dict['new_output_tensor_of'].update(
            {model.layers[0].name: model.input})

    # Iterate over all layers after the input
    model_outputs = []
    for layer in model.layers[1:]:

        # Determine input tensors
        layer_input = [network_dict['new_output_tensor_of'][layer_aux] 
                for layer_aux in network_dict['input_layers_of'][layer.name]]
        if len(layer_input) == 1:
            layer_input = layer_input[0]

        # Insert layer if name matches the regular expression
        if re.match(layer_regex, layer.name):
            if position == 'replace':
                x = layer_input
            elif position == 'after':
                x = layer(layer_input)
            elif position == 'before':
                pass
            else:
                raise ValueError('position must be: before, after or replace')

            new_layer = insert_layer_factory(layer.get_config())
            if insert_layer_name:
                new_layer.name = insert_layer_name
            else:
                new_layer.name = '{}_{}'.format(layer.name, 
                                                new_layer.name)
            x = new_layer(x)
            print('New layer: {} Old layer: {} Type: {}'.format(new_layer.name,
                                                            layer.name, position))
            if position == 'before':
                x = layer(x)
        else:
            x = layer(layer_input)

        # Set new output tensor (the original one, or the one of the inserted
        # layer)
        network_dict['new_output_tensor_of'].update({layer.name: x})

        # Save tensor in output list if it is output in initial model
        if layer.name in model.output_names:
            model_outputs.append(x)

    return keras.models.Model(inputs=model.inputs, outputs=model_outputs)

## 
## A BatchNormalization layer that does not update during training
## The goal of this layer is to update the running mean and standar deviation
## with selected samples only, instead of using batch statistics. In the context
## or DeepProfiler, the selected samples will be control samples
##

class ControlNormalization(keras.layers.BatchNormalization):

    def __init__(self, **kwargs):
        super(ControlNormalization, self).__init__(**kwargs)
        self.updating_statistics = False
        self.stop_updates()

    def call(self, inputs, training=None):
        if not self.updating_statistics:
            return super().call(inputs, training=False)
        elif self.updating_statistics:
            return super().call(inputs, training=True)

    def start_updates(self):
        self.trainable = True
        self.updating_statistics = True

    def stop_updates(self):
        self.trainable = False
        self.updating_statistics = False
        

def cn_factory(layer_config):
    layer = ControlNormalization.from_config(layer_config)
    layer.name = "ctrlnorm"
    return layer



class UpdateControlStatistics(keras.callbacks.Callback):

    def __init__(self, config):
        super(UpdateControlStatistics, self).__init__()
        self.directory = config["paths"]["single_cell_sample"]
        self.num_channels = len(config["dataset"]["images"]["channels"])
        self.box_size = config["dataset"]["locations"]["box_size"]
        self.batch_size = config["train"]["model"]["params"]["batch_size"]

        self.samples = pd.read_csv(os.path.join(self.directory, "sc-metadata.csv"))
        control_filter = self.samples["Class_Name"] == config["dataset"]["metadata"]["control_value"]
        self.samples = self.samples[control_filter]
        self.samples = self.samples.sample(frac=1.0).reset_index(drop=True)

        self.layer_regex = ".*bn_ctrlnorm.*"


    def on_epoch_begin(self, epoch, logs=None):
        # Make ControlNormalization layers trainable
        print("Updating control normalization statistics")
        self.model.stop_training = True
        for layer in self.model.layers:
            if re.match(self.layer_regex, layer.name):
                layer.start_updates()

        # Compute forward pass on control samples
        i = 0
        x = np.zeros([self.batch_size, self.box_size, self.box_size, self.num_channels])
        for k,r in tqdm(self.samples.iterrows()):
            filename = os.path.join(self.directory, r.Image_Name)
            unfolded_im = skimage.io.imread(filename).astype(np.float32)
            x[i,:,:,:] = deepprofiler.imaging.cropping.fold_channels(unfolded_im)
            i += 1
            if i == self.batch_size:
                y = self.model.predict(x)
                i = 0
        self.model.predict(x)

        # Return ControlNormalization layers back to normal
        for layer in self.model.layers:
            if re.match(self.layer_regex, layer.name):
                layer.stop_updates()




