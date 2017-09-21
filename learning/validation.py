import os
import numpy as np
import tensorflow as tf

import imaging.boxes
import learning.metrics
import learning.models
import learning.training

import keras


class Validation(object):

    def __init__(self, config, dset):
        self.config = config
        self.dset = dset
        self.config["queueing"]["min_size"] = 0
        self.save_features = config["validation"]["save_features"] and config["validation"]["sample_first_crops"]
        self.metrics = []


    def output_name(self, meta):
        # TODO: parameterize file structure?
        filename = os.path.join(
            self.config["profiling"]["output_dir"],
            str(meta["Metadata_Plate"]) + "_" +
            str(meta["Metadata_Well"]) + "_" +
            str(meta["Metadata_Site"]) + ".npz"
        )
        return filename


    def configure(self, session, checkpoint_file):
        # Create model and load weights
        batch_size = self.config["training"]["minibatch"]  
        feature_layer = self.config["profiling"]["feature_layer"]
        input_shape = (
            self.config["sampling"]["box_size"],      # height
            self.config["sampling"]["box_size"],      # width
            len(self.config["image_set"]["channels"]) # channels
        )
        self.model = learning.models.create_keras_resnet(input_shape, self.dset.targets)
        print("Checkpoint:", checkpoint_file)
        self.model.load_weights(checkpoint_file)

        # Create feature extraction function
        feature_embedding = self.model.get_layer(feature_layer).output
        self.num_features = feature_embedding.shape[1]
        self.feat_extractor = keras.backend.function([self.model.input], [feature_embedding])

        # Configure metrics for each target
        for i in range(len(self.dset.targets)):
            tgt = self.dset.targets[i]
            mtr = learning.metrics.Metrics(name=tgt.field_name, k=self.config["validation"]["top_k"])
            mtr.configure_ops(batch_size, tgt.shape[1])
            self.metrics.append(mtr)

        # Prepare output directory
        if not os.path.exists(self.config["profiling"]["output_dir"]):
            os.mkdir(self.config["profiling"]["output_dir"])

        # Initiate generator
        self.crop_generator = imaging.cropping.SingleImageCropGenerator(self.config, self.dset)
        self.crop_generator.start(session)
        self.session = session


    def check(self, meta):
        if not self.save_features:
            return True
        filename = self.output_name(meta)
        if os.path.isfile(filename):
            print(filename, "already done")
            return False
        else:
            return True
   

    def predict(self, key, image_array, meta):
        # Prepare image for cropping
        batch_size = self.config["training"]["minibatch"]  
        filename = self.output_name(meta)
        total_crops, pads = self.crop_generator.prepare_image(
                                   self.session, 
                                   image_array, 
                                   meta, 
                                   self.config["validation"]["sample_first_crops"]
                            )
        features = np.zeros(shape=(total_crops, self.num_features))

        bp = 0
        for batch in self.crop_generator.generate(self.session):
            # Forward propagate crops into the network and get the outputs
            output = self.model.predict(batch[0])
            if self.save_features:
                f = self.feat_extractor((batch[0], 0))
                features[bp * batch_size:(bp + 1) * batch_size, :] = f[0]

            # Compute performance metrics for each target
            for i in range(len(self.metrics)):
                metric_values = self.session.run(
                        self.metrics[i].get_ops(), 
                        feed_dict=self.metrics[i].set_inputs(batch[1], output[i])
                    )
                self.metrics[i].update(metric_values, batch[0].shape[0])
            bp += 1

        # Save features and report performance
        status = ""
        for metric in self.metrics:
            status += " " + metric.result_string()
        if self.save_features:
            features = features[:-pads, :]
            np.savez_compressed(filename, f=features)
            print(filename, features.shape, status)
        else:
            print(filename, "(not saved)", status)


def validate(config, dset, checkpoint_file):
    configuration = tf.ConfigProto()
    configuration.gpu_options.allow_growth = True
    configuration.gpu_options.visible_device_list = "0"
    session = tf.Session(config=configuration)
    keras.backend.set_session(session)
    keras.backend.set_learning_phase(0)

    validation = Validation(config, dset)
    validation.configure(session, checkpoint_file)
    dset.scan(validation.predict, frame=config["validation"]["frame"], check=validation.check)

    print("Validation: done")

