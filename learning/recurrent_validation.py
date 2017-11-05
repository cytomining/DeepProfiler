import os
import numpy as np
import tensorflow as tf
import pickle

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
        self.save_features = config["validation"]["save_features"] #and config["validation"]["sample_first_crops"]
        self.metrics = []


    def output_base(self, meta):
        filebase = os.path.join(
            self.val_dir,
            str(meta["Metadata_Plate"]) + "_" +
            str(meta["Metadata_Well"]) + "_" +
            str(meta["Metadata_Site"])
        )
        return filebase


    def configure(self, session, checkpoint_file):
        # Create model and load weights
        batch_size = self.config["validation"]["minibatch"]
        self.config["training"]["minibatch"] = batch_size
        feature_layer = self.config["profiling"]["feature_layer"]
        input_shape = (
            self.config["image_set"]["crop_set_length"],
            self.config["sampling"]["box_size"],      # height
            self.config["sampling"]["box_size"],      # width
            len(self.config["image_set"]["channels"]) # channels
        )
        self.model = learning.models.create_recurrent_keras_resnet(input_shape, self.dset.targets, is_training=False)
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
            mtr.configure_ops(tgt.index)
            self.metrics.append(mtr)

        # Prepare output directory
        self.val_dir = self.config["training"]["output"] + "/validation/"
        if not os.path.isdir(self.val_dir):
            os.mkdir(self.val_dir)

        # Initiate generator
        self.crop_generator = imaging.cropping.SingleImageCropSetGenerator(self.config, self.dset)
        self.crop_generator.start(session)
        self.session = session


    def load_batches(self, meta):
        filebase = self.output_base(meta)
        if os.path.isfile(filebase + ".pkl"):
            with open(filebase + ".pkl", "rb") as input_file:
                batches = pickle.load(input_file)
                self.predict(batches, meta)
            return False
        else:
            return True


    def process_batches(self, key, image_array, meta):
        # Prepare image for cropping
        batch_size = self.config["validation"]["minibatch"] 
        total_crops, pads = self.crop_generator.prepare_image(
                                   self.session, 
                                   image_array, 
                                   meta, 
                                   self.config["validation"]["sample_first_crops"]
                            )
        batches = []
        for batch in self.crop_generator.generate(self.session):
            batches.append(batch)

        filebase = self.output_base(meta)
        batch_data = {"total_crops": total_crops, "pads": pads, "batches": batches}
        with open(filebase + ".pkl", "wb") as output_file:
            pickle.dump(batch_data, output_file)
        self.predict(batch_data, meta)


    def predict(self, batch_data, meta):
        features = np.zeros(shape=(batch_data["total_crops"], self.num_features))

        bp = 0
        for batch in batch_data["batches"]:
            # Forward propagate crops into the network and get the outputs
            output = self.model.predict(batch[0])
            if type(output) is not list:
                output = [output]
            bp += 1

            # Remove padded crops
            if len(batch_data["batches"]) == bp and batch_data["pads"] > 0:
                p = batch_data["pads"]
                for i in range(len(batch)):
                    batch[i] = batch[i][0:-p,...]
                for i in range(len(output)):
                    output[i] = output[i][0:-p,...]

            # Compute performance metrics for each target
            # batch[0] contains images, batch[i+1] contains the targets
            for i in range(len(self.metrics)):
                metric_values = self.session.run(
                        self.metrics[i].get_ops(), 
                        feed_dict=self.metrics[i].set_inputs(batch[i+1], output[i])
                    )
                self.metrics[i].update(metric_values, batch[0].shape[0])

            # Extract features (again) 
            # TODO: compute predictions and features at the same time
            if self.save_features:
                f = self.feat_extractor((batch[0], 0))
                batch_size = batch[0].shape[0]
                features[(bp - 1) * batch_size:bp * batch_size, :] = f[0]


        # Save features and report performance
        filebase = self.output_base(meta)
        if self.save_features:
            if batch_data["pads"] > 0:
                features = features[:-batch_data["pads"], :]
            np.savez_compressed(filebase + ".npz", f=features)
            print(filebase, features.shape)


    def report_results(self):
        status = ""
        for metric in self.metrics:
            status += " " + metric.result_string()
        print(status)


def validate(config, dset, checkpoint_file):
    configuration = tf.ConfigProto()
    configuration.gpu_options.allow_growth = True
    configuration.gpu_options.visible_device_list = "0"
    session = tf.Session(config=configuration)
    keras.backend.set_session(session)
    keras.backend.set_learning_phase(0)

    validation = Validation(config, dset)
    validation.configure(session, checkpoint_file)
    dset.scan(validation.process_batches, frame=config["validation"]["frame"], check=validation.load_batches)
    validation.report_results()

    print("Validation: done")

