from comet_ml import Experiment
from abc import ABC, abstractmethod
import gc
import os
import random

import tensorflow as tf
import numpy as np
import keras
import sklearn.metrics

import deepprofiler.imaging.cropping
import deepprofiler.learning.validation
import deepprofiler.dataset.utils


##################################################
# This class should be used as an abstract base
# class for plugin models.
##################################################


class DeepProfilerModel(ABC):

    def __init__(self, config, dset, crop_generator, val_crop_generator):
        self.feature_model = None
        self.loss = None
        self.optimizer = None
        self.config = config
        self.dset = dset
        self.train_crop_generator = crop_generator(config, dset)
        self.val_crop_generator = val_crop_generator(config, dset)
        self.random_seed = None
        if "comet_ml" not in config["train"].keys():
            self.config["train"]["comet_ml"]["track"] = False

    def seed(self, seed):
        self.random_seed = seed
        random.seed(seed)
        np.random.seed(seed)
        tf.set_random_seed(seed)

    def train(self, epoch=1, metrics=['accuracy'], verbose=1):  # TODO: simplify default train method
        if 'feature_model' not in vars(self) or self.feature_model is None:
            raise ValueError("Feature model is not defined.")
        print(self.feature_model.summary())
        self.feature_model.compile(self.optimizer, self.loss, metrics)
        timestamp = deepprofiler.dataset.utils.tic()
        if self.config["train"]["comet_ml"]["track"]:
            experiment = Experiment(
                api_key=self.config["train"]["comet_ml"]["api_key"],
                project_name=self.config["train"]["comet_ml"]["project_name"]
            )
        # Create cropping graph
        crop_graph = tf.Graph()
        with crop_graph.as_default():
            cpu_config = tf.ConfigProto(device_count={'CPU': 1, 'GPU': 0})
            cpu_config.gpu_options.visible_device_list = ""
            crop_session = tf.Session(config=cpu_config)
            self.train_crop_generator.start(crop_session)
        gc.collect()
        # Start validation session
        configuration = tf.ConfigProto()
        configuration.gpu_options.visible_device_list = self.config["train"]["gpus"]
        crop_graph = tf.Graph()
        with crop_graph.as_default():
            val_session = tf.Session(config=configuration)
            keras.backend.set_session(val_session)
            self.val_crop_generator.start(val_session)
            x_validation, y_validation = deepprofiler.learning.validation.validate(
                self.config,
                self.dset,
                self.val_crop_generator,
                val_session)
        gc.collect()
        # Start main session
        main_session = tf.Session(config=configuration)
        keras.backend.set_session(main_session)
        if verbose != 0:
            if self.config["train"]["model"]["save_all"]:
                os.makedirs(self.config["paths"]["checkpoints"] + "/" + str(timestamp))
                output_file = self.config["paths"]["checkpoints"] + "/" + str(timestamp) + "/checkpoint_{epoch:04d}.hdf5"
            else:
                output_file = self.config["paths"]["checkpoints"] + "/checkpoint_{epoch:04d}.hdf5"
            callback_model_checkpoint = keras.callbacks.ModelCheckpoint(
                filepath=output_file,
                save_weights_only=True,
                save_best_only=False
            )
            if self.config["train"]["model"]["save_all"]:
                os.makedirs(self.config["paths"]["logs"] + "/" + str(timestamp))
                csv_output = self.config["paths"]["logs"] + "/" + str(timestamp) + "/log.csv"
            else:
                csv_output = self.config["paths"]["logs"] + "/log.csv"
            callback_csv = keras.callbacks.CSVLogger(filename=csv_output)

            callbacks = [callback_model_checkpoint, callback_csv]

            previous_model = output_file.format(epoch=epoch - 1)
            if epoch >= 1 and os.path.isfile(previous_model):
                self.feature_model.load_weights(previous_model)
                print("Weights from previous model loaded:", previous_model)
        else:
            callbacks = None

        epochs = self.config["train"]["model"]["epochs"]
        steps = self.config["train"]['model']["steps"]

        if self.config["train"]["comet_ml"]["track"]:
            params = self.config["train"]["model"]["params"]
            experiment.log_multiple_params(params)


        keras.backend.get_session().run(tf.initialize_all_variables())
        self.feature_model.fit_generator(
            generator=self.train_crop_generator.generate(crop_session),
            steps_per_epoch=steps,
            epochs=epochs,
            callbacks=callbacks,
            verbose=verbose,
            initial_epoch=epoch - 1,
            validation_data=(x_validation, y_validation)
        )

        # Close session and stop threads
        print("Complete! Closing session.", end="", flush=True)
        self.train_crop_generator.stop(crop_session)
        crop_session.close()
        print("All set.")
        gc.collect()

        return self.feature_model, x_validation, y_validation
