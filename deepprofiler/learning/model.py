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


##################################################
# This class should be used as an abstract base
# class for plugin models.
##################################################


class DeepProfilerModel(ABC):

    def __init__(self, config, dset, crop_generator, val_crop_generator, verbose=1):
        self.model = None
        self.loss = None
        self.optimizer = None
        self.config = config
        self.dset = dset
        self.verbose = verbose
        self.train_crop_generator = crop_generator(config, dset)
        self.val_crop_generator = val_crop_generator(config, dset)
        self.random_seed = None
        if "comet_ml" not in config["model"].keys():
            self.config["model"]["comet_ml"] = False

    def seed(self, seed):
        self.random_seed = seed
        random.seed(seed)
        np.random.seed(seed)
        tf.set_random_seed(seed)

    def train(self, epoch=1, metrics=['accuracy']):
        if self.model is None:
            raise ValueError("Model is not defined!")
        print(self.model.summary())
        self.model.compile(self.optimizer, self.loss, metrics)
        if not os.path.isdir(self.config["training"]["output"]):
            os.mkdir(self.config["training"]["output"])
        if self.config["model"]["comet_ml"]:
            experiment = Experiment(
                api_key=self.config["validation"]["api_key"],
                project_name=self.config["validation"]["project_name"]
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
        configuration.gpu_options.visible_device_list = self.config["training"]["visible_gpus"]
        crop_graph = tf.Graph()
        with crop_graph.as_default():
            val_session = tf.Session(config=configuration) #TODO
            keras.backend.set_session(val_session) #TODO
            self.val_crop_generator.start(val_session) #TODO
            x_validation, y_validation = deepprofiler.learning.validation.validate(
                self.config,
                self.dset,
                self.val_crop_generator,
                val_session) #TODO
        gc.collect() #TODO
        # Start main session
        main_session = tf.Session(config=configuration)
        keras.backend.set_session(main_session)

        if self.verbose:
            output_file = self.config["training"]["output"] + "/checkpoint_{epoch:04d}.hdf5"
            callback_model_checkpoint = keras.callbacks.ModelCheckpoint(
                filepath=output_file,
                save_weights_only=True,
                save_best_only=False
            )
            csv_output = self.config["training"]["output"] + "/log.csv"
            callback_csv = keras.callbacks.CSVLogger(filename=csv_output)

            callbacks = [callback_model_checkpoint, callback_csv]
        else:
            callbacks = None

        previous_model = output_file.format(epoch=epoch - 1)
        if epoch >= 1 and os.path.isfile(previous_model):
            self.model.load_weights(previous_model)
            print("Weights from previous model loaded:", previous_model)

        epochs = self.config["model"]["params"]["epochs"]
        steps = self.config["model"]["params"]["steps"]

        if self.config["model"]["comet_ml"]:
            params = self.config["model"]["params"]
            experiment.log_multiple_params(params)

        keras.backend.get_session().run(tf.initialize_all_variables())
        self.model.fit_generator(
            generator=self.train_crop_generator.generate(crop_session),
            steps_per_epoch=steps,
            epochs=epochs,
            callbacks=callbacks,
            verbose=self.verbose,
            initial_epoch=epoch - 1,
            validation_data=(x_validation, y_validation)
        )

        # Close session and stop threads
        print("Complete! Closing session.", end="", flush=True)
        self.train_crop_generator.stop(crop_session)
        crop_session.close()
        print("All set.")
        gc.collect()

        return self.model, x_validation, y_validation
