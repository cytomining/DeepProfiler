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

    def train(self, epoch=1, metrics=['accuracy'], verbose=1):
        # Raise ValueError if feature model isn't properly defined
        check_feature_model(self)
        # Print model summary
        if verbose != 0:
            self.feature_model.summary()
        # Compile model
        self.feature_model.compile(self.optimizer, self.loss, metrics)
        # Create comet ml experiment
        experiment = setup_comet_ml(self)
        # Create tf configuration
        configuration = tf_configure(self)
        # Start train crop generator
        crop_session = start_crop_session(self, configuration)
        # Start val crop generator
        val_session, x_validation, y_validation = start_val_session(self, configuration)
        # Create main session
        main_session = start_main_session(configuration)
        # Initialize all tf variables to avoid tf bug (TODO: this causes problem with saving/loading weights, do not use)
#         init_tf_vars()
        if verbose != 0:  # verbose is only 0 when optimizing hyperparameters
            # Load weights
            load_weights(self, epoch)
            # Create callbacks
            callbacks = setup_callbacks(self)
        else:
            callbacks = None
        # Create params (epochs, steps, log model params to comet ml)
        epochs, steps = setup_params(self, experiment)
        # Train model
        self.feature_model.fit_generator(
            generator=self.train_crop_generator.generate(crop_session),
            steps_per_epoch=steps,
            epochs=epochs,
            callbacks=callbacks,
            verbose=verbose,
            initial_epoch=epoch - 1,
            validation_data=(x_validation, y_validation)
        )
        # Stop threads and close sessions
        close(self, crop_session)
        # Return the feature model and validation data
        return self.feature_model, x_validation, y_validation


def check_feature_model(dpmodel):
    if 'feature_model' not in vars(dpmodel) or not isinstance(dpmodel.feature_model, keras.Model):
        raise ValueError("Feature model is not properly defined.")


def setup_comet_ml(dpmodel):
    if dpmodel.config["train"]["comet_ml"]["track"]:
        experiment = Experiment(
            api_key=dpmodel.config["train"]["comet_ml"]["api_key"],
            project_name=dpmodel.config["train"]["comet_ml"]["project_name"]
        )
        if "experiment_name" in dpmodel.config["train"]["comet_ml"].keys():
            experiment.set_name(dpmodel.config["train"]["comet_ml"]["experiment_name"])
    else:
        experiment = None
    return experiment


def start_crop_session(dpmodel, configuration):
    crop_graph = tf.Graph()
    with crop_graph.as_default():
#         cpu_config = tf.ConfigProto(device_count={'CPU': 1, 'GPU': 0})
#         cpu_config.gpu_options.visible_device_list = ""
        crop_session = tf.Session(config=configuration)
        dpmodel.train_crop_generator.start(crop_session)
    gc.collect()
    return crop_session


def tf_configure(dpmodel):
    configuration = tf.ConfigProto()
    configuration.gpu_options.visible_device_list = dpmodel.config["train"]["gpus"]
    configuration.gpu_options.allow_growth = True
    return configuration


def start_val_session(dpmodel, configuration):
    crop_graph = tf.Graph()
    with crop_graph.as_default():
        val_session = tf.Session(config=configuration)
        keras.backend.set_session(val_session)
        dpmodel.val_crop_generator.start(val_session)
        x_validation, y_validation = deepprofiler.learning.validation.validate(
            dpmodel.config,
            dpmodel.dset,
            dpmodel.val_crop_generator,
            val_session)
    gc.collect()
    return val_session, x_validation, y_validation


def start_main_session(configuration):
    main_session = tf.Session(config=configuration)
    keras.backend.set_session(main_session)
    return main_session


def load_weights(dpmodel, epoch):
    output_file = dpmodel.config["paths"]["checkpoints"] + "/checkpoint_{epoch:04d}.hdf5"
    previous_model = output_file.format(epoch=epoch - 1)
    if epoch >= 1 and os.path.isfile(previous_model):
        dpmodel.feature_model.load_weights(previous_model)
        print("Weights from previous model loaded:", previous_model)


def setup_callbacks(dpmodel):
    output_file = dpmodel.config["paths"]["checkpoints"] + "/checkpoint_{epoch:04d}.hdf5"
    callback_model_checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=output_file,
        save_weights_only=True,
        save_best_only=False
    )
    csv_output = dpmodel.config["paths"]["logs"] + "/log.csv"
    callback_csv = keras.callbacks.CSVLogger(filename=csv_output)
    callbacks = [callback_model_checkpoint, callback_csv]
    return callbacks


def setup_params(dpmodel, experiment):
    epochs = dpmodel.config["train"]["model"]["epochs"]
    steps = dpmodel.config["train"]['model']["steps"]
    if dpmodel.config["train"]["comet_ml"]["track"]:
        params = dpmodel.config["train"]["model"]["params"]
        experiment.log_multiple_params(params)
    return epochs, steps


def init_tf_vars():
    keras.backend.get_session().run(tf.global_variables_initializer())


def close(dpmodel, crop_session):
    print("Complete! Closing session.", end=" ", flush=True)
    dpmodel.train_crop_generator.stop(crop_session)
    crop_session.close()
    print("All set.")
    gc.collect()
