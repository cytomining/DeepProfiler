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
from deepprofiler.learning import model_utils


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
        model_utils.check_feature_model(self)
        # Print model summary
        self.feature_model.summary()
        # Compile model
        self.feature_model.compile(self.optimizer, self.loss, metrics)
        # Create comet ml experiment
        experiment = model_utils.setup_comet_ml(self)
        # Start train crop generator
        crop_session = model_utils.start_crop_generator(self)
        # Create tf configuration
        configuration = model_utils.tf_configure(self)
        # Start val crop generator
        val_session, x_validation, y_validation = model_utils.start_val_session(self, configuration)
        # Create main session
        main_session = model_utils.start_main_session(configuration)
        # Create callbacks and load weights
        callbacks = model_utils.setup_callbacks(self, epoch, verbose)
        # Create params (epochs, steps, log model params to comet ml)
        epochs, steps = model_utils.setup_params(self, experiment)
        # Initialize all tf variables to avoid tf bug
        model_utils.init_tf_vars()
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
        model_utils.close(self, crop_session, val_session)
        # Return the feature model and validation data
        return self.feature_model, x_validation, y_validation
