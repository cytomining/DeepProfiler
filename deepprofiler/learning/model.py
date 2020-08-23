import gc
import os
import random
import abc

import comet_ml
import keras
import numpy as np
import tensorflow as tf

import deepprofiler.dataset.utils
import deepprofiler.imaging.cropping
import deepprofiler.learning.validation
import deepprofiler.learning.layers

##################################################
# This class should be used as an abstract base
# class for plugin models.
##################################################


class DeepProfilerModel(abc.ABC):

    def __init__(self, config, dset, crop_generator, val_crop_generator, is_training):
        self.feature_model = None
        self.loss = None
        self.optimizer = None
        self.config = config
        self.dset = dset
        self.train_crop_generator = crop_generator(config, dset)
        self.val_crop_generator = val_crop_generator(config, dset)
        self.random_seed = None
        self.is_training = is_training

    def seed(self, seed):
        self.random_seed = seed
        random.seed(seed)
        np.random.seed(seed)
        tf.set_random_seed(seed)

    def train(self, epoch=1, metrics=["accuracy"], verbose=1):
        # Raise ValueError if feature model isn't properly defined
        check_feature_model(self)

        # Print model summary
        self.feature_model.summary()

        # Compile model
        self.feature_model.compile(self.optimizer, self.loss, metrics)

        # Create comet ml experiment
        experiment = setup_comet_ml(self)

        # Create main session
        main_session = start_main_session()

        # Start val crop generator
        x_validation, y_validation = load_validation_data(self, main_session)

        # Start train crop generator
        self.train_crop_generator.start(main_session)

        # Get training parameters
        epochs, steps, schedule_epochs, schedule_lr, freq = setup_params(self, experiment)

        # Load weights
        self.load_weights(epoch)

        # Create callbacks
        callbacks = setup_callbacks(self, schedule_epochs, schedule_lr, self.dset, experiment)

        # Train model
        self.feature_model.fit_generator(
            generator=self.train_crop_generator.generate(main_session),
            steps_per_epoch=steps,
            epochs=epochs,
            callbacks=callbacks,
            verbose=verbose,
            initial_epoch=epoch - 1,
            validation_data=(x_validation, y_validation),
            validation_freq=freq
        ) 
            
        # Stop threads and close sessions
        close(self, main_session)

        # Return the feature model and validation data
        return self.feature_model, x_validation, y_validation

    
    def copy_pretrained_weights(self):
        # Override this method if the model can load pretrained weights
        print("This model does not support ImageNet pretrained weights initialization")
        return

    def load_weights(self, epoch):
        output_file = self.config["paths"]["checkpoints"] + "/checkpoint_{epoch:04d}.hdf5"
        previous_model = output_file.format(epoch=epoch - 1)
        if epoch >= 1 and os.path.isfile(previous_model):
            self.feature_model.load_weights(previous_model)
            print("Weights from previous model loaded:", previous_model)
            return True
        else:
            # Initialize all tf variables
            keras.backend.get_session().run(tf.global_variables_initializer())
            if self.config["train"]["model"]["initialization"] == "ImageNet":
                self.copy_pretrained_weights()
            return False


def check_feature_model(dpmodel):
    if "feature_model" not in vars(dpmodel): #or not isinstance(dpmodel.feature_model, keras.Model):
        raise ValueError("Feature model is not properly defined.")


def setup_comet_ml(dpmodel):
    if 'comet_ml' in dpmodel.config["train"].keys():
        experiment = comet_ml.Experiment(
            api_key=dpmodel.config["train"]["comet_ml"]["api_key"],
            project_name=dpmodel.config["train"]["comet_ml"]["project_name"]
        )
        if dpmodel.config["experiment_name"] != "results":
            experiment.set_name(dpmodel.config["experiment_name"])
        experiment.log_others(dpmodel.config)
    else:
        experiment = None
    return experiment


def start_main_session():
    configuration = tf.ConfigProto()
    configuration.gpu_options.allow_growth = True
    main_session = tf.Session(config=configuration)
    keras.backend.set_session(main_session)
    return main_session


def load_validation_data(dpmodel, session):
    dpmodel.val_crop_generator.start(session)
    x_validation, y_validation = deepprofiler.learning.validation.load_validation_data(
        dpmodel.config,
        dpmodel.dset,
        dpmodel.val_crop_generator,
        session)
    return x_validation, y_validation


def setup_callbacks(dpmodel, lr_schedule_epochs, lr_schedule_lr, dset, experiment):
    # Checkpoints
    output_file = dpmodel.config["paths"]["checkpoints"] + "/checkpoint_{epoch:04d}.hdf5"
    period = 1
    save_best = False
    if "checkpoint_policy" in dpmodel.config["train"]["model"] and isinstance(dpmodel.config["train"]["model"]["checkpoint_policy"], int):
        period = int(dpmodel.config["train"]["model"]["checkpoint_policy"])
    elif "checkpoint_policy" in dpmodel.config["train"]["model"] and dpmodel.config["train"]["model"]["checkpoint_policy"] == 'best':
        save_best = True

    callback_model_checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=output_file,
        save_weights_only=True,
        save_best_only=save_best,
        period=period
    )
    
    # CSV Log
    csv_output = dpmodel.config["paths"]["logs"] + "/log.csv"
    callback_csv = keras.callbacks.CSVLogger(filename=csv_output)

    # Queue stats
    qstats = keras.callbacks.LambdaCallback(
        on_train_begin=lambda logs: dset.show_setup(),
        on_epoch_end=lambda epoch, logs: experiment.log_metrics(dset.show_stats()) if experiment else dset.show_stats()
    )
    
    # Control Normalization statistics
    cn_stats = deepprofiler.learning.layers.UpdateControlStatistics(dpmodel.config)

    # Learning rate schedule
    def lr_schedule(epoch, lr):
        if epoch in lr_schedule_epochs:
            return lr_schedule_lr[lr_schedule_epochs.index(epoch)]
        else:
            return lr

    # Collect all callbacks
    if lr_schedule_epochs:
        callback_lr_schedule = keras.callbacks.LearningRateScheduler(lr_schedule, verbose=1)
        callbacks = [callback_model_checkpoint, callback_csv, callback_lr_schedule, qstats, cn_stats]
    else:
        callbacks = [callback_model_checkpoint, callback_csv, qstats, cn_stats]
    return callbacks


def setup_params(dpmodel, experiment):
    epochs = dpmodel.config["train"]["model"]["epochs"]
    steps = dpmodel.dset.steps_per_epoch
    lr_schedule_epochs = []
    lr_schedule_lr = []
    if 'comet_ml' in dpmodel.config["train"].keys():
        params = dpmodel.config["train"]["model"]["params"]
        experiment.log_others(params)
    if "lr_schedule" in dpmodel.config["train"]["model"]:
        if dpmodel.config["train"]["model"]["lr_schedule"] == "cosine":
            lr_schedule_epochs = [x for x in range(epochs)]
            init_lr = dpmodel.config["train"]["model"]["params"]["learning_rate"]
            # Linear warm up
            lr_schedule_lr = [init_lr/(5-t) for t in range(5)]
            # Cosine decay
            lr_schedule_lr += [0.5 * (1 + np.cos((np.pi * t) / epochs)) * init_lr for t in range(5, epochs)]
        else:
            assert len(dpmodel.config["train"]["model"]["lr_schedule"]["epoch"]) == \
                   len(dpmodel.config["train"]["model"]["lr_schedule"]["lr"]), "Make sure that the length of " \
                                                                               "lr_schedule->epoch equals the length of " \
                                                                               "lr_schedule->lr in the config file."

            lr_schedule_epochs = dpmodel.config["train"]["model"]["lr_schedule"]["epoch"]
            lr_schedule_lr = dpmodel.config["train"]["model"]["lr_schedule"]["lr"]

    # Validation frequency
    if "frequency" in dpmodel.config["train"]["validation"].keys():
        freq = dpmodel.config["train"]["validation"]["frequency"]
    else:
        freq = 1

    return epochs, steps, lr_schedule_epochs, lr_schedule_lr, freq


def close(dpmodel, crop_session):
    print("Complete! Closing session.", end=" ", flush=True)
    dpmodel.train_crop_generator.stop(crop_session)
    crop_session.close()
    print("All set.")
    gc.collect()

