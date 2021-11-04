import os
import abc

import comet_ml
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from plugins.crop_generators.sampled_crop_generator import GeneratorClass

AUTOTUNE = tf.data.AUTOTUNE

tf.compat.v1.enable_v2_behavior()
tf.config.run_functions_eagerly(True)


class DeepProfilerModel(abc.ABC):
    def __init__(self, config, generator, val_generator, is_training):
        self.generator = generator
        self.val_generator = val_generator
        self.is_training = is_training
        self.config = config

        self.loss = None
        self.optimizer = None
        self.feature_model = None
        self.all_cells = pd.read_csv(self.config["paths"]["sc_index"])

        self.target = self.config["train"]["partition"]["targets"][0]
        self.classes = list(self.all_cells[self.target].unique())
        self.config["num_classes"] = len(self.classes)

    def setup_callbacks(self, config):
        callbacks = []

        # CSV Log
        csv_output = config["paths"]["logs"] + "/log.csv"
        callback_csv = tf.keras.callbacks.CSVLogger(filename=csv_output)
        callbacks.append(callback_csv)

        # Checkpoints
        output_file = config["paths"]["checkpoints"] + "/checkpoint_{epoch:04d}.hdf5"
        period = 1
        save_best = False
        if "checkpoint_policy" in config["train"]["model"] and isinstance(
                config["train"]["model"]["checkpoint_policy"], int):
            period = int(config["train"]["model"]["checkpoint_policy"])
        elif "checkpoint_policy" in config["train"]["model"] and \
                config["train"]["model"]["checkpoint_policy"] == 'best':
            save_best = True

        callback_model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=output_file,
            save_weights_only=True,
            save_best_only=save_best,
            period=period
        )
        callbacks.append(callback_model_checkpoint)
        epochs = config["train"]["model"]["epochs"]

        def lr_schedule(epoch, lr):
            if lr_schedule_epochs and epoch in lr_schedule_epochs:
                return lr_schedule_lr[lr_schedule_epochs.index(epoch)]
            else:
                return lr

        if "lr_schedule" in config["train"]["model"]:
            if config["train"]["model"]["lr_schedule"] == "cosine":
                lr_schedule_epochs = [x for x in range(epochs)]
                init_lr = config["train"]["model"]["params"]["learning_rate"]
                # Linear warm up
                lr_schedule_lr = [init_lr / (5 - t) for t in range(5)]
                # Cosine decay
                lr_schedule_lr += [0.5 * (1 + np.cos((np.pi * t) / epochs)) * init_lr for t in range(5, epochs)]
                callback_lr_schedule = tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose=1)
            elif config["train"]["model"]["lr_schedule"] == "plateau":
                callback_lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5,
                                                                            min_lr=0.0001)
                config["train"]["validation"]["frequency"] = 1
            else:
                assert len(config["train"]["model"]["lr_schedule"]["epoch"]) == \
                       len(config["train"]["model"]["lr_schedule"]["lr"]), "Make sure that the length of " \
                                                                           "lr_schedule->epoch equals the length of " \
                                                                           "lr_schedule->lr in the config file."

                lr_schedule_epochs = config["train"]["model"]["lr_schedule"]["epoch"]
                lr_schedule_lr = config["train"]["model"]["lr_schedule"]["lr"]
                callback_lr_schedule = tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose=1)

            callbacks.append(callback_lr_schedule)

        return callbacks

    def copy_pretrained_weights(self):
        # Override this method if the model can load pretrained weights
        print("This model does not support ImageNet pretrained weights initialization")
        return

    def setup_comet_ml(self, config):
        if 'comet_ml' in config["train"].keys():
            experiment = comet_ml.Experiment(
                api_key=config["train"]["comet_ml"]["api_key"],
                project_name=config["train"]["comet_ml"]["project_name"],
                auto_param_logging=True,
                auto_histogram_weight_logging=False,
                auto_histogram_gradient_logging=False,
                auto_histogram_activation_logging=False
            )
            if config["experiment_name"] != "results":
                experiment.set_name(config["experiment_name"])
            experiment.log_others(config)
        else:
            experiment = None
        return experiment

    def load_weights(self, epoch):
        output_file = self.config["paths"]["checkpoints"] + "/checkpoint_{epoch:04d}.hdf5"
        previous_model = output_file.format(epoch=epoch - 1)
        # Initialize all tf variables
        if epoch >= 1 and os.path.isfile(previous_model):
            self.load_weights(previous_model)
            print("Weights from previous model loaded:", previous_model)
            return True
        else:
            if self.config["train"]["model"]["initialization"] == "ImageNet":
                self.copy_pretrained_weights()
            return False

    def train(self, epoch):
        batch_size = self.config["train"]["model"]["params"]["batch_size"]
        self.all_cells["Categorical"] = pd.Categorical(self.all_cells[self.target]).codes

        experiment = self.setup_comet_ml(self.config)

        crop_generator = GeneratorClass(self.config)
        dataset = tf.data.Dataset.from_generator(crop_generator.generator, output_signature=(
            tf.TensorSpec(shape=(96, 96, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(3,), dtype=tf.int32)
        )).batch(batch_size)

        val_crop_generator = GeneratorClass(self.config, mode="validation")
        validation_dataset = tf.data.Dataset.from_generator(val_crop_generator.generate, output_signature=(
            tf.TensorSpec(shape=(self.config['dataset']['locations']['box_size'],
                                 self.config['dataset']['locations']['box_size'],
                                 len(self.config['dataset']['images']['channels'])),
                          dtype=tf.float32),
            tf.TensorSpec(shape=(len(self.classes),), dtype=tf.int32)
        )).batch(batch_size)
        if self.config["train"]["validation"].get("top_k"):
            top_k = self.config["train"]["validation"].get("top_k")
        else:
            top_k = 5
        self.feature_model.compile(self.optimizer, self.loss, metrics=["accuracy",
                                                                       tfa.metrics.F1Score(
                                                                           num_classes=self.config["num_classes"],
                                                                           average='macro'),
                                                                       tf.keras.metrics.TopKCategoricalAccuracy(k=top_k),
                                                                       tf.keras.metrics.Precision()])
        print(self.feature_model.summary())
        callbacks = self.setup_callbacks(self.config)

        if experiment:
            with experiment.train():
                self.feature_model.fit(dataset,
                                       epochs=self.config["train"]["model"]["epochs"],
                                       callbacks=callbacks,
                                       verbose=1,
                                       validation_data=validation_dataset,
                                       validation_freq=self.config["train"]["validation"]["frequency"],
                                       initial_epoch=epoch - 1,
                                       steps_per_epoch=crop_generator.expected_steps
                                       )
        else:
            self.feature_model.fit(dataset,
                                   epochs=self.config["train"]["model"]["epochs"],
                                   callbacks=callbacks,
                                   verbose=1,
                                   validation_data=validation_dataset,
                                   validation_freq=self.config["train"]["validation"]["frequency"],
                                   initial_epoch=epoch - 1,
                                   steps_per_epoch=crop_generator.expected_steps
                                   )
