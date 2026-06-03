import os
import abc

import comet_ml
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa

AUTOTUNE = tf.data.AUTOTUNE

tf.compat.v1.enable_v2_behavior()
tf.config.run_functions_eagerly(True)

class DeepProfilerModelV2(abc.ABC):
    def __init__(self, config, dset, generator, val_generator, is_training):  # CG and dset params to match signatures
        self.loss = None
        self.optimizer = None
        self.feature_model = None
        self.is_training = is_training
        self.config = config
        self.all_cells = pd.read_csv(self.config["paths"]["sc_index"])

        self.target = self.config["train"]["partition"]["targets"][0]
        self.classes = list(self.all_cells[self.target].unique())
        self.config["num_classes"] = len(self.classes)

    def make_dataset(self, path, batch_size, single_cell_metadata):
        @tf.function
        def fold_channels(crop):
            assert tf.executing_eagerly()
            crop = crop.numpy()
            output = np.reshape(crop, (crop.shape[0], crop.shape[0], -1), order="F").astype(np.float32)
            output = output / 255.
            for i in range(output.shape[-1]):
                mean = np.mean(output[:, :, i])
                std = np.std(output[:, :, i])
                output[:, :, i] = (output[:, :, i] - mean) / std
            return tf.convert_to_tensor(output, dtype=tf.float32)

        def parse_image(filename):
            image = tf.io.read_file(filename)
            image = tf.image.decode_png(image, channels=0)
            image = tf.py_function(func=fold_channels, inp=[image], Tout=tf.float32)
            return image

        def configure_for_performance(ds):
            ds = ds.shuffle(buffer_size=50000)
            ds = ds.batch(batch_size)
            ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            return ds

        filenames = single_cell_metadata["Image_Name"].tolist()
        for i in range(len(filenames)):
            filenames[i] = os.path.join(path, filenames[i])

        steps = np.math.ceil(len(filenames) / batch_size)
        filenames_ds = tf.data.Dataset.from_tensor_slices(filenames)
        images_ds = filenames_ds.map(parse_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        labels = tf.keras.utils.to_categorical(single_cell_metadata["Categorical"])
        labels_ds = tf.data.Dataset.from_tensor_slices(labels)
        ds = tf.data.Dataset.zip((images_ds, labels_ds))
        ds = configure_for_performance(ds)
        return ds, steps

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

        split_field = self.config["train"]["partition"]["split_field"]
        training_split_values = self.config["train"]["partition"]["training"]
        validation_split_values = self.config["train"]["partition"]["validation"]

        experiment = self.setup_comet_ml(self.config)

        directory = self.config["paths"]["single_cell_set"]
        dataset, steps_per_epoch = self.make_dataset(directory, batch_size, self.all_cells[
            self.all_cells[split_field].isin(training_split_values)])
        validation_dataset, _ = self.make_dataset(directory, batch_size, self.all_cells[
            self.all_cells[split_field].isin(validation_split_values)])

        self.feature_model.compile(self.optimizer, self.loss, metrics=["accuracy",
                                        tfa.metrics.F1Score(num_classes=self.config["num_classes"], average='macro'),
                                        tf.keras.metrics.TopKCategoricalAccuracy(k=5),
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
                          initial_epoch=epoch - 1
                          )
        else:
            self.feature_model.fit(dataset,
                      epochs=self.config["train"]["model"]["epochs"],
                      callbacks=callbacks,
                      verbose=1,
                      validation_data=validation_dataset,
                      validation_freq=self.config["train"]["validation"]["frequency"],
                      initial_epoch=epoch - 1
                      )
