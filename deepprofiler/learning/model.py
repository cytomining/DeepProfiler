from comet_ml import Experiment
from abc import ABC, abstractmethod
import gc
import os
import random

import tensorflow as tf
import numpy as np
import keras
from sklearn.metrics import confusion_matrix

import deepprofiler.imaging.cropping
import deepprofiler.learning.validation


##################################################
# This class should be used as an abstract base
# class for plugin models.
##################################################


class DeepProfilerModel(ABC):

    def __init__(self, config, dset, crop_generator):
        self.model = None
        self.config = config
        self.dset = dset
        self.crop_generator = crop_generator
        self.random_seed = None

    def seed(self, seed):
        self.random_seed = seed
        random.seed(seed)
        np.random.seed(seed)
        tf.set_random_seed(seed)

    def train(self, epoch):
        if self.model is None:
            raise ValueError("Model is not defined!")
        print(self.model.summary())
        if not os.path.isdir(self.config["training"]["output"]):
            os.mkdir(self.config["training"]["output"])

        experiment = Experiment(
            api_key=self.config["validation"]["api_key"],
            project_name=self.config["validation"]["project_name"]
        )
        # Create cropping graph
        crop_graph = tf.Graph()
        with crop_graph.as_default():
            val_crop_generator = deepprofiler.imaging.cropping.SingleImageCropGenerator(self.config, self.dset)
            cpu_config = tf.ConfigProto(device_count={'CPU': 1, 'GPU': 0})
            cpu_config.gpu_options.visible_device_list = ""
            crop_session = tf.Session(config=cpu_config)
            self.crop_generator.start(crop_session)
        gc.collect()
        # Start val session
        configuration = tf.ConfigProto()
        configuration.gpu_options.visible_device_list = self.config["training"]["visible_gpus"]
        crop_graph = tf.Graph()
        with crop_graph.as_default():
            val_session = tf.Session(config=configuration)
            keras.backend.set_session(val_session)
            val_crop_generator.start(val_session)
            x_validation, y_validation = deepprofiler.learning.validation.validate(
                self.config,
                self.dset,
                val_crop_generator,
                val_session)
        gc.collect()
        # Start main session
        main_session = tf.Session(config=configuration)
        keras.backend.set_session(main_session)

        output_file = self.config["training"]["output"] + "/checkpoint_{epoch:04d}.hdf5"
        callback_model_checkpoint = keras.callbacks.ModelCheckpoint(
            filepath=output_file,
            save_weights_only=True,
            save_best_only=False
        )
        csv_output = self.config["training"]["output"] + "/log.csv"
        callback_csv = keras.callbacks.CSVLogger(filename=csv_output)

        callbacks = [callback_model_checkpoint, callback_csv]

        previous_model = output_file.format(epoch=epoch - 1)
        if epoch >= 1 and os.path.isfile(previous_model):
            self.model.load_weights(previous_model)
            print("Weights from previous model loaded:", previous_model)

        epochs = self.config["training"]["epochs"]
        steps = self.config["training"]["steps"]

        params = {
            'steps_per_epoch': steps,
            'epochs': epochs,
            'learning_rate': self.config["training"]["learning_rate"],
            "k_value": self.config["validation"]["top_k"],
            "training_batch_size": self.config["training"]["minibatch"],
            "validation_batch_size": self.config["validation"]["minibatch"]
        }
        experiment.log_multiple_params(params)

        keras.backend.get_session().run(tf.initialize_all_variables())
        self.model.fit_generator(
            generator=self.crop_generator.generate(crop_session),
            steps_per_epoch=steps,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1,
            initial_epoch=epoch - 1,
            validation_data=(x_validation, y_validation)
        )

        pred = self.model.predict(x_validation)
        new_pred = []
        for line in pred:
            new_pred.append(np.argmax(line))
        new_y_validation = []
        for line in y_validation:
            new_y_validation.append(np.argmax(line))
        output_confusion_matrix = confusion_matrix(new_y_validation, new_pred)
        np.savetxt(self.config["training"]["output"] + "/confusion_matrix.txt", output_confusion_matrix)

        # Close session and stop threads
        print("Complete! Closing session.", end="", flush=True)
        self.crop_generator.stop(crop_session)
        crop_session.close()
        print("All set.")
        gc.collect()
