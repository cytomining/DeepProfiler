import gc
import os
import random

import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim.nets

import keras_resnet.models
import keras


class DeepProfilerModel(object):

    def __init__(self, model, config, dset, crop_generator):
        self.model = model
        self.config = config
        self.dset = dset
        self.crop_generator = crop_generator
        self.seed = None

    def seed(self, seed):
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        tf.set_random_seed(seed)

    def train(self, epoch):
        # Create cropping graph
        crop_graph = tf.Graph()
        with crop_graph.as_default():
            cpu_config = tf.ConfigProto(device_count={'CPU': 1, 'GPU': 0})
            cpu_config.gpu_options.visible_device_list = ""
            crop_session = tf.Session(config=cpu_config)
            self.crop_generator.start(crop_session)
        gc.collect()

        # Start main session
        configuration = tf.ConfigProto()
        configuration.gpu_options.visible_device_list = self.config["training"]["visible_gpus"]
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
        self.model.fit_generator(
            generator=self.crop_generator.generate(crop_session),
            steps_per_epoch=steps,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1,
            initial_epoch=epoch - 1
        )

        # Close session and stop threads
        print("Complete! Closing session.", end="", flush=True)
        self.crop_generator.stop(crop_session)
        crop_session.close()
        print("All set.")
        gc.collect()
