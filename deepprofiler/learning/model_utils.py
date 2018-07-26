from comet_ml import Experiment
import keras
import tensorflow as tf
import gc
import os

import deepprofiler.learning.validation


def check_feature_model(dpmodel):
    if 'feature_model' not in vars(dpmodel) or not isinstance(dpmodel.feature_model, keras.Model):
        raise ValueError("Feature model is not properly defined.")


def setup_comet_ml(dpmodel):
    if dpmodel.config["train"]["comet_ml"]["track"]:
        experiment = Experiment(
            api_key=dpmodel.config["train"]["comet_ml"]["api_key"],
            project_name=dpmodel.config["train"]["comet_ml"]["project_name"]
        )
    else:
        experiment = None
    return experiment


def start_crop_generator(dpmodel):
    crop_graph = tf.Graph()
    with crop_graph.as_default():
        cpu_config = tf.ConfigProto(device_count={'CPU': 1, 'GPU': 0})
        cpu_config.gpu_options.visible_device_list = ""
        crop_session = tf.Session(config=cpu_config)
        dpmodel.train_crop_generator.start(crop_session)
    gc.collect()
    return crop_session


def tf_configure(dpmodel):
    configuration = tf.ConfigProto()
    configuration.gpu_options.visible_device_list = dpmodel.config["train"]["gpus"]
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


def setup_callbacks(dpmodel, epoch, verbose):
    if verbose != 0:
        output_file = dpmodel.config["paths"]["checkpoints"] + "/checkpoint_{epoch:04d}.hdf5"
        callback_model_checkpoint = keras.callbacks.ModelCheckpoint(
            filepath=output_file,
            save_weights_only=True,
            save_best_only=False
        )
        csv_output = dpmodel.config["paths"]["logs"] + "/log.csv"
        callback_csv = keras.callbacks.CSVLogger(filename=csv_output)
        callbacks = [callback_model_checkpoint, callback_csv]
        previous_model = output_file.format(epoch=epoch - 1)
        if epoch >= 1 and os.path.isfile(previous_model):
            dpmodel.feature_model.load_weights(previous_model)
            print("Weights from previous model loaded:", previous_model)
    else:
        callbacks = None
    return callbacks


def setup_params(dpmodel, experiment):
    epochs = dpmodel.config["train"]["model"]["epochs"]
    steps = dpmodel.config["train"]['model']["steps"]
    if dpmodel.config["train"]["comet_ml"]["track"]:
        params = dpmodel.config["train"]["model"]["params"]
        experiment.log_multiple_params(params)
    return epochs, steps


def init_tf_vars():
    keras.backend.get_session().run(tf.initialize_all_variables())


def close(dpmodel, crop_session, val_session):
    print("Complete! Closing session.", end=" ", flush=True)
    dpmodel.train_crop_generator.stop(crop_session)
    crop_session.close()
    val_session.close()
    print("All set.")
    gc.collect()
