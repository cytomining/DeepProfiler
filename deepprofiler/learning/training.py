from comet_ml import Experiment

import gc
import os
import numpy as np

import tensorflow as tf
import keras
from sklearn.metrics import confusion_matrix

import deepprofiler.learning.models
import deepprofiler.imaging.cropping
import deepprofiler.learning.validation

#################################################
## MAIN TRAINING ROUTINE
#################################################

def learn_model(config, dset, epoch):

    if not os.path.isdir(config["training"]["output"]):
        os.mkdir(config["training"]["output"])

    experiment = Experiment(
        api_key=config["validation"]["api_key"],
        project_name=config["validation"]["project_name"]
    )

    # Create cropping graph
    crop_graph = tf.Graph()
    with crop_graph.as_default():
        val_crop_generator = deepprofiler.imaging.cropping.SingleImageCropGenerator(config, dset)
        if config["model"]["type"] == "convnet":
            crop_generator = deepprofiler.imaging.cropping.CropGenerator(config, dset)
        elif config["model"]["type"] == "recurrent":
            crop_generator = deepprofiler.imaging.cropping.SetCropGenerator(config, dset)
        elif config["model"]["type"] == "mixup":
            crop_generator = deepprofiler.imaging.cropping.SetCropGenerator(config, dset)
        elif config["model"]["type"] == "mixup":
            crop_generator = deepprofiler.imaging.cropping.SetCropGenerator(config, dset)
        cpu_config = tf.ConfigProto( device_count={'CPU' : 1, 'GPU' : 0} )
        cpu_config.gpu_options.visible_device_list = ""
       
        crop_session = tf.Session(config = cpu_config)

        crop_generator.start(crop_session)
        gc.collect()
    # Start main session
    configuration = tf.ConfigProto()
    configuration.gpu_options.visible_device_list = "0"
    main_session = tf.Session(config = configuration)
    val_session = tf.Session(config = configuration)
    keras.backend.set_session(val_session)
    val_crop_generator.start(val_session)
    x_validation, y_validation = deepprofiler.learning.validation.validate(
            config,
            dset,
            val_crop_generator,
            val_session)
    keras.backend.set_session(main_session)

    if config["model"]["type"] == "convnet":
        input_shape = (
            config["sampling"]["box_size"],      # height 
            config["sampling"]["box_size"],      # width
            len(config["image_set"]["channels"]) # channels
        )
        model = deepprofiler.learning.models.create_keras_resnet(
                    input_shape, 
                    dset.targets,
                    config["validation"]["top_k"], 
                    config["training"]["learning_rate"], 
                    is_training=True
                )
    elif config["model"]["type"] == "recurrent":
        input_shape = (
            config["model"]["sequence_length"],  # time
            config["sampling"]["box_size"],      # height 
            config["sampling"]["box_size"],      # width
            len(config["image_set"]["channels"]) # channels
        )
        model = deepprofiler.learning.models.create_recurrent_keras_resnet(
                    input_shape, 
                    dset.targets, 
                    config["validation"]["top_k"],
                    config["training"]["learning_rate"], 
                    is_training=True
                )

    elif config["model"]["type"] == "mixup":
        input_shape = (
            config["sampling"]["box_size"],      # height 
            config["sampling"]["box_size"],      # width
            len(config["image_set"]["channels"]) # channels
        )
        model = deepprofiler.learning.models.create_keras_resnet(
                    input_shape, 
                    dset.targets,
                    config["validation"]["top_k"], 
                    config["training"]["learning_rate"], 
                    is_training=True
                )

    # keras-resnet model
    output_file = config["training"]["output"] + "/checkpoint_{epoch:04d}.hdf5"
    callback_model_checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=output_file,
        save_weights_only=True,
        save_best_only=False
    )
    csv_output = config["training"]["output"] + "/log.csv"
    callback_csv = keras.callbacks.CSVLogger(filename=csv_output)

    callbacks = [callback_model_checkpoint, callback_csv]


    previous_model = output_file.format(epoch=epoch-1)
    if epoch >= 1 and os.path.isfile(previous_model):
        model.load_weights(previous_model)
        print("Weights from previous model loaded", previous_model)
    else:
        print("Model does not exist:", previous_model)

    epochs = config["training"]["epochs"]
    steps = config["training"]["steps"]

    params={
        'steps_per_epoch':steps,
        'epochs':epochs,
        'input_shape':input_shape,
        'learning_rate':config["training"]["learning_rate"],
        "k_value":config["validation"]["top_k"],
        "training_batch_size":config["training"]["minibatch"],
        "validation_batch_size":config["validation"]["minibatch"]
    }
    experiment.log_multiple_params(params)

    model.fit_generator(
        generator=crop_generator.generate(crop_session),
        steps_per_epoch=steps,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
        initial_epoch=epoch-1,
        validation_data = (x_validation, y_validation)
    )

    pred=model.predict(x_validation)
    new_pred = []
    for line in pred:
        new_pred.append(np.argmax(line))
    new_y_validation = []
    for line in y_validation:
        new_y_validation.append(np.argmax(line))
    output_confusion_matrix = confusion_matrix(new_y_validation,new_pred)
    np.savetxt(config["training"]["output"]+"/confusion_matrix.txt",output_confusion_matrix)

    # Close session and stop threads
    print("Complete! Closing session.", end="", flush=True)
    crop_generator.stop(crop_session)
    crop_session.close()
    print(" All set.")
    gc.collect()
