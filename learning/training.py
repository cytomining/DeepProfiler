import os

import tensorflow as tf
import keras

import learning.models
import imaging.cropping


#################################################
## MAIN TRAINING ROUTINE
#################################################

def learn_model(config, dset, epoch):

    # Start session
    configuration = tf.ConfigProto()
    configuration.gpu_options.visible_device_list = "0"
    session = tf.Session(config = configuration)
    keras.backend.set_session(session)

    if config["model"]["type"] == "convnet":
        crop_generator = imaging.cropping.CropGenerator(config, dset)
        input_shape = (
            config["sampling"]["box_size"],      # height 
            config["sampling"]["box_size"],      # width
            len(config["image_set"]["channels"]) # channels
        )
        model = learning.models.create_keras_resnet(
                    input_shape, 
                    dset.targets, 
                    config["training"]["learning_rate"], 
                    is_training=True
                )
    elif config["model"]["type"] == "recurrent":
        crop_generator = imaging.cropping.SetCropGenerator(config, dset)
        input_shape = (
            config["model"]["sequence_length"],  # time
            config["sampling"]["box_size"],      # height 
            config["sampling"]["box_size"],      # width
            len(config["image_set"]["channels"]) # channels
        )
        model = learning.models.create_recurrent_keras_resnet(
                    input_shape, 
                    dset.targets, 
                    config["training"]["learning_rate"], 
                    is_training=True
                )

    elif config["model"]["type"] == "mixup":
        crop_generator = imaging.cropping.SetCropGenerator(config, dset)
        input_shape = (
            config["sampling"]["box_size"],      # height 
            config["sampling"]["box_size"],      # width
            len(config["image_set"]["channels"]) # channels
        )
        model = learning.models.create_keras_resnet(
                    input_shape, 
                    dset.targets, 
                    config["training"]["learning_rate"], 
                    is_training=True
                )
    elif config["model"]["type"] == "same_label_mixup":
        crop_generator = imaging.cropping.SetCropGenerator(config, dset)
        input_shape = (
            config["sampling"]["box_size"],      # height 
            config["sampling"]["box_size"],      # width
            len(config["image_set"]["channels"]) # channels
        )
        model = learning.models.create_keras_resnet(
                    input_shape, 
                    dset.targets, 
                    config["training"]["learning_rate"], 
                    is_training=True
                )

    crop_generator.start(session)

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
    model.fit_generator(
        generator=crop_generator.generate(session),
        steps_per_epoch=steps,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
        initial_epoch=epoch-1
    )

    # Close session and stop threads
    print("Complete! Closing session.", end="", flush=True)
    crop_generator.stop(session)
    session.close()
    print(" All set.")

