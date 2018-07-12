import gc
import os

import tensorflow as tf
import keras

import deepprofiler.learning.models
import deepprofiler.imaging.cropping


#################################################
## MAIN TRAINING ROUTINE
#################################################

def learn_model(config, dset, epoch):

    # Create cropping graph
    crop_graph = tf.Graph()
    with crop_graph.as_default():
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
    model.fit_generator(
        generator=crop_generator.generate(crop_session),
        steps_per_epoch=steps,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
        initial_epoch=epoch-1
    )

    # Close session and stop threads
    print("Complete! Closing session.", end="", flush=True)
    crop_generator.stop(crop_session)
    crop_session.close()
    print(" All set.")
    gc.collect()
