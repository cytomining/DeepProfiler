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
    #configuration.gpu_options.allow_growth = True
    configuration.gpu_options.visible_device_list = "0"
    session = tf.Session(config = configuration)
    keras.backend.set_session(session)

    crop_generator = imaging.cropping.CropGenerator(config)
    crop_generator.start(dset, session)

    # keras-resnet model
    output_file = config["training"]["output"] + "/checkpoint_{epoch:04d}.hdf5"
    callback_model_checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=output_file,
        save_weights_only=True,
        save_best_only=False
    )
    csv_output = config["training"]["output"] + "/log.csv"
    callback_csv = keras.callbacks.CSVLogger(filename=csv_output)

    epochs = 100

    def lrs(e):
        new_lr = config["training"]["learning_rate"]
        if    .0 <= e/100 < .30: new_lr /= 1.
        elif .30 <= e/100 < .60: new_lr /= 10.
        elif .60 <= e/100 < .80: new_lr /= 100.
        elif .80 <= e/100     : new_lr /= 1000.
        print("Learning rate:", new_lr)
        return new_lr
         
    lr_schedule = keras.callbacks.LearningRateScheduler(schedule=lrs)
    callbacks = [callback_model_checkpoint, callback_csv, lr_schedule]

    input_shape = (
        config["sampling"]["box_size"],      # height 
        config["sampling"]["box_size"],      # width
        len(config["image_set"]["channels"]) # channels
    )
    model = learning.models.create_keras_resnet(input_shape, dset.numberOfClasses())
    optimizer = keras.optimizers.Adam(lr=config["training"]["learning_rate"])
    model.compile(optimizer, "categorical_crossentropy", ["accuracy"])

    previous_model = output_file.format(epoch=epoch-1)
    if epoch > 1 and os.path.isfile(previous_model):
        model.load_weights(previous_model)
        print("Weights from previous model loaded", previous_model)
    else:
        print("Model does not exist:", previous_model)


    steps = config["training"]["iterations"] / epochs
    model.fit_generator(
        generator=crop_generator.generate(session),
        steps_per_epoch=steps,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
        initial_epoch=epoch
    )

    # Close session and stop threads
    print("Complete! Closing session.", end="", flush=True)
    crop_generator.stop(session)
    session.close()
    print(" All set.")

