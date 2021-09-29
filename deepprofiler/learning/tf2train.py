import comet_ml
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
import efficientnet.tfkeras as efn

from deepprofiler.imaging.augmentations import AugmentationLayerV2

tf.compat.v1.enable_v2_behavior()
tf.config.run_functions_eagerly(True)

AUTOTUNE = tf.data.AUTOTUNE


def make_dataset(path, batch_size, single_cell_metadata, config, is_training):
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
        ds = ds.shuffle(buffer_size=5000)
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


def setup_callbacks(config):
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
    elif "checkpoint_policy" in config["train"]["model"] and config["train"]["model"]["checkpoint_policy"] == 'best':
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
            callback_lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                                                        patience=5, min_lr=0.0001)
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


def setup_comet_ml(config):
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


def learn_model(config, epoch):
    DENSE_KERNEL_INITIALIZER = {
        'class_name': 'VarianceScaling',
        'config': {
            'scale': 1. / 3.,
            'mode': 'fan_out',
            'distribution': 'uniform'
        }
    }

    BATCH_SIZE = config["train"]["model"]["params"]["batch_size"]
    experiment = setup_comet_ml(config)
    single_cell_metadata = pd.read_csv(os.path.join(config["paths"]["index"]))

    num_classes = len(pd.unique(single_cell_metadata["Class_Name"]))
    single_cell_metadata["Categorical"] = pd.Categorical(single_cell_metadata["Class_Name"]).codes

    path = config["paths"]["single_cell_sample"]
    dataset, steps_per_epoch = make_dataset(path, BATCH_SIZE,
                                            single_cell_metadata[
                                                single_cell_metadata[config["train"]["partition"]["split_field"]].isin(
                                                    config["train"]["partition"]["training_values"])],
                                            config,
                                            is_training=True)
    validation_dataset, _ = make_dataset(path, BATCH_SIZE,
                                         single_cell_metadata[
                                             single_cell_metadata[config["train"]["partition"]["split_field"]].isin(
                                                 config["train"]["partition"]["validation_values"])],
                                         config,
                                         is_training=False)

    input_shape = (config["dataset"]["locations"]["box_size"], config["dataset"]["locations"]["box_size"],
                   len(config["dataset"]["images"]["channels"]))
    input_image = tf.keras.layers.Input(input_shape)
    augmented_input = AugmentationLayerV2(input_shape=input_shape)(input_image)

    model = efn.EfficientNetB0(
        include_top=False, weights=None, input_tensor=augmented_input,
        input_shape=input_shape
    )
    features = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(model.output)
    y = tf.keras.layers.Dense(num_classes, activation='softmax', name='predictions',
                              kernel_initializer=DENSE_KERNEL_INITIALIZER)(features)
    model = tf.keras.models.Model(inputs=input_image, outputs=y)

    regularizer = tf.keras.regularizers.l2(0.00001)
    for layer in model.layers:
        if hasattr(layer, "kernel_regularizer"):
            setattr(layer, "kernel_regularizer", regularizer)

    model = tf.keras.models.model_from_json(model.to_json(), custom_objects={'AugmentationLayerV2': AugmentationLayerV2})
    optimizer = tf.keras.optimizers.SGD(learning_rate=config["train"]["model"]["params"]["learning_rate"])
    loss_func = tf.keras.losses.CategoricalCrossentropy(from_logits=False)  # , label_smoothing = 0.6)

    model.compile(optimizer, loss_func,
                  metrics=["accuracy"])#, tfa.metrics.F1Score(num_classes=num_classes, average='macro'),
                           #tf.keras.metrics.TopKCategoricalAccuracy(k=5), tf.keras.metrics.Precision()])

    callbacks = setup_callbacks(config)

    if epoch == 1 and config["train"]["model"]["initialization"] == "ImageNet":
        base_model = efn.EfficientNetB0(weights='imagenet', include_top=False)
        total_layers = len(base_model.layers)
        for i in range(2, total_layers):
            if len(base_model.layers[i].weights) > 0:
                model.layers[i].set_weights(base_model.layers[i].get_weights())

        # => Replicate filters of first layer as needed

        weights = base_model.layers[1].get_weights()
        available_channels = weights[0].shape[2]
        target_shape = model.layers[1].weights[0].shape
        new_weights = np.zeros(target_shape)

        for i in range(new_weights.shape[2]):
            j = i % available_channels
            new_weights[:, :, i, :] = weights[0][:, :, j, :]

        weights_array = [new_weights]
        if len(weights) > 1:
            weights_array += weights[1:]

        model.layers[1].set_weights(weights_array)

        print("Network initialized with pretrained ImageNet weights")

    elif epoch > 1:
        output_file = config["paths"]["checkpoints"] + "/checkpoint_{epoch:04d}.hdf5"
        previous_model = output_file.format(epoch=epoch - 1)
        model.load_weights(previous_model)

    #print(model.summary())
    if experiment:
        with experiment.train():
            model.fit(dataset,
                      epochs=config["train"]["model"]["epochs"],
                      callbacks=callbacks,
                      verbose=1,
                      validation_data=validation_dataset,
                      validation_freq=config["train"]["validation"]["frequency"],
                      initial_epoch=epoch - 1
                      )
    else:
        model.fit(dataset,
                  epochs=config["train"]["model"]["epochs"],
                  callbacks=callbacks,
                  verbose=1,
                  validation_data=validation_dataset,
                  validation_freq=config["train"]["validation"]["frequency"],
                  initial_epoch=epoch - 1
                  )
