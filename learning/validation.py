import os
import numpy as np
import tensorflow as tf

import imaging.boxes
import learning.metrics
import learning.models
import learning.training

import keras


def output_name(config, meta):
    # Save features TODO: parameterize
    filename = os.path.join(
            config["profiling"]["output_dir"],
            str(meta["Metadata_Plate"]) + "_" +
            str(meta["Metadata_Well"]) + "_" +
            str(meta["Metadata_Site"]) + ".npz"
        )
    return filename


def validate(config, dset, checkpoint_file):
    config["queueing"]["min_size"] = 0
    # TODO: Number of classes should be in the config file?
    #num_classes = dset.numberOfClasses()
    # TODO: parameterize all these constants
    with_k = 2
    num_features = 2048
    layer_id = 176
    batch_size = config["training"]["minibatch"] 

    configuration = tf.ConfigProto()
    configuration.gpu_options.allow_growth = True
    configuration.gpu_options.visible_device_list = "0"
    session = tf.Session(config=configuration)
    keras.backend.set_session(session)

    input_shape = (
        config["sampling"]["box_size"],      # height
        config["sampling"]["box_size"],      # width
        len(config["image_set"]["channels"]) # channels
    )
    model = learning.models.create_keras_resnet(input_shape, dset.targets)
    print("Checkpoint:",checkpoint_file)
    model.load_weights(checkpoint_file)
    # TODO: Parameterize the layer from which features are computed
    features = model.layers[layer_id].output
    feat_extractor = keras.backend.function([model.input] + [keras.backend.learning_phase()], [features])

    metrics = []
    for i in range(len(dset.targets)):
        tgt = dset.targets[i]
        mtr = learning.metrics.Metrics(k=with_k)
        mtr.configure_ops(tgt.field_name, batch_size, tgt.shape[1])
        metrics.append(mtr)

    crop_generator = imaging.cropping.SingleImageCropGenerator(config, dset)
    crop_generator.start(session)

    # TODO: Convert the check and predict functions into methods of a class
    # that the scan method of the dataset can interact with
    def check(meta):
        filename = output_name(config, meta)
        if os.path.isfile(filename):
            print(filename, "already done")
            return False
        else:
            return True
   

    def predict(key, image_array, meta):
        filename = output_name(config, meta)
        total_crops, pads = crop_generator.prepare_image(session, image_array, meta)
        features = np.zeros(shape=(total_crops, num_features))
        bp = 0
        for batch in crop_generator.generate(session):
            result = model.predict(batch[0])
            # TODO: parameterize extracting and saving features
            f = feat_extractor((batch[0],0))
            features[bp * batch_size:(bp + 1) * batch_size, :] = f[0]

            for metric in metrics:
                corr, intop = session.run(
                    [metric.correct_op, metric.in_top_k_op],
                    feed_dict={metric.true_labels:batch[1], metric.predictions:result}
                )
                metric.update(corr, intop, batch[0].shape[0])
            bp += 1

        features = features[:-pads, :]
        np.savez_compressed(filename, f=features)
        print(filename, features.shape)
        #metrics.print(with_k)

    # TODO: parameterize the data frame to loop through 
    dset.scan(predict, frame="all", check=check)
    print("Validate: done")

