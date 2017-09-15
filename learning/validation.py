import tensorflow as tf
import numpy as np
import pandas as pd
import os

import dataset.utils
import learning.models
import imaging.boxes
import learning.training

import keras


# TODO: Move metrics to some other module
class Metrics():

    def __init__(self):
        self.correct = 0.0
        self.in_topK = 0.0
        self.counts = 0.0

    def update(self, corr, topK, counts):
        self.correct += corr
        self.in_topK += topK
        self.counts += counts

    def print(self, with_k):
        message = "Acc: {:0.4f} Top-{}: {:0.4f} Samples: {:0.0f}"
        acc = self.correct/self.counts
        topk = self.in_topK/self.counts
        print(message.format(acc, with_k, topk, self.counts))


def validate(config, dset, checkpoint_file):
    config["queueing"]["min_size"] = 0
    # TODO: Number of classes should be in the config file?
    #num_classes = dset.numberOfClasses()
    num_classes = 594
    # TODO: parameterize all these constants
    with_k = 2
    num_features = 2048
    layer_id = 176
    batch_size = config["training"]["minibatch"] 

    input_vars = learning.training.input_graph(config)
    images = input_vars["labeled_crops"][0]
    labels = tf.one_hot(input_vars["labeled_crops"][1], num_classes)

    input_shape = (
        config["sampling"]["box_size"],      # height
        config["sampling"]["box_size"],      # width
        len(config["image_set"]["channels"]) # channels
    )
    model = learning.models.create_keras_resnet(input_shape, num_classes)
    true_labels = tf.placeholder(tf.float32, shape=(batch_size, num_classes))
    predictions = tf.placeholder(tf.float32, shape=(batch_size, num_classes))

    # TODO: Move metrics to somewhere else
    correct = tf.reduce_sum(
        tf.to_float(
            tf.equal(
                tf.argmax(true_labels, 1), 
                tf.argmax(predictions, 1)
            )
        )
    )
    in_top_k = tf.reduce_sum(
        tf.to_float( 
            tf.nn.in_top_k(
                predictions=predictions,
                targets=tf.argmax(true_labels, 1),
                k=with_k
            )
        )
    )

    configuration = tf.ConfigProto()
    configuration.gpu_options.allow_growth = True
    configuration.gpu_options.visible_device_list = "0"
    session = tf.Session(config = configuration)
    keras.backend.set_session(session)

    print("Checkpoint:",checkpoint_file)
    model.load_weights(checkpoint_file)
    # TODO: Parameterize the layer from which features are computed
    features = model.layers[layer_id].output
    feat_extractor = keras.backend.function( [model.input] + [keras.backend.learning_phase()], [features] )

    metrics = Metrics()

    # TODO: Convert the check and predict functions into methods of a class
    # that the scan method of the dataset can interact with
    def check(meta):
        filename = os.path.join(
                        config["profiling"]["output_dir"], 
                        str(meta["Metadata_Plate"]) + "_" + 
                        str(meta["Metadata_Well"])  + "_" + 
                        str(meta["Metadata_Site"])  + ".npz"
                   )
        if os.path.isfile(filename):
            print(filename, "already done")
            return False
        else:
            return True
   

    def predict(key, image_array, meta):
        # Save features TODO: parameterize
        filename = os.path.join(
                        config["profiling"]["output_dir"], 
                        str(meta["Metadata_Plate"]) + "_" + 
                        str(meta["Metadata_Well"])  + "_" + 
                        str(meta["Metadata_Site"])  + ".npz"
                   )

        image_key, image_names, outlines = dset.getImagePaths(meta)

        batch = {"images":[], "locations":[], "labels":[]}
        batch["images"].append(image_array)
        batch["locations"].append(imaging.boxes.getLocations(image_key, config, randomize=False))
        batch["labels"].append(meta[config["training"]["label_field"]])

        # Add trailing locations to fit the batch size
        pads = batch_size - len(batch["locations"][0]) % batch_size
        padding = pd.DataFrame(columns=batch["locations"][0].columns, data=np.zeros(shape=(pads, 2), dtype=np.int32))
        batch["locations"][0] = pd.concat((batch["locations"][0], padding), ignore_index=True)

        boxes, box_ind, labels_data, mask_ind = imaging.boxes.prepareBoxes(batch, config)
        batch["images"] = np.reshape(image_array, input_vars["shapes"]["batch"])

        session.run(input_vars["enqueue_op"], {
                        input_vars["image_ph"]:batch["images"],
                        input_vars["boxes_ph"]:boxes,
                        input_vars["box_ind_ph"]:box_ind,
                        input_vars["labels_ph"]:labels_data,
                        input_vars["mask_ind_ph"]:mask_ind
        })
        # TODO: push crops to the queue and process them in parallel.
        # No need for this to be sequential

        items = session.run(input_vars["queue"].size())
        features = np.zeros(shape=(len(batch["locations"][0]), num_features))
        bp = 0
        while items >= batch_size:
            batch = session.run([images, labels])
            result = model.predict(batch[0])

            # TODO: parameterize extracting and saving features
            f = feat_extractor((batch[0],0))
            features[bp * batch_size:(bp + 1) * batch_size, :] = f[0]

            items, corr, intop = session.run(
                [input_vars["queue"].size(), correct, in_top_k], 
                feed_dict={true_labels:batch[1], predictions:result}
            )
            metrics.update(corr, intop, batch[0].shape[0])
            bp += 1

        features = features[:-pads, :]
        np.savez_compressed(filename, f=features)
        print(filename, features.shape)
        #metrics.print(with_k)

    # TODO: parameterize the data frame to loop through 
    dset.scan(predict, frame="all", check=check)
    print("Validate: done")

 
