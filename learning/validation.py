import tensorflow as tf
import numpy as np

import dataset.utils
import learning.models
import learning.cropping
import learning.training


def validate(config, dset):
    num_classes = dset.numberOfClasses()
    input_vars = learning.training.input_graph(config)
    images = input_vars["labeled_crops"][0]
    labels = tf.one_hot(input_vars["labeled_crops"][1], num_classes)
    net = learning.models.create_resnet(images, num_classes, is_training=True)
    #net = learning.models.create_vgg(images, num_classes, is_training=False)
    saver = tf.train.Saver()

    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.per_process_gpu_memory_fraction = config["queueing"]["gpu_mem_fraction"]
    sess = tf.Session(config=gpu_config)

    validation_ops, val_writer = learning.models.create_validator(net, labels, sess, config)
    # Run training ops instead
    #validation_ops, val_writer = learning.models.create_trainer(net, labels, sess, config)
    #validation_ops = validation_ops[1:]
    #sess.run(tf.global_variables_initializer())

    #saver.restore(sess, "/data1/luad/experiments/5class-6/model/weights.ckpt-4000")
    #saver.restore(sess, "/data1/luad/experiments/exp9/model/weights.ckpt-80000")
    saver.restore(sess, "/data1/luad/debug/model/weights.ckpt-8000")


    def predict(key, image_array, meta):
        image_key, image_names = dset.getImagePaths(meta)
        locations = [ learning.cropping.getLocations(image_key, config, randomize=False) ]
        labels_data = [ meta[config["training"]["label_field"]] ]
        boxes, box_ind, labels_data = learning.cropping.prepareBoxes(locations, labels_data, config)
        images_data = np.reshape(image_array, input_vars["shapes"]["batch"])

        sess.run(input_vars["enqueue_op"], {
                        input_vars["image_ph"]:images_data,
                        input_vars["boxes_ph"]:boxes,
                        input_vars["box_ind_ph"]:box_ind,
                        input_vars["labels_ph"]:labels_data
        })
        items = sess.run(input_vars["queue"].size())
        while items > config["training"]["minibatch"]:
            #to, loss, acc, top5, gt, pr, vw = sess.run(validation_ops)
            loss, acc, top5, gt, pr, vw = sess.run(validation_ops)
            val_writer.add_summary(vw)
            print(acc, loss, top5)
            print(gt, pr)
            items = sess.run(input_vars["queue"].size())
        print(" *",image_key, "done")

    dset.scan(predict, frame="val")
    print("Validate: done")

 
