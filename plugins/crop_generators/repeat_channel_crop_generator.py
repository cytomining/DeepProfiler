import numpy as np
import tensorflow as tf
import deepprofiler.imaging.cropping
from keras.applications import inception_resnet_v2

def repeat_channels(crops, labels, repeats):
    crops = crops[np.newaxis, :]
    labels = labels[np.newaxis, :]
    print('crops begin', crops.shape)
    print('crops begin', labels.shape)
    print(repeats)
    crops = np.reshape(crops, (crops.shape[0] * crops.shape[3], crops.shape[1], crops.shape[2], 1))
    print(crops.shape)
    crops = np.tile(crops, (1, 1, 1, repeats))
    print(crops.shape, crops.__class__)
    print(labels.shape)
    labels = np.tile(labels, (repeats, 1))
    print(labels.shape)
    #crops = tf.image.resize_images(crops, size=(299, 299))
    print(crops.shape, crops.__class__)
    return crops, labels
    #return inception_resnet_v2.preprocess_input(crops), labels
    #def crop_transform(crop_ph, image_size):
    #crops_shape = crops.shape
    #resized_crops = tf.image.resize_images(crops, size=(299, 299))
    #print(resized_crops.shape)
    #reordered_channels = tf.transpose(resized_crops, [3, 0, 1, 2])
    #print(reordered_channels.shape)
    #reshaped_data = tf.reshape(reordered_channels, shape=[-1, 299, 299, 1])
    #print(reshaped_data.shape)
    #rgb_data = tf.image.grayscale_to_rgb(reordered_channels)
    #print(rgb_data.shape)
    #return inception_resnet_v2.preprocess_input(rgb_data), labels


class GeneratorClass(deepprofiler.imaging.cropping.CropGenerator):

    def generate(self, sess, global_step=0):
        pool_index = np.arange(self.image_pool.shape[0])
        while True:
            if self.coord.should_stop():
                break
            data = self.sample_batch(pool_index)
            crops = data[0]
            labels = data[1]  # TODO: enable multiple targets
            crops, labels = repeat_channels(crops, labels, self.config["dataset"]["images"]["channel_repeats"])
            crops = crops.eval(session=sess)
            global_step += 1
            yield (crops, labels)


class SingleImageGeneratorClass(deepprofiler.imaging.cropping.SingleImageCropGenerator):

    def generate(self, session, global_step=0):
        crops = self.image_pool
        labels = self.label_pool
        #new_crops = np.array([[[[]]]])
        #new_labels = np.array([[[[]]]])
        for i in range(crops.shape[0]):
            crop, label = repeat_channels(crops[i, :, :, :], labels[i, :], self.config["dataset"]["images"]["channel_repeats"])
            #crop = crop.eval(session=session)
            crop = session.run(tf.image.resize_images(crop, size=(299, 299)))
            if i == 0:
                new_crops = crop
                new_labels = label
            else:
                new_crops = np.concatenate((new_crops, crop), axis=0)
                new_labels = np.concatenate((new_labels, label), axis=0)

        #print('last shape', crop.shape, label.shape)
        #print('generator shapes', new_crops.shape, new_labels.shape)
        #new_crops = new_crops.eval(session=session)
        #crops, labels = repeat_channels(crops, labels, self.config["dataset"]["images"]["channel_repeats"])
        #crops = crops.eval(session=session)
        yield [new_crops, new_labels]
