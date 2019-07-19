import numpy as np

import deepprofiler.imaging.cropping


def repeat_channels(crops, labels, repeats):
    crops = np.reshape(crops, (crops.shape[0] * crops.shape[3], crops.shape[1], crops.shape[2], 1))
    crops = np.tile(crops, (1, 1, 1, repeats))
    labels = np.tile(labels, (repeats, 1))
    return crops, labels


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
            global_step += 1
            yield (crops, labels)


class SingleImageGeneratorClass(deepprofiler.imaging.cropping.SingleImageCropGenerator):

    def generate(self, session, global_step=0):
        crops = self.image_pool
        labels = self.label_pool
        crops, labels = repeat_channels(crops, labels, self.config["dataset"]["images"]["channel_repeats"])
        yield [crops, labels]
