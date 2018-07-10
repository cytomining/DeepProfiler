import numpy as np

from deepprofiler.imaging.cropping import CropGenerator


class AutoencoderCropGenerator(CropGenerator):  # TODO: this is a hack, rewrite entire class
    def generate(self, sess, global_step=0):
        pool_index = np.arange(self.image_pool.shape[0])
        while True:
            if self.coord.should_stop():
                break
            data = self.sample_batch(pool_index)
            global_step += 1
            yield (data[0], data[0])


def define_crop_generator(config, dset):
    return AutoencoderCropGenerator(config, dset)
