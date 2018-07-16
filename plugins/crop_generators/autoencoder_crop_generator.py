import numpy as np

import deepprofiler.imaging.cropping


class GeneratorClass(deepprofiler.imaging.cropping.CropGenerator):  # TODO: this is a hack, rewrite entire class
    def generate(self, sess, global_step=0):
        pool_index = np.arange(self.image_pool.shape[0])
        while True:
            if self.coord.should_stop():
                break
            data = self.sample_batch(pool_index)
            global_step += 1
            yield (data[0], data[0])

class SingleImageGeneratorClass(deepprofiler.imaging.cropping.SingleImageCropGenerator):
    def generate(self, session, global_step=0):
        yield [self.image_pool, self.image_pool]
