from deepprofiler.imaging.cropping import CropGenerator


def define_crop_generator(config, dset):
    return CropGenerator(config, dset)
