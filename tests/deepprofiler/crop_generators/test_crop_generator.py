import deepprofiler.crop_generators.crop_generator
import deepprofiler.imaging.cropping


def test_crop_generator():
    assert deepprofiler.crop_generators.crop_generator.GeneratorClass == deepprofiler.imaging.cropping.CropGenerator
    assert deepprofiler.crop_generators.crop_generator.SingleImageGeneratorClass == deepprofiler.imaging.cropping.SingleImageCropGenerator
