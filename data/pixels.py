import skimage.io
import numpy as np
import pickle as pickle

#################################################
## COMMON IMAGE HANDLING OPPERATIONS
#################################################

# Main image reading function. Images are treated as 3D tensors: (height, width, channels)
def openImage(paths, pixelProcessor):
    channels = [ skimage.io.imread(p) for p in paths ]
    img = np.zeros( (channels[0].shape[0], channels[0].shape[1], len(channels)) )
    for c in range(len(channels)):
        img[:,:,c] = channels[c]
    return pixelProcessor.run(img)

#################################################
## PIXEL PROCESSING CLASSES
#################################################

# Abstract class to extend operations that can be applied to images while reading them
class PixelProcessor():

    def process(self, pixels):
        return pixels

    def run(self, pixels):
        return self.process(pixels)

