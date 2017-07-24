import numpy as np
import skimage.io
import skimage.measure


#################################################
## COMMON IMAGE HANDLING OPPERATIONS
#################################################

# Main image reading function. Images are treated as 3D tensors: (height, width, channels)
def openImage(paths, outlines):
    channels = [ skimage.io.imread(p) for p in paths ]
    img = np.zeros( (channels[0].shape[0], channels[0].shape[1], len(channels)) )
    for c in range(len(channels)):
        img[:,:,c] = channels[c] / 255.0
    if outlines is not None:
        boundaries = skimage.io.imread(outlines)
        labels = skimage.measure.label(boundaries, background=1)
        img = np.concatenate( (img, labels[:,:,np.newaxis]), axis=2)
    return img

