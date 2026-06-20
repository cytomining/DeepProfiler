"""Low-level image reading utilities.

Provides :func:`openImage`, the single point in the codebase where image
files are read from disk.  All channel files for one site are loaded and
stacked into a single ``(H, W, C)`` float64 array.  An optional object-mask
channel can be appended when cell outline images are supplied.
"""

import numpy as np
import skimage.io
import skimage.measure


def openImage(paths, outlines):
    """Load a multi-channel microscopy image from per-channel files.

    Reads each channel file listed in ``paths`` with :func:`skimage.io.imread`
    and stacks them along the last axis to produce a ``(H, W, C)`` array.
    Pixel values are kept in their native range (e.g. uint16 for 16-bit TIFFs)
    without normalisation — normalisation happens later in the crop graph.

    If ``outlines`` is provided, the outline image is read, connected
    components are labelled with :func:`skimage.measure.label`, and the label
    map is appended as an extra channel.  The crop graph can use this channel
    to zero out pixels that belong to neighbouring cells
    (``mask_objects: true`` in config).

    Args:
        paths: List of file paths, one per channel, in channel order matching
            ``config["dataset"]["images"]["channels"]``.
        outlines: Path to a cell-outline image, or ``None`` to skip masking.

    Returns:
        numpy array of shape ``(H, W, C)`` (float64), or ``(H, W, C+1)`` if
        outlines are provided.
    """
    channels = [skimage.io.imread(p) for p in paths]
    img = np.zeros((channels[0].shape[0], channels[0].shape[1], len(channels)))
    for c in range(len(channels)):
        img[:, :, c] = channels[c]
    if outlines is not None:
        boundaries = skimage.io.imread(outlines)
        labels = skimage.measure.label(boundaries, background=1)
        img = np.concatenate((img, labels[:, :, np.newaxis]), axis=2)
    return img
