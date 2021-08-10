"""
Helper functions for checking images, locations and crops before running profile and train.
"""

import pandas as pd
import numpy as np
import cv2
import os

def imgs_dont_exist(ls, image_dir, channels):
    """ Adds images to a list if those images are not found.

    Parameters
    ----------
    ls : empty list, will be filled with missing images
    image_dir : directory to the images
    channels : different channels for each image
    """
    for img in channels:
        if not os.path.isfile(os.path.join(image_dir, img)):
            ls.append(img)
    return None

def locs_dont_exist(ls, locs_dir, loc):
    """ Adds location files to a list if they are missing are not found.

    Parameters
    ----------
    ls : empty list, will be filled with missing files
    locs_dir : directory to the location files
    """
    if not os.path.isfile(os.path.join(locs_dir, loc)):
        ls.append(loc)
    return None


def check_profile(dset):
    """Checks images and location files to prepare for the profiling function.
    If this function runs correctly, the function 'profile' will also run without errors.

    Parameters
    ----------
    config :
    dset :

    Returns
    -------

    """
    project_dir = '/Users/mbornhol/git/DeepProf/DP2'
    feat_rows = ['DNA','Tubulin','Actin']

    # Checking images
    index = pd.read_csv('/Users/mbornhol/git/DeepProf/DP2/inputs/metadata/index.csv')
    image_dir = os.path.join(project_dir, 'inputs', 'images')
    ls = []

    # use this: row[dset.channels]
    index.apply(lambda row: imgs_dont_exist(ls, image_dir, row[feat_rows]), axis=1)
    pd.DataFrame(ls, columns=['missing_images']).to_csv('missing_images.csv', index=False)


    # Checking location files
    # image_dir = os.path.join(project_dir, 'outputs', 'images')
    # ls = []
    # index.apply(lambda row: locs_dont_exist(ls, dset.locations, row), axis=1)
    # pd.DataFrame(ls, columns=['missing_locations']).to_csv('missing_locations.csv', index=False)\
    return ls


"""
Checking all crops before training.
"""

def crop_checks(ls_missing, ls_zero, img_name, sample_dir):
    if not os.path.isfile(os.path.join(sample_dir, img_name)):
        ls_missing.append(img_name)
    else:
        img = cv2.imread(os.path.join(sample_dir, img_name), cv2.IMREAD_GRAYSCALE)
        pos = np.nonzero(img)
        if len(pos[0]) == 0:
            ls_zero.append(img_name)


def check_training(dset):
    """Check all crops before training in order to avoid errors during training.

    Returns
    -------

    """
    # First check if images exist
    crops_dir = '/Users/mbornhol/git/DeepProf/DP2/outputs/single-cell-sample'
    # use dset.sample_directory?
    df = pd.read_csv(os.path.join(crops_dir, 'sc-metadata.csv'))

    ls_missing = []
    ls_zero = []
    res = df.apply(lambda row: crop_checks(ls_missing, ls_zero, row['Image_Name'], crops_dir), axis = 1)

    pd.DataFrame(ls_missing, columns=['missing_crops']).to_csv('missing_crops.csv', index=False)
    pd.DataFrame(ls_zero, columns=['zero_crops']).to_csv('zero_crops.csv', index=False)