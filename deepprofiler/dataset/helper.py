"""
Helper functions for checking images, locations and crops before running profile and train.
"""

import pandas as pd
import numpy as np
import cv2
import os
import deepprofiler.imaging.boxes


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
    ls_imgs, ls_locs = [], []
    os.makedirs('checks', exist_ok=True)

    frame = dset.meta.data.iterrows()
    images = [dset.get_image_paths(r) for i, r in frame]
    for channels in images:
        for img in channels[1]:
            if not os.path.isfile(img):
                ls_imgs.append(img)
    print('found {} missing images'.format(len(ls_imgs)), '|| saving list of missing files to checks/')
    pd.DataFrame(ls_imgs, columns=['missing_images']).to_csv('checks/missing_images.csv', index=False)

    # start checking location files
    frame = dset.meta.data.iterrows()
    for i, r in frame:
        df = deepprofiler.imaging.boxes.get_single_cell_locations("{}/{}-{}".format(r["Metadata_Plate"], r["Metadata_Well"], r["Metadata_Site"]), dset.config)
        if df.empty:
            ls_locs.append("{}/{}-{}".format(r["Metadata_Plate"], r["Metadata_Well"], r["Metadata_Site"]))

    print('found {} missing location files'.format(len(ls_locs)), '|| saving list of missing files to checks/')
    pd.DataFrame(ls_locs, columns=['missing_locs']).to_csv('checks/missing_locs.csv', index=False)

    return ls_imgs, ls_locs


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


def check_train(dset):
    """Check all crops before training in order to avoid errors during training.

    Returns
    -------

    """
    # print(dset.meta.data.columns)
    # First check if images exist
    df = # read sc-metadata file

    ls_missing = []
    ls_zero = []
    res = df.apply(lambda row: crop_checks(ls_missing, ls_zero, row['Image_Name'], crops_dir), axis = 1)

    pd.DataFrame(ls_missing, columns=['missing_crops']).to_csv('missing_crops.csv', index=False)
    pd.DataFrame(ls_zero, columns=['zero_crops']).to_csv('zero_crops.csv', index=False)