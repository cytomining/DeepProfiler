"""
Helper functions for checking images, locations and crops before running profile and train.
"""

import pandas as pd
import numpy as np
import cv2
import os
import tensorflow as tf

import deepprofiler.imaging.boxes
import plugins.crop_generators.sampled_crop_generator


def check_profile(dset):
    """Checks images and location files to prepare for the profiling function.
    If this function runs correctly, the function 'profile' will also run without errors.
    The names of the missing files are saved in two different files.

    Parameters
    ----------
    dset : Data structure with metadata and location files

    Returns
    -------
    ls_imgs : list of missing images
    ls_locs : list of missing location files

    """
    ls_imgs, ls_locs = [], []
    os.makedirs("checks", exist_ok=True)

    # start checking image files
    frame = dset.meta.data.iterrows()
    images = [dset.get_image_paths(r) for i, r in frame]

    for channels in images:
        for img in channels[1]:
            if not os.path.isfile(img):
                ls_imgs.append(img)
    print(
        ">>> found {} missing images".format(len(ls_imgs)),
        "|| saving list of missing files to checks/missing_images.csv",
    )
    pd.DataFrame(ls_imgs, columns=["missing_images"]).to_csv(
        "checks/missing_images.csv", index=False
    )

    # start checking location files
    frame = dset.meta.data.iterrows()
    for i, r in frame:
        df = deepprofiler.imaging.boxes.get_single_cell_locations(
            "{}/{}-{}".format(
                r["Metadata_Plate"], r["Metadata_Well"], r["Metadata_Site"]
            ),
            dset.config,
        )
        if df.empty:
            ls_locs.append(
                "{}/{}-{}".format(
                    r["Metadata_Plate"], r["Metadata_Well"], r["Metadata_Site"]
                )
            )

    print(
        ">>> found {} missing location files".format(len(ls_locs)),
        "|| saving list of missing files to checks/missing_locs.csv",
    )
    pd.DataFrame(ls_locs, columns=["missing_locs"]).to_csv(
        "checks/missing_locs.csv", index=False
    )

    return ls_imgs, ls_locs


def crop_checks(img_name, ls_missing, ls_zero):
    """Utility function for check_train to check images for existence and non-zero values.
    Parameters
    ----------
    img_name : crop image name

    Returns
    -------
    ls_missing, ls_zero : lists detailing the missing crops and the zero crops

    """
    if not os.path.isfile(img_name):
        ls_missing.append(img_name)
    else:
        img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
        pos = np.nonzero(img)
        if len(pos[0]) == 0:
            ls_zero.append(img_name)


def check_train(config, dset):
    """Checks if the data is ready for training by checking if the crops are sampled correctly.
    Missing and zero crops are saved into two files.
    Parameters
    ----------
    config : config input
    dset : Data structure with metadata

    Returns
    -------
    ls_missing, ls_zero : lists of missing and zero crops

    """
    os.makedirs("checks", exist_ok=True)

    crop_generator = plugins.crop_generators.sampled_crop_generator.GeneratorClass(
        config, dset
    )
    sess = tf.compat.v1.Session()
    crop_generator.start(sess)
    df = crop_generator.samples

    ls_missing, ls_zero = [], []
    res = df.apply(
        lambda row: crop_checks(
            os.path.join(config["paths"]["single_cell_sample"], row["Image_Name"]),
            ls_missing,
            ls_zero,
        ),
        axis=1,
    )

    print(
        ">>> found {} missing crops".format(len(ls_missing)),
        "|| saving list of missing crops to checks/missing_crops.csv",
    )
    pd.DataFrame(ls_missing, columns=["missing_crops"]).to_csv(
        "checks/missing_crops.csv", index=False
    )
    print(
        ">>> found {} crops with zero values".format(len(ls_zero)),
        "|| saving list of zero crops to checks/missing_crops.csv",
    )
    pd.DataFrame(ls_zero, columns=["zero_crops"]).to_csv(
        "checks/zero_crops.csv", index=False
    )

    return ls_missing, ls_zero
