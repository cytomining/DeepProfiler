"""
Helper functions for checking images, locations and crops before running profile and train.
"""

import pandas as pd
import numpy as np
import os.path

def check_imgs(ls, image_dir, channels):
    for img in channels:
        if not os.path.isfile(os.path.join(image_dir, img)):
            ls.append(img)
    return ls

def check_profile(config, dset):
    index = pd.read_csv('metadata/top20_moa.csv')
    image_dir = os.path.join(project_dir, 'outputs', 'images')
    ls = []
    res = index.apply(lambda row: check_imgs(ls, image_dir, row[dset.channels]), axis=1)
    return ls
