import sys
import pandas as pd
import tempfile
import os

import deepprofiler.dataset.helper


# Dont know how to test with CLI input?

tempdir = tempfile.TemporaryFile()

def test_check_profile():
    dset = []
    deepprofiler.dataset.helper.check_profile(dset)
    df = pd.read_csv('missing_images.csv')
    print('Missing images:')
    print(df.missing_images.tolist())
    assert len(df) == 0


def test_check_training():
    dset = []
    deepprofiler.dataset.helper.check_training(dset)
    miss_crops = pd.read_csv('missing_crops.csv')
    print('Missing crops:')
    print(miss_crops.missing_crops.tolist())

    zero_crops = pd.read_csv('zero_crops.csv')
    print('Zero crops:')
    print(zero_crops.zero_crops.tolist())
    assert len(miss_crops) == 0
    assert len(zero_crops) == 0


