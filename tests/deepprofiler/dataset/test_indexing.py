import deepprofiler.dataset.indexing
import pytest
import pandas as pd
import json
import os
import shutil


def test_write_compression_index():
    temp = os.path.dirname("tests/files/metadata/tmp/")
    if os.path.exists(temp) == False:
        os.makedirs(temp)
    open_file = open("tests/files/config/test_config.json")
    config = json.load(open_file)
    deepprofiler.dataset.indexing.write_compression_index(config)
    test_output = pd.read_csv("tests/files/metadata/tmp/index.csv", index_col=0)
    assert test_output.shape == (36,9)
    assert test_output.values[31][5] == 'Week1_22123/pngs/Week1_150607_B03_s2_w25CEC2D43-E105-42BB-BC00-6962B3ADEBED.png'   
    shutil.rmtree(temp)    

def test_split_index():
    temp = os.path.dirname("tests/files/metadata/tmp/")
    if os.path.exists(temp) == False:
        os.makedirs(temp)
    open_file = open("tests/files/config/test_config.json")
    config = json.load(open_file)
    test_parts = 3
    test_paths = ["tests/files/metadata/tmp/index-000.csv",
                  "tests/files/metadata/tmp/index-001.csv",
                  "tests/files/metadata/tmp/index-002.csv"]
    deepprofiler.dataset.indexing.write_compression_index(config)
    deepprofiler.dataset.indexing.split_index(config, test_parts)   
    assert os.path.exists(test_paths[0]) == True
    assert os.path.exists(test_paths[1]) == True
    assert os.path.exists(test_paths[2]) == True
    test_outputs = [pd.read_csv("tests/files/metadata/tmp/index-000.csv", index_col=0),
                    pd.read_csv("tests/files/metadata/tmp/index-001.csv", index_col=0),
                    pd.read_csv("tests/files/metadata/tmp/index-002.csv", index_col=0)]
    assert test_outputs[0].shape == (12,9)
    assert test_outputs[1].shape == (12,9)
    assert test_outputs[2].shape == (12,9)
    assert test_outputs[0].values[5][5] == 'Week1_22123/pngs/Week1_150607_B03_s2_w25CEC2D43-E105-42BB-BC00-6962B3ADEBED.png'
    assert test_outputs[1].values[11][6] == 'Week1_22123/pngs/Week1_150607_B04_s2_w4342F300D-60F8-4256-A637-F1367E14BE5E.png'
    assert test_outputs[2].values[0][4] == 'Week1_22141/pngs/Week1_150607_B03_s2_w1A7BCCCBB-5B8B-45B2-858A-A57A37EE0D58.png'
    shutil.rmtree(temp)   