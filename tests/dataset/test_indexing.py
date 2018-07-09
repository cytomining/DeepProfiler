import deepprofiler.dataset.indexing
import pytest
import pandas as pd
import json
import os
import tempfile
import shutil


def test_relative_paths():
    test_input = pd.DataFrame(data={'path': ['/Users/pytest/Documents/Plate1/','/Users/pytest/Documents/Plate2/','/Users/pytest/Documents/Plate2/'],'filename': ['test1.tiff','test2.tiff','test3.tiff']})
    test_output = deepprofiler.dataset.indexing.relative_paths(test_input,'target','path','filename','/Users/pytest/Documents')
    expected_output = pd.DataFrame(data={'target': ['/Plate1/test1.tiff','/Plate2/test2.tiff','/Plate2/test3.tiff']})
    assert test_output.shape == (3,1)
    assert test_output.equals(expected_output)
    
#def test_create_metadata_index():
    #missing examples to properly test/too large
    
def test_write_compression_index():
    temp = os.path.dirname("tests/files/metadata/tmp/")
    if os.path.exists(temp) == False:
        os.makedirs(temp)
    open_file = open("tests/files/config/test_config.json")
    config = json.load(open_file)
    deepprofiler.dataset.indexing.write_compression_index(config)
    test_output = pd.DataFrame.from_csv("tests/files/metadata/tmp/index.csv")
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
    test_outputs = [pd.DataFrame.from_csv("tests/files/metadata/tmp/index-000.csv"),
                    pd.DataFrame.from_csv("tests/files/metadata/tmp/index-001.csv"),
                    pd.DataFrame.from_csv("tests/files/metadata/tmp/index-002.csv")]
    assert test_outputs[0].shape == (12,9)
    assert test_outputs[1].shape == (12,9)
    assert test_outputs[2].shape == (12,9)
    assert test_outputs[0].values[5][5] == 'Week1_22123/pngs/Week1_150607_B03_s2_w25CEC2D43-E105-42BB-BC00-6962B3ADEBED.png'
    assert test_outputs[1].values[11][6] == 'Week1_22123/pngs/Week1_150607_B04_s2_w4342F300D-60F8-4256-A637-F1367E14BE5E.png'
    assert test_outputs[2].values[0][4] == 'Week1_22141/pngs/Week1_150607_B03_s2_w1A7BCCCBB-5B8B-45B2-858A-A57A37EE0D58.png'
    shutil.rmtree(temp)   