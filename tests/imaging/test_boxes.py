import numpy as np
import pytest
import pandas as pd
import os
import shutil
import json
import random
import skimage.io

import deepprofiler.imaging.boxes
import deepprofiler.dataset.image_dataset
import deepprofiler.dataset.metadata

def test_getLocations():
    test_image_key = "dog/cat"
    open_file = open("deepprofiler/examples/config/test_boxes.json")
    config = json.load(open_file)
    test_output = deepprofiler.imaging.boxes.getLocations(test_image_key, config)
    expected_output = pd.DataFrame(columns=["Nuclei_Location_Center_X", "Nuclei_Location_Center_Y"])
    assert test_output.equals(expected_output)
    
    test_locations_path = "deepprofiler/examples/compressed/dog/locations/"
    test_file_name = "cat-Nuclei.csv"
    if os.path.exists(test_locations_path) == False:
        os.makedirs(test_locations_path)
    expected_output = pd.DataFrame(columns=["Dog", "Cat"])
    expected_output.to_csv(test_locations_path+test_file_name)
    expected_output=pd.read_csv(test_locations_path+test_file_name)
    assert os.path.exists(test_locations_path+test_file_name) == True  
    test_output = deepprofiler.imaging.boxes.getLocations(test_image_key, config)
    assert test_output.equals(expected_output)
    
    expected_output = pd.DataFrame(index=range(45),columns=["Nuclei_Location_Center_X", "Nuclei_Location_Center_Y"])
    expected_output.to_csv(test_locations_path+test_file_name,mode='w')
    expected_output=pd.read_csv(test_locations_path+test_file_name)
    test_output = deepprofiler.imaging.boxes.getLocations(test_image_key, config)
    assert test_output.equals(expected_output)
    
    expected_output = pd.DataFrame(index=range(60),columns=["Nuclei_Location_Center_X", "Nuclei_Location_Center_Y"])
    expected_output.to_csv(test_locations_path+test_file_name,mode='w')
    expected_output = pd.read_csv(test_locations_path+test_file_name)
    expected_output = expected_output.sample(n=50,random_state=1414)
    test_output = deepprofiler.imaging.boxes.getLocations(test_image_key, config,randomize=True,seed=1414)
    assert test_output.equals(expected_output)
    shutil.rmtree("deepprofiler/examples/compressed/dog/")
    
def __rand_array():
    return np.array(random.sample(range(100), 12))

@pytest.fixture(scope='function')
def out_dir(tmpdir):
    return os.path.abspath(tmpdir.mkdir("test"))

@pytest.fixture(scope='function')
def metadata(out_dir):
    filename = os.path.join(out_dir, 'metadata.csv')
    df = pd.DataFrame({
        'Metadata_Plate': __rand_array(),
        'Metadata_Well': __rand_array(),
        'Metadata_Site': __rand_array(),
        'R': [str(x) + '.png' for x in __rand_array()],
        'G': [str(x) + '.png' for x in __rand_array()],
        'B': [str(x) + '.png' for x in __rand_array()],
        'Sampling': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        'Split': [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    }, dtype=int)
    df.to_csv(filename, index=False)
    meta = deepprofiler.dataset.metadata.Metadata(filename)
    train_rule = lambda data: data['Split'].astype(int) == 0
    val_rule = lambda data: data['Split'].astype(int) == 1
    meta.splitMetadata(train_rule, val_rule)
    return meta

@pytest.fixture(scope='function')
def dataset(metadata, out_dir):
    keygen = lambda r: "{}/{}-{}".format(r["Metadata_Plate"], r["Metadata_Well"], r["Metadata_Site"])
    return deepprofiler.dataset.image_dataset.ImageDataset(metadata, 'Sampling', ['R', 'G', 'B'], out_dir, keygen)

@pytest.fixture(scope='function')
def loadbatch(dataset, metadata, out_dir):
    images = np.random.randint(0, 256, (128, 128, 36), dtype=np.uint8)
    for i in range(0, 36, 3):
        skimage.io.imsave(os.path.join(out_dir, dataset.meta.data['R'][i // 3]), images[:, :, i])
        skimage.io.imsave(os.path.join(out_dir, dataset.meta.data['G'][i // 3]), images[:, :, i + 1])
        skimage.io.imsave(os.path.join(out_dir, dataset.meta.data['B'][i // 3]), images[:, :, i + 2])
    open_file = open("deepprofiler/examples/config/test_boxes.json")
    config = json.load(open_file)
    result = deepprofiler.imaging.boxes.loadBatch(dataset, config)
    return result
    
def test_loadBatch(loadbatch):
    test_batch = loadbatch
    expected_batch_locations = 20*[pd.DataFrame(columns=["Nuclei_Location_Center_X", "Nuclei_Location_Center_Y"])]
    for i in range(20):
       assert test_batch["locations"][i].equals(expected_batch_locations[i])
    
def test_prepareBoxes():
    open_file = open("deepprofiler/examples/config/test_boxes.json")
    config = json.load(open_file)
    test_batch = {"images": [np.random.randint(256, size=(64, 64), dtype=np.uint16)], "targets": [[1]], "locations": [pd.DataFrame(data=[[32,32]],columns=["Nuclei_Location_Center_X", "Nuclei_Location_Center_Y"])]}
    test_result = deepprofiler.imaging.boxes.prepareBoxes(test_batch,config)
    assert np.array(test_result[0]).shape == (1,4)
    assert np.array(test_result[1]).shape == (1,)
    assert np.array(test_result[2]).shape == (1,1)
    #ignores masks for testing