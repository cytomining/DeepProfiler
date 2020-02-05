import numpy as np
import pandas as pd
import os

import deepprofiler.imaging.boxes
import deepprofiler.dataset.image_dataset
import deepprofiler.dataset.metadata


def test_get_locations(config, make_struct):
    test_image_key = "dog/cat"
    test_output = deepprofiler.imaging.boxes.get_locations(test_image_key, config)
    expected_output = pd.DataFrame(columns=["Nuclei_Location_Center_X", "Nuclei_Location_Center_Y"])
    assert test_output.equals(expected_output)
    
    test_locations_path = os.path.abspath(os.path.join(config["paths"]["locations"], "dog"))
    os.makedirs(test_locations_path)
    test_file_name = "cat-Nuclei.csv"
    test_locations_path = os.path.join(test_locations_path, test_file_name)
    expected_output = pd.DataFrame(columns=["Nuclei_Location_Center_X", "Nuclei_Location_Center_Y"])
    expected_output.to_csv(test_locations_path)
    expected_output = pd.read_csv(test_locations_path)
    assert os.path.exists(test_locations_path) == True
    test_output = deepprofiler.imaging.boxes.get_locations(test_image_key, config)
    assert test_output.equals(expected_output)
    
    expected_output = pd.DataFrame(index=range(10), columns=["Nuclei_Location_Center_X", "Nuclei_Location_Center_Y"])
    expected_output.to_csv(test_locations_path, mode="w")
    expected_output = pd.read_csv(test_locations_path)
    test_output = deepprofiler.imaging.boxes.get_locations(test_image_key, config)
    assert test_output.equals(expected_output)
    
    expected_output = pd.DataFrame(index=range(60), columns=["Nuclei_Location_Center_X", "Nuclei_Location_Center_Y"])
    expected_output.to_csv(test_locations_path, mode="w")
    expected_output = pd.read_csv(test_locations_path)
    expected_output = expected_output.sample(n=10, random_state=1414)
    test_output = deepprofiler.imaging.boxes.get_locations(test_image_key, config, random_sample=10, seed=1414)
    assert test_output.equals(expected_output)


def test_prepare_boxes(config):
    test_batch = {"images": [np.random.randint(256, size=(64, 64), dtype=np.uint16)], "targets": [[1]], "locations": [pd.DataFrame(data=[[32,32]],columns=["Nuclei_Location_Center_X", "Nuclei_Location_Center_Y"])]}
    test_result = deepprofiler.imaging.boxes.prepare_boxes(test_batch, config)
    assert np.array(test_result[0]).shape == (1, 4)
    assert np.array(test_result[1]).shape == (1,)
    assert np.array(test_result[2]).shape == (1, 1)
    #ignores masks for testing