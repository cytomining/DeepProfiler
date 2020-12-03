import deepprofiler.dataset.indexing
import deepprofiler.dataset.metadata
import deepprofiler.dataset.image_dataset
import pandas as pd
import os


def test_split_index(config, metadata, dataset):
    test_parts = 3
    test_paths = [config["paths"]["metadata"]+"/index-000.csv",
                  config["paths"]["metadata"]+"/index-001.csv",
                  config["paths"]["metadata"]+"/index-002.csv"]
    deepprofiler.dataset.indexing.split_index(config, test_parts)   
    assert os.path.exists(test_paths[0]) == True
    assert os.path.exists(test_paths[1]) == True
    assert os.path.exists(test_paths[2]) == True
    test_outputs = [pd.read_csv(config["paths"]["metadata"]+"/index-000.csv"),
                    pd.read_csv(config["paths"]["metadata"]+"/index-001.csv"),
                    pd.read_csv(config["paths"]["metadata"]+"/index-002.csv")]
    assert test_outputs[0].shape == (4,8)
    assert test_outputs[1].shape == (4,8)
    assert test_outputs[2].shape == (4,8)
 