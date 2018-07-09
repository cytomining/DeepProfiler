import deepprofiler.imaging.cropset
import pytest
import numpy as np
import pandas as pd
from pandas.util.testing import assert_frame_equal
import random

@pytest.fixture(scope="function")
def cropset():
    test_set_size = 500
    test_table_size = 500
    test_crop_shape = [(16,16,3),()]
    test_target_size = 500
    return deepprofiler.imaging.cropset.CropSet(test_set_size,
                                                     test_table_size,
                                                     test_crop_shape,
                                                     test_target_size)

def test_init_cropset(cropset):
    test_set_size = 500
    test_table_size = 500
    test_target_size = 500
    assert cropset.set_size == test_set_size
    assert cropset.table_size == test_table_size
    assert cropset.target_size == test_target_size
    np.testing.assert_array_equal(cropset.crops,np.zeros( (test_table_size, 16, 16, 3) ))
    assert_frame_equal(cropset.labels,pd.DataFrame(data=np.zeros((test_table_size), dtype=np.int32),  columns=["target"]))
    assert cropset.pointer == 0
    assert cropset.ready == False

def test_add_crops(cropset):
    test_table_size = 500
    test_crops = []
    test_load = 50
    np.random.seed(50)
    for num in range(test_load):
        test_crops.append(np.random.randint(256, size=(16, 16, 3), dtype=np.uint16))
    test_crops = np.array(test_crops)
    test_labels = np.ones( (test_load,2) )
    test_labels[:,1] = np.array(test_load*[2])
    cropset.add_crops(test_crops, test_labels)
    assert cropset.crops.shape == (test_table_size,16,16,3)
    assert cropset.labels.shape == (test_table_size,1)
    np.testing.assert_array_equal(cropset.crops,np.concatenate((test_crops,np.zeros( (test_table_size-test_load, 16, 16, 3) ))))
    assert_frame_equal(cropset.labels,pd.DataFrame(data=np.concatenate((np.ones((test_load), dtype=np.int64),np.zeros((test_table_size-test_load), dtype=np.int64))),columns=["target"]))
    assert cropset.pointer == test_load
    test_load = 500
    test_crops = []
    np.random.seed(500)
    for num in range(test_load):
        test_crops.append(np.random.randint(256, size=(16, 16, 3), dtype=np.uint16))
    test_crops = np.array(test_crops)
    test_labels = np.ones( (test_load,3) )
    test_labels[:,2] = np.array(test_load*[4])
    cropset.add_crops(test_crops, test_labels)
    assert cropset.pointer == test_load - (cropset.table_size - 50) #works because test_load > cropset.table_size by 50
    assert cropset.ready == True
    np.testing.assert_array_equal(cropset.crops,np.concatenate((test_crops[450:500],test_crops[0:450] )))
    assert_frame_equal(cropset.labels,pd.DataFrame(data=np.array((test_load*[2]), dtype=np.int64),columns=["target"]))


def test_batch_cropset(cropset):
    test_load = 500
    test_crops = []
    np.random.seed(600)
    for num in range(test_load):
        test_crops.append(np.random.randint(256, size=(16, 16, 3), dtype=np.uint16))
    test_crops = np.array(test_crops)
    test_labels = np.ones( (test_load,3) )
    test_labels[:,2] = np.array(test_load*[4])
    cropset.add_crops(test_crops, test_labels)
    test_batch_size = 50
    cropset.set_size = 450
    assert cropset.batch(test_batch_size)[0].shape == (test_batch_size, cropset.set_size, cropset.crops.shape[1], cropset.crops.shape[2], cropset.crops.shape[3])
    assert cropset.batch(test_batch_size)[1].shape == (test_batch_size, cropset.target_size)
    np.testing.assert_array_equal(cropset.labels["target"].unique(),np.array(([2]), dtype=np.int64))
    expected_data = np.zeros( (test_batch_size, cropset.set_size, cropset.crops.shape[1], cropset.crops.shape[2], cropset.crops.shape[3]) )    
    for i in range(test_batch_size):
        test_sample = cropset.labels[cropset.labels["target"] == 2]
        test_sample = test_sample.sample(n=cropset.set_size, replace=False, random_state=26)
        test_index = test_sample.index.tolist()
        expected_data[i,:,:,:,:] = cropset.crops[test_index, ...]
    np.testing.assert_array_equal(cropset.batch(test_batch_size,seed=26)[0],expected_data)
    expected_labels = np.zeros((test_batch_size, cropset.target_size))
    expected_labels[:,2] = 1.0
    np.testing.assert_array_equal(cropset.batch(test_batch_size,seed=26)[1],expected_labels)
    cropset.set_size = 550
    expected_data = np.zeros( (test_batch_size, cropset.set_size, cropset.crops.shape[1], cropset.crops.shape[2], cropset.crops.shape[3]) )    
    for i in range(test_batch_size):
        test_sample = cropset.labels[cropset.labels["target"] == 2]
        test_sample = test_sample.sample(n=cropset.set_size, replace=True, random_state=26)
        test_index = test_sample.index.tolist()
        expected_data[i,:,:,:,:] = cropset.crops[test_index, ...]
    np.testing.assert_array_equal(cropset.batch(test_batch_size,seed=26)[0],expected_data)
    
@pytest.fixture(scope="function")
def mixup(cropset):
    test_alpha = 1
    test_table_size = 500
    test_crop_shape = [(16,16,3),()]
    test_target_size = 500
    return deepprofiler.imaging.cropset.Mixup(test_alpha,
                                                     test_table_size,
                                                     test_crop_shape,
                                                     test_target_size)
def test_init_mixup(mixup):
    test_set_size = 2
    test_table_size = 500
    test_target_size = 500
    test_alpha = 1
    assert mixup.set_size == test_set_size
    assert mixup.table_size == test_table_size
    assert mixup.target_size == test_target_size
    np.testing.assert_array_equal(mixup.crops,np.zeros( (test_table_size, 16, 16, 3) ))
    assert_frame_equal(mixup.labels,pd.DataFrame(data=np.zeros((test_table_size), dtype=np.int32),  columns=["target"]))
    assert mixup.pointer == 0
    assert mixup.ready == False
    assert mixup.alpha == test_alpha

def test_batch_mixup(mixup):
    test_load = 500
    test_crops = []
    mixup.alpha = 0.3
    test_seed = 600
    np.random.seed(test_seed)
    for num in range(test_load):
        test_crops.append(np.random.randint(256, size=(16, 16, 3), dtype=np.uint16))
    test_crops = np.array(test_crops)
    test_labels = np.ones( (test_load,3) )
    test_labels[:,2] = np.array(test_load*[4])
    mixup.add_crops(test_crops, test_labels)
    test_batch_size = 50
    
    assert mixup.batch(test_batch_size)[0].shape == (test_batch_size, mixup.crops.shape[1], mixup.crops.shape[2], mixup.crops.shape[3])
    assert mixup.batch(test_batch_size)[1].shape == (test_batch_size, mixup.target_size)
    np.testing.assert_array_equal(mixup.labels["target"].unique(),np.array(([2]), dtype=np.int64))
    
    expected_data = np.zeros( (test_batch_size, mixup.crops.shape[1], mixup.crops.shape[2], mixup.crops.shape[3]) ) 
    np.random.seed(test_seed)
    for i in range(test_batch_size):
        test_lam = np.random.beta(mixup.alpha, mixup.alpha)
        test_sample = mixup.labels.sample(n=2,random_state=test_seed)
        test_idx = test_sample.index.tolist()
        expected_data[i,:,:,:] = test_lam*mixup.crops[test_idx[0],...] + (1. - test_lam)*mixup.crops[test_idx[1],...]
    np.testing.assert_array_equal(mixup.batch(test_batch_size,seed=test_seed)[0],expected_data)
    
    expected_labels = np.zeros((test_batch_size, mixup.target_size))
    expected_labels[:,2] = 1.0
    np.testing.assert_array_equal(mixup.batch(test_batch_size,seed=test_seed)[1],expected_labels)


    