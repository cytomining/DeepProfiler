import os
import deepprofiler.learning.training
import deepprofiler.dataset.image_dataset
import deepprofiler.dataset.metadata
import deepprofiler.dataset.target


def test_learn_model(config, dataset, data, locations, out_dir, make_struct):
    epoch = 1
    verbose = 1
    deepprofiler.learning.training.learn_model(config, dataset, epoch, verbose=verbose)
    assert os.path.exists(os.path.join(config["paths"]["checkpoints"], "checkpoint_0001.hdf5"))
    assert os.path.exists(os.path.join(config["paths"]["checkpoints"], "checkpoint_0002.hdf5"))
    assert os.path.exists(os.path.join(config["paths"]["logs"], "log.csv"))
    epoch = 3
    config["train"]["model"]["epochs"] = 4
    deepprofiler.learning.training.learn_model(config, dataset, epoch, verbose=verbose)
    assert os.path.exists(os.path.join(config["paths"]["checkpoints"], "checkpoint_0003.hdf5"))
    assert os.path.exists(os.path.join(config["paths"]["checkpoints"], "checkpoint_0004.hdf5"))
    assert os.path.exists(os.path.join(config["paths"]["logs"], "log.csv"))
