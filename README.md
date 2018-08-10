![DeepProfiler](images/logo/banner.png)

-----------------

[![Build Status](https://travis-ci.org/broadinstitute/DeepProfiler.svg?branch=master)](https://travis-ci.org/broadinstitute/DeepProfiler)
[![codecov](https://codecov.io/gh/broadinstitute/DeepProfiler/branch/master/graph/badge.svg)](https://codecov.io/gh/broadinstitute/DeepProfiler)
[![Requirements Status](https://requires.io/github/broadinstitute/DeepProfiler/requirements.svg?branch=master)](https://requires.io/github/broadinstitute/DeepProfiler/requirements/?branch=master)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/1966/badge)](https://bestpractices.coreinfrastructure.org/projects/1966)
[![CometML](https://img.shields.io/badge/comet.ml-track-brightgreen.svg)](https://www.comet.ml)

# DeepProfiler
Morphological profiling using deep learning 

## Contents

This project provides tools and APIs to manipulate high-throughput images for deep learning. The dataset tools are the only ones currently implemented. This project only supports Python 3.

## Tools

All of the following commands require the --root flag to be set to the root project directory. They also require a configuration file to be present (excluding the setup command). The commands use the following syntax:

<pre>
    python deepprofiler --root=[project root] [command] [command flags]
</pre>

Additionally, the --config flag can be used to manually specify a configuration file not in the config directory. See the project Wiki for documentation on configuration files.

### Setting up the project directory

The project directory can be set up automatically from a specified root directory:

<pre>
    python deepprofiler --root=[project root] setup
</pre>

A configuration file is not necessary for this step, but you can specify one with the --config flag after deepprofiler if you want to use existing directories.

### Preparing the dataset

The dataset can be optionally preprocessed with illumination correction and compression, as specified in the configuration file, with one command:

<pre>
    python deepprofiler --root=[project root] prepare
</pre>

### Training the model

To train your model on the dataset:

<pre>
    python deepprofiler --root=[project root] train
</pre>

You may optionally specify the --epoch and --seed flags after train, to set the current epoch or the random seed.

### Extracting features

To extract single-cell features for profiling:

<pre>
    python deepprofiler --root=[project root] profile
</pre>

You will need to specify the name of the model checkpoint to use in the configuration file.

### Hyperparameter optimization

Optionally, you may run DeepProfiler's hyperparameter optimization feature (which uses GPyOpt) to find optimal values for your model's hyperparameters:

<pre>
    python deepprofiler --root=[project root] optimize
</pre>

This may take a while as it will train a model for the number of epochs specified for each step in the process. It is recommended to decrease the epochs and steps to manageable values when running this command.
