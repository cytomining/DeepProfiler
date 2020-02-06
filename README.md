![DeepProfiler](figures/logo/banner.png)

-----------------

[![Build Status](https://travis-ci.com/broadinstitute/DeepProfiler.svg?branch=master)](https://travis-ci.com/broadinstitute/DeepProfiler)
[![codecov](https://codecov.io/gh/broadinstitute/DeepProfiler/branch/master/graph/badge.svg)](https://codecov.io/gh/broadinstitute/DeepProfiler)
[![Requirements Status](https://requires.io/github/broadinstitute/DeepProfiler/requirements.svg?branch=master)](https://requires.io/github/broadinstitute/DeepProfiler/requirements/?branch=master)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/1966/badge)](https://bestpractices.coreinfrastructure.org/projects/1966)
[![CometML](https://img.shields.io/badge/comet.ml-track-brightgreen.svg)](https://www.comet.ml)

# Image-based profiling using deep learning 

DeepProfiler is a set of tools that allow you to use deep learning for analyzing imaging data in high-throughput biological experiments. Please, see our [Wiki documentation page](https://github.com/broadinstitute/DeepProfiler/wiki) for more details about how to use it.

# Quick Guide 

Clone or fork this repository and [install](https://github.com/broadinstitute/DeepProfiler/wiki/1.-Installing-DeepProfiler) it using:
<pre>
pip install -e .
</pre>

When running DeepProfiler you usually need to specify a root directory where your data is stored and a command that you want to run. 
For instance, to initialize your project, you can use:
<pre>
python deepprofiler --root=/home/ubuntu/project --config=config.json --gpu 0 setup
</pre>
In the created directories, you will need to organize your input files, including metadata, images and single-cell locations. 
See more details about the [project structure here](https://github.com/broadinstitute/DeepProfiler/wiki/2.-Project-structure).

Next, if you want to train a model, as specified in the [configuration file](https://github.com/broadinstitute/DeepProfiler/wiki/3.-The-configuration-file), you can then run the following command:

<pre>
python deepprofiler --root=/home/ubuntu/project/ --config filename.json --gpu 0 train
</pre>

And to extract single-cell embeddings, use:

<pre>
python deepprofiler --root=/home/ubuntu/project/ --config filename.json --gpu 0 profile
</pre>

Find more information in the [training and profiling](https://github.com/broadinstitute/DeepProfiler/wiki/4.-Training-and-Profiling) section of our wiki.


**Happy profiling!**
