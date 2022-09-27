![DeepProfiler](figures/logo/banner.png)

-----------------
[![Python 3.6+](https://img.shields.io/badge/python-3.6%2B-blue)](https://www.python.org/downloads/release/python-360/)
[![Tensorflow 2.5+](https://img.shields.io/badge/tensorflow-2.5%2B-brightgreen)](https://www.tensorflow.org/install/pip)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7114558.svg)](https://doi.org/10.5281/zenodo.7114558)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/1966/badge)](https://bestpractices.coreinfrastructure.org/projects/1966)
[![CometML](https://img.shields.io/badge/comet.ml-track-brightgreen.svg)](https://www.comet.ml)


# Image-based profiling using deep learning 

DeepProfiler is a set of tools that allow you to use deep learning for analyzing imaging data in high-throughput biological experiments. 
Please, see our [DeepProfiler Handbook](https://cytomining.github.io/DeepProfiler-handbook/) for more details about how to use it.

There is [an analysis preprint](https://doi.org/10.1101/2022.08.12.503783) on bioRxiv.

[_**Cell Painting CNN**_ weights are availible on Zenodo.](https://doi.org/10.5281/zenodo.7114558)

# Quick Guide

First, clone or fork this repository with example data (example data is stored with `git-lfs`):
<pre>
git clone https://github.com/broadinstitute/DeepProfiler.git
</pre>

If you don't need an example data, you can clone without it:
<pre>
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/broadinstitute/DeepProfiler.git
</pre>



Then [install](https://cytomining.github.io/DeepProfiler-handbook/docs/01-install.html) it using:
<pre>
pip install -e .
</pre>

When running DeepProfiler you usually need to specify a root directory where your data is stored and a command that you want to run. 
For instance, to initialize your project, you can use:
<pre>
python deepprofiler --root=/home/ubuntu/project --config=config.json setup
</pre>
In the created directories, you will need to organize your input files, including metadata, images and single-cell locations. 
See more details about the [project structure here](https://cytomining.github.io/DeepProfiler-handbook/docs/02-structure.html).

Next, if you want to train a model, as specified in the [configuration file](https://cytomining.github.io/DeepProfiler-handbook/docs/03-config.html), you can then run the following command:

<pre>
python deepprofiler --root=/home/ubuntu/project/ --config filename.json train
</pre>

And to extract single-cell embeddings, use:

<pre>
python deepprofiler --root=/home/ubuntu/project/ --config filename.json profile
</pre>

Find more information in the [training and profiling](https://cytomining.github.io/DeepProfiler-handbook/docs/04-train-infer.html) section of our wiki.


**Happy profiling!**
