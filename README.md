![DeepProfiler](figures/logo/banner.png)

-----------------

[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/1966/badge)](https://bestpractices.coreinfrastructure.org/projects/1966)
[![CometML](https://img.shields.io/badge/comet.ml-track-brightgreen.svg)](https://www.comet.ml)

# Image-based profiling using deep learning 

DeepProfiler is a set of tools that allow you to use deep learning for analyzing imaging data in high-throughput biological experiments. 
Please, see our [DeepProfiler Handbook](https://cytomining.github.io/DeepProfiler-handbook/) for more details about how to use it.

There is [an analysis preprint](https://doi.org/10.1101/2022.08.12.503783) on bioRxiv. 

# Quick Guide 

Clone or fork this repository and [install](https://github.com/broadinstitute/DeepProfiler/wiki/1.-Installing-DeepProfiler) it using:
<pre>
pip install -e .
</pre>

When running DeepProfiler you usually need to specify a root directory where your data is stored and a command that you want to run. 
For instance, to initialize your project, you can use:
<pre>
python deepprofiler --root=/home/ubuntu/project --config=config.json setup
</pre>
In the created directories, you will need to organize your input files, including metadata, images and single-cell locations. 
See more details about the [project structure here](https://github.com/broadinstitute/DeepProfiler/wiki/2.-Project-structure).

Next, if you want to train a model, as specified in the [configuration file](https://github.com/broadinstitute/DeepProfiler/wiki/3.-The-configuration-file), you can then run the following command:

<pre>
python deepprofiler --root=/home/ubuntu/project/ --config filename.json train
</pre>

And to extract single-cell embeddings, use:

<pre>
python deepprofiler --root=/home/ubuntu/project/ --config filename.json profile
</pre>

Find more information in the [training and profiling](https://github.com/broadinstitute/DeepProfiler/wiki/4.-Training-and-Profiling) section of our wiki.


**Happy profiling!**
