# DeepProfiler
Morphological profiling using deep learning 

## Contents

This projects provide tools and APIs to manipulate high-throughput images for deep learning. The dataset tools are the only ones currently implemented. 

## Dataset Tools

To prepare microscopy datasets for deep learning we have implemented the following steps that should be run sequentially: 1) Collect illumination statistics, 2) Compress images, and 3) Create cell location indices. Prior to these three steps, we need to create a metadata file with image locations and labels.

Any of these three steps requires a configuration file written in JSON format. With this file available for a particular dataset, you can run the dataset tools as follows:

<pre>
    python dataset --config=data.json illumination
    python dataset --config=data.json compression
    python dataset --config=data.json locations
</pre>

These commands take some time to get your dataset ready. After that, you can launch the learning commands [under construction].
