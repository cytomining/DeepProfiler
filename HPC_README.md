# Download Instructions

## Conda Environment
Before downloading DeepProfiler, you need to setup the correct conda environment (if not already made):

```
conda create --name dp_env python=3.8
conda activate dp_env
```

## Clone Repo
Now clone the repo where you want the folder to be stored:

```
git clone https://github.com/broadinstitute/DeepProfiler.git
```

## Install dependencies
First you'll need to download the correct version of TensorFlow. As of writing this, TensforFlow version 2.5.3 is the stable version for DeepProfiler:

```
pip install tensorflow-gpu==2.5.3
```

Once that's done, you can download DeepProfiler with:

```
cd DeepProfiler/
pip install -e .
```

## Verifying Download & Usage
DeepProfiler needs cuda version 11, so you'll need to activate this before using the package.

To first verify the downloaded DeepProfiler package, you'll need to request a GPU node. Once on the node, you can use the `module avail` command to see what versions of cuda exist. Once cuda 11.x is found, use the following command:

```
module load cuda/11.x
```

where x is the latest stable release downloaded on your GPU node.

To verify the cuda version, use:

```
nvcc --version
```

The release should say 11.x with the release you chose.

Finally, while in directory where you cloned DeepProfiler, run:

```
python3 deepprofiler
```

and the resulting output should say

```
Successfully opened dynamic library libcudart.so.11.0
Usage: deepprofiler [OPTIONS] COMMAND [ARGS]...

Options:
  --root PATH          Root directory for DeepProfiler experiment
  --config TEXT        Path to existing config file (filename in
                       project_root/inputs/config/)
  --cores INTEGER      Number of CPU cores for parallel processing (all=0) for
                       prepare command
  --gpu TEXT           GPU device id (the id can be checked with nvidia-smi)
  --exp TEXT           Name of experiment, this folder will be created in
                       project_root/outputs/
  --single-cells TEXT  Name of the folder with single-cell dataset (output for
                       export-sc command, input for training with sampled crop
                       generator or online labels crop generator)
  --metadata TEXT      data filename, for exporting or profiling it is a
                       filename for project_root/inputs/metadata/, for
                       training with sampled crop generator or online labels
                       crop generator the filename in
                       project_root/outputs/<single-cell-dataset>/
  --logging TEXT       Path to file with comet.ml API key (filename in
                       project_root/inputs/config/)
  --help               Show this message and exit.

Commands:
  export-sc  export crops of single-cells for training
  prepare    Run illumination correction and compression
  profile    run feature extraction
  setup      initialize folder structure of DeepProfiler project
  split      split metadata into multiple parts
  train      train a model
  traintf2   train a model with TensorFlow 2 dataset
```

### Note
To use DeepProfiler in the future, you will need to be on a GPU node and remember to load cuda 11.x.