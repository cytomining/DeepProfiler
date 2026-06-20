"""Command-line interface for DeepProfiler.

Four subcommands are available, intended to be run in order:

1. ``setup``   — create the project directory structure under ``--root``.
2. ``prepare`` — compute per-plate illumination statistics and compress images
                 to 8-bit PNG (optional but recommended for large datasets).
3. ``profile`` — extract per-cell deep learning features using the Cell
                 Painting CNN v1 checkpoint and write ``.npz`` files.
4. ``split``   — split the metadata index into N parts for parallel profiling
                 across multiple machines or jobs.

Typical usage::

    deepprofiler --root=/data/project --config=config.json --exp=run1 profile

See README.md and the DeepProfiler Handbook for full configuration details.
"""

import copy
import json
import os

import click

import deepprofiler.dataset.compression
import deepprofiler.dataset.illumination_statistics
import deepprofiler.dataset.image_dataset
import deepprofiler.dataset.indexing
import deepprofiler.dataset.metadata
import deepprofiler.dataset.utils
import deepprofiler.profiling


# Main interaction point
@click.group()
@click.option("--root", prompt="Root directory for DeepProfiler experiment",
              help="Root directory for DeepProfiler experiment",
              type=click.Path(exists=True))
@click.option("--config", default=None,
              help="Path to existing config file (filename in project_root/inputs/config/)",
              type=click.STRING)
@click.option("--cores", default=0,
              help="Number of CPU cores for parallel processing (all=0) for prepare command",
              type=click.INT)
@click.option("--gpu", default="0",
              help="GPU device id (the id can be checked with nvidia-smi)",
              type=click.STRING)
@click.option("--exp", default="results",
              help="Name of experiment, this folder will be created in project_root/outputs/",
              type=click.STRING)
@click.option("--metadata", default='index.csv',
              help="Metadata index filename in project_root/inputs/metadata/",
              type=click.STRING)
@click.pass_context
def cli(context, root, config, exp, cores, gpu, metadata):
    """Configure paths and load the experiment config, then dispatch to a subcommand."""
    dirs = {
        "root": root,
        "locations": root + "/inputs/locations/",  # TODO: use os.path.join()
        "config": root + "/inputs/config/",
        "images": root + "/inputs/images/",
        "metadata": root + "/inputs/metadata/",
        "intensities": root + "/outputs/intensities/",
        "compressed_images": root + "/outputs/compressed/images/",
        "results": root + "/outputs/" + exp + "/",
        "checkpoints": root + "/outputs/" + exp + "/checkpoint/",
        "logs": root + "/outputs/" + exp + "/logs/",
        "summaries": root + "/outputs/" + exp + "/summaries/",
        "features": root + "/outputs/" + exp + "/features/"
    }
    if context.invoked_subcommand == 'setup':
        context.obj["dirs"] = dirs
        return 

    config = dirs["config"] + "/" + config
    context.obj["cores"] = cores
    context.obj["gpu"] = gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    # Load configuration file
    if config is not None and os.path.isfile(config):
        with open(config, "r") as f:
            params = json.load(f)

        # Override paths defined by user
        if "paths" in params.keys():
            for key, value in dirs.items():
                if key not in params["paths"].keys():
                    params["paths"][key] = dirs[key]
                else:
                    dirs[key] = params["paths"][key]
        else:
            params["paths"] = copy.deepcopy(dirs)

        if os.path.isdir(dirs["root"]):
            for k in ["results", "checkpoints", "logs", "summaries", "features"]:
                os.makedirs(dirs[k], exist_ok=True)

        # Update references
        params["experiment_name"] = exp
        params["paths"]["index"] = params["paths"]["metadata"] + metadata
        context.obj["config"] = params
    else:
        raise Exception("Config does not exists; make sure that the file exists in /inputs/config/")

    context.obj["dirs"] = dirs


# Optional tool: Create the support file and folder structure in a root directory
@cli.command(help='initialize folder structure of DeepProfiler project')
@click.pass_context
def setup(context):
    """Create the project directory tree under the configured root."""
    for path in context.obj["dirs"].values():
        if not os.path.isdir(path):
            print("Creating directory: ", path)
            os.makedirs(path)
        else:
            print("Directory exists: ", path)
    context.obj["config"] = {}
    context.obj["config"]["paths"] = context.obj["dirs"]


# First tool: Compute illumination statistics and compress images
@cli.command(help='Run illumination correction and compression')
@click.pass_context
def prepare(context):
    """Compute per-plate illumination statistics and compress images to 8-bit PNG."""
    metadata = deepprofiler.dataset.metadata.read_plates(context.obj["config"]["paths"]["index"])
    process = deepprofiler.dataset.utils.Parallel(context.obj["config"], numProcs=context.obj["cores"])
    process.compute(deepprofiler.dataset.illumination_statistics.calculate_statistics, metadata)
    print("Illumination complete!")
    metadata = deepprofiler.dataset.metadata.read_plates(
        context.obj["config"]["paths"]["index"])  # reinitialize generator
    process.compute(deepprofiler.dataset.compression.compress_plate, metadata)
    print("Compression complete!")


# Second tool: Profile cells and extract features
@cli.command(help='run feature extraction')
@click.pass_context
@click.option("--part",
              help="Part of index to process",
              default=-1,
              type=click.INT)
def profile(context, part):
    """Extract per-cell deep learning features and write .npz files."""
    if context.parent.obj["config"]["prepare"]["compression"]["implement"]:
        context.parent.obj["config"]["paths"]["images"] = context.obj["config"]["paths"]["compressed_images"]
    config = context.obj["config"]
    if part >= 0:
        partfile = "index-{0:03d}.csv".format(part)
        config["paths"]["index"] = context.obj["config"]["paths"]["index"].replace("index.csv", partfile)
    dset = deepprofiler.dataset.image_dataset.read_dataset(context.obj["config"], mode='profile')
    deepprofiler.profiling.profile(context.obj["config"], dset)


# Auxiliary tool: Split index in multiple parts
@cli.command(help='split metadata into multiple parts')
@click.pass_context
@click.option("--parts",
              help="Number of parts to split the index",
              type=click.INT)
def split(context, parts):
    """Split the metadata index into N parts for parallel profiling jobs."""
    if context.parent.obj["config"]["prepare"]["compression"]["implement"]:
        context.parent.obj["config"]["paths"]["images"] = context.obj["config"]["paths"]["compressed_images"]
    deepprofiler.dataset.indexing.split_index(context.obj["config"], parts)


if __name__ == "__main__":
    cli(obj={})
