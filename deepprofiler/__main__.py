import json
import os

import click

import deepprofiler.dataset.compression
import deepprofiler.dataset.image_dataset
import deepprofiler.dataset.indexing
import deepprofiler.dataset.illumination_statistics
import deepprofiler.dataset.metadata
import deepprofiler.dataset.utils
import deepprofiler.dataset.image_dataset
import deepprofiler.learning.training
import deepprofiler.learning.profiling
import deepprofiler.learning.optimization
import deepprofiler.download.normalize_bbbc021_metadata


# Main interaction point
@click.group()
@click.option("--root", prompt="Root directory for DeepProfiler experiment",
              help="Root directory for DeepProfiler experiment",
              type=click.Path("r"))
@click.option("--config", default=None,
              help="Path to existing config file",
              type=click.Path("r"))
@click.option("--cores", default=0,
              help="Number of CPU cores for parallel processing (all=0)",
              type=click.INT)
@click.pass_context
def cli(context, root, config, cores):
    dirs = {
        "root": root,
        "locations": root+"/inputs/locations/",  # TODO: use os.path.join()
        "config": root+"/inputs/config/",
        "images": root+"/inputs/images/",
        "metadata": root+"/inputs/metadata/",
        "preprocessed": root+"/inputs/preprocessed/",
        "pretrained": root+"/inputs/pretrained/",
        "intensities": root+"/outputs/intensities/",
        "compressed_images": root+"/outputs/compressed/images/",
        "compressed_metadata": root+"/outputs/compressed/metadata/",
        "training": root+"/outputs/training/",
        "checkpoints": root+"/outputs/training/checkpoints/",
        "logs": root+"/outputs/training/logs/",
        "summaries": root+"/outputs/training/summaries/",
        "features": root+"/outputs/features/"
    }
    if config is not None:
        context.obj["config"] = {}
        context.obj["config"]["paths"] = {}
        context.obj["config"]["paths"]["config"] = config
        dirs["config"] = os.path.dirname(os.path.abspath(config))
    else:
        config = dirs["config"] + "/config.json"
    context.obj["cores"] = cores
    if os.path.isfile(config):
        with open(config, "r") as f:
            params = json.load(f)
        if "paths" in params.keys():
            for key, value in dirs.items():
                if key not in params["paths"].keys():
                    params["paths"][key] = dirs[key]
                else:
                    dirs[key] = params["paths"][key]
        else:
            params["paths"] = dirs
        params["paths"]["index"] = params["paths"]["metadata"] + "/index.csv"
        context.obj["config"] = params
        process = deepprofiler.dataset.utils.Parallel(context.obj["config"], numProcs=context.obj["cores"])
        context.obj["process"] = process
    context.obj["dirs"] = dirs


# Optional tool: Create the support file and folder structure in a root directory
@cli.command()
@click.pass_context
def setup(context):
    for path in context.obj["dirs"].values():
        if not os.path.isdir(path):
            print("Creating directory: ", path)
            os.makedirs(path)
        else:
            print("Directory exists: ", path)
    context.obj["config"] = {}
    context.obj["config"]["paths"] = context.obj["dirs"]


# Optional tool: Download and prepare the BBBC021 dataset
@cli.command()
@click.pass_context
def download_bbbc021(context):
    context.invoke(setup)
    deepprofiler.download.normalize_bbbc021_metadata.normalize_bbbc021_metadata(context)
    print("BBBC021 download and preparation complete!")


# First tool: Compute illumination statistics and compress images
@cli.command()
@click.pass_context
def prepare(context):
    metadata = deepprofiler.dataset.metadata.read_plates(context.obj["config"]["paths"]["index"])
    process = context.obj["process"]
    process.compute(deepprofiler.dataset.illumination_statistics.calculate_statistics, metadata)
    print("Illumination complete!")
    metadata = deepprofiler.dataset.metadata.read_plates(context.obj["config"]["paths"]["index"])  # reinitialize generator
    process.compute(deepprofiler.dataset.compression.compress_plate, metadata)
    deepprofiler.dataset.indexing.write_compression_index(context.obj["config"])
    context.parent.obj["config"]["paths"]["index"] = context.obj["config"]["paths"]["compressed_metadata"]+"/compressed.csv"
    print("Compression complete!")


# Optional learning tool: Optimize the hyperparameters of a model
@cli.command()
@click.option("--epoch", default=1)
@click.option("--seed", default=None)
@click.pass_context
def optimize(context, epoch, seed):
    if context.parent.obj["config"]["prepare"]["compression"]["implement"]:
        context.parent.obj["config"]["paths"]["index"] = context.obj["config"]["paths"]["compressed_metadata"]+"/compressed.csv"
        context.parent.obj["config"]["paths"]["images"] = context.obj["config"]["paths"]["compressed_images"]
    metadata = deepprofiler.dataset.image_dataset.read_dataset(context.obj["config"])
    optim = deepprofiler.learning.optimization.Optimize(context.obj["config"], metadata, epoch, seed)
    optim.optimize()


# Second tool: Train a network
@cli.command()
@click.option("--epoch", default=1)
@click.option("--seed", default=None)
@click.pass_context
def train(context, epoch, seed):
    if context.parent.obj["config"]["prepare"]["compression"]["implement"]:
        context.parent.obj["config"]["paths"]["index"] = context.obj["config"]["paths"]["compressed_metadata"]+"/compressed.csv"
        context.parent.obj["config"]["paths"]["images"] = context.obj["config"]["paths"]["compressed_images"]
    metadata = deepprofiler.dataset.image_dataset.read_dataset(context.obj["config"])
    deepprofiler.learning.training.learn_model(context.obj["config"], metadata, epoch, seed)


# Third tool: Profile cells and extract features
@cli.command()
@click.pass_context
@click.option("--part",
              help="Part of index to process", 
              default=-1, 
              type=click.INT)
def profile(context, part):
    if context.parent.obj["config"]["prepare"]["compression"]["implement"]:
        context.parent.obj["config"]["paths"]["index"] = context.obj["config"]["paths"]["compressed_metadata"]+"/compressed.csv"
        context.parent.obj["config"]["paths"]["images"] = context.obj["config"]["paths"]["compressed_images"]
    config = context.obj["config"]
    if part >= 0:
        partfile = "index-{0:03d}.csv".format(part)
        config["paths"]["index"] = context.obj["config"]["paths"]["index"].replace("index.csv", partfile)
    metadata = deepprofiler.dataset.image_dataset.read_dataset(context.obj["config"])
    deepprofiler.learning.profiling.profile(context.obj["config"], metadata)
    

# Auxiliary tool: Split index in multiple parts
@cli.command()
@click.pass_context
@click.option("--parts", 
              help="Number of parts to split the index",
              type=click.INT)
def split(context, parts):
    if context.parent.obj["config"]["prepare"]["compression"]["implement"]:
        context.parent.obj["config"]["paths"]["index"] = context.obj["config"]["paths"]["compressed_metadata"]+"/compressed.csv"
        context.parent.obj["config"]["paths"]["images"] = context.obj["config"]["paths"]["compressed_images"]
    deepprofiler.dataset.indexing.split_index(context.obj["config"], parts)


if __name__ == "__main__":
    cli(obj={})
