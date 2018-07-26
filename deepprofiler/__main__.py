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


def cmd_setup(context):
    with open(context.obj["paths"]["root_dir"]+"/inputs/config/config.json", 'r') as f:
        params = json.load(f)
    process = deepprofiler.dataset.utils.Parallel(params, numProcs=context.obj["cores"])
    context.parent.obj["process"] = process
    context.parent.obj["config"] = params
    context.parent.obj["config"]["paths"] = context.obj["paths"]
    context.parent.obj["config"]["paths"]["index"] = context.obj["paths"]["root_dir"]+"/inputs/metadata/index.csv"
    context.parent.obj["config"]["paths"]["config_file"] = context.obj["paths"]["root_dir"]+"/inputs/config/config.json"
    context.parent.obj["setup"] = True


# Main interaction point
@click.group()
@click.option("--root", prompt="Root directory for DeepProfiler experiment",
              help="Root directory, written in JSON format",
              type=click.Path('r'))
@click.option("--cores", default=0,
              help="Number of CPU cores for parallel processing (all=0)",
              type=click.INT)
@click.pass_context
def cli(context, root, cores):

    paths = {
        "root_dir": root,
        "locations": root+"/inputs/locations",
        "config_folder": root+"/inputs/config",
        "images": root+"/inputs/images",
        "metadata": root+"/inputs/metadata",
        "pre-processed": root+"/inputs/preprocessed",
        "pre-trained": root+"/inputs/pretrained",
        "intensities": root+"/outputs/intensities",
        "compressed_images": root+"/outputs/compressed/images",
        "compressed_metadata": root+"/outputs/compressed/metadata",
        "training": root+"/outputs/training",
        "checkpoints": root+"/outputs/training/checkpoints",
        "logs": root+"/outputs/training/logs",
        "summaries": root+"/outputs/training/summaries",
        "features": root+"/outputs/features",
    }
    context.obj["paths"] = paths
    context.obj["cores"] = cores


# Optional tool: Create the support file and folder structure in a root directory
@cli.command()
@click.pass_context
def make_struct(context):
    for path in context.obj["paths"]:
        if os.path.isdir(context.obj["paths"].get(path)+"/") == False:
            print("Creating directory: " + context.obj["paths"].get(path)+"/")
            os.makedirs(context.obj["paths"].get(path)+"/")
        else:
            print("Directory already exists: " + context.obj["paths"].get(path)+"/")
    
# First tool: Compute illumination statistics and compress images
@cli.command()
@click.pass_context
def prepare_data(context):
    if "setup" not in context.obj.keys():
        cmd_setup(context)
    process = context.obj["process"]
    metadata = deepprofiler.dataset.metadata.read_plates(context.obj["config"]["paths"]["index"])
    process.compute(deepprofiler.dataset.illumination_statistics.calculate_statistics, metadata)
    print("Illumination complete!")
    metadata = deepprofiler.dataset.metadata.read_plates(context.obj["config"]["paths"]["index"])
    process.compute(deepprofiler.dataset.compression.compress_plate, metadata)
    deepprofiler.dataset.indexing.write_compression_index(context.obj["config"])
    context.parent.obj["config"]["paths"]["index"] = context.obj["config"]["paths"]['compressed_metadata']+"/compressed.csv"
    print("Compression complete!")

# Optional learning tool: Optimize the hyperparameters of a model
@cli.command()
@click.option("--epoch", default=1)
@click.option("--seed", default=None)
@click.pass_context
def optimize(context, epoch, seed):
    if "setup" not in context.obj.keys():
        cmd_setup(context)
        if context.parent.obj["config"]["prepare"]["compression"]["implement"]:
            context.parent.obj["config"]["paths"]["index"] = context.obj["config"]["paths"]['compressed_metadata']+"/compressed.csv"
            context.parent.obj["config"]["paths"]["images"] = context.obj["config"]["paths"]['compressed_images']
    metadata = deepprofiler.dataset.image_dataset.read_dataset(context.obj["config"])
    optim = deepprofiler.learning.optimization.Optimize(context.obj["config"], metadata, epoch, seed)
    optim.optimize()

# Second tool: Train a network
@cli.command()
@click.option("--epoch", default=1)
@click.option("--seed", default=None)
@click.pass_context
def train(context, epoch, seed):
    if "setup" not in context.obj.keys():
        cmd_setup(context)
        if context.parent.obj["config"]["prepare"]["compression"]["implement"]:
            context.parent.obj["config"]["paths"]["index"] = context.obj["config"]["paths"]['compressed_metadata']+"/compressed.csv"
            context.parent.obj["config"]["paths"]["images"] = context.obj["config"]["paths"]['compressed_images']
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
    if "setup" not in context.obj.keys():
        cmd_setup(context)
        if context.parent.obj["config"]["prepare"]["compression"]["implement"]:
            context.parent.obj["config"]["paths"]["index"] = context.obj["config"]["paths"]['compressed_metadata']+"/compressed.csv"
            context.parent.obj["config"]["paths"]["images"] = context.obj["config"]["paths"]['compressed_images']
    config = context.obj["config"]
    if part >= 0:
        partfile = "index-{0:03d}.csv".format(part)
        config["image_set"]["index"] = os.path.join(context.obj["config"]["paths"]["index"], partfile)
    metadata = deepprofiler.dataset.image_dataset.read_dataset(context.obj["config"])
    deepprofiler.learning.profiling.profile(context.obj["config"], metadata)
    

# Auxiliary tool: Split index in multiple parts
@cli.command()
@click.pass_context
@click.option("--parts", 
              help="Number of parts to split the index",
              type=click.INT)
def split_index(context, parts):
    if "setup" not in context.obj.keys():
        cmd_setup(context)
        if context.parent.obj["config"]["prepare"]["compression"]["implement"]:
            context.parent.obj["config"]["paths"]["index"] = context.obj["config"]["paths"]['compressed_metadata']+"/compressed.csv"
            context.parent.obj["config"]["paths"]["images"] = context.obj["config"]["paths"]['compressed_images']
    deepprofiler.dataset.indexing.split_index(context.obj["config"], parts)

if __name__ == "__main__":
    cli(obj={})
