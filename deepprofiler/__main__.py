import comet_ml

import tensorflow as tf
import json
import os
import copy
import click

import deepprofiler.dataset.compression
import deepprofiler.dataset.image_dataset
import deepprofiler.dataset.indexing
import deepprofiler.dataset.illumination_statistics
import deepprofiler.dataset.metadata
import deepprofiler.dataset.utils
import deepprofiler.dataset.image_dataset
import deepprofiler.dataset.sampling
import deepprofiler.learning.training
import deepprofiler.learning.tf2train
import deepprofiler.learning.profiling


# Main interaction point
@click.group()
@click.option("--root", prompt="Root directory for DeepProfiler experiment",
              help="Root directory for DeepProfiler experiment",
              type=click.Path("r"))
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
@click.option("--single-cells", default="single-cells",
              help="Name of the folder with single-cell dataset (output for export-sc command, "
                   "input for training with sampled crop generator or online labels crop generator)",
              type=click.STRING)
@click.option("--metadata", default='index.csv',
              help="data filename, for exporting or profiling it is a filename for project_root/inputs/metadata/, "
                   "for training with sampled crop generator or online labels crop generator "
                   "the filename in project_root/outputs/<single-cell-dataset>/",
              type=click.STRING)
@click.option("--logging", default=None,
              help="Path to file with comet.ml API key (filename in project_root/inputs/config/)",
              type=click.STRING)
@click.pass_context
def cli(context, root, config, exp, cores, gpu, single_cells, metadata, logging):
    dirs = {
        "root": root,
        "locations": root + "/inputs/locations/",  # TODO: use os.path.join()
        "config": root + "/inputs/config/",
        "images": root + "/inputs/images/",
        "metadata": root + "/inputs/metadata/",
        "intensities": root + "/outputs/intensities/",
        "compressed_images": root + "/outputs/compressed/images/",
        "single_cell_set": root + "/outputs/" + single_cells + "/",
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
        if metadata != 'index.csv':
            params["paths"]["sc_index"] = os.path.join(params["paths"]["single_cell_set"], metadata)
        else:
            params["paths"]["sc_index"] = os.path.join(params["paths"]["single_cell_set"], 'sc-metadata.csv')
        context.obj["config"] = params
        if logging:
            with open(os.path.join(dirs["config"], logging), "r") as f:
                logging_params = json.load(f)
                if logging_params["log_type"] == "comet_ml":
                    context.obj["config"]["train"]["comet_ml"] = {}
                    context.obj["config"]["train"]["comet_ml"]["api_key"] = logging_params["api_key"]
                    context.obj["config"]["train"]["comet_ml"]["project_name"] = logging_params["project_name"]
    else:
        raise Exception("Config does not exists; make sure that the file exists in /inputs/config/")

    context.obj["dirs"] = dirs


# Optional tool: Create the support file and folder structure in a root directory
@cli.command(help='initialize folder structure of DeepProfiler project')
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


# First tool: Compute illumination statistics and compress images
@cli.command(help='Run illumination correction and compression')
@click.pass_context
def prepare(context):
    metadata = deepprofiler.dataset.metadata.read_plates(context.obj["config"]["paths"]["index"])
    process = deepprofiler.dataset.utils.Parallel(context.obj["config"], numProcs=context.obj["cores"])
    process.compute(deepprofiler.dataset.illumination_statistics.calculate_statistics, metadata)
    print("Illumination complete!")
    metadata = deepprofiler.dataset.metadata.read_plates(
        context.obj["config"]["paths"]["index"])  # reinitialize generator
    process.compute(deepprofiler.dataset.compression.compress_plate, metadata)
    print("Compression complete!")


# Second tool: Export single cells for training
@cli.command(help='export crops of single-cells for training')
@click.pass_context
def export_sc(context):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if context.parent.obj["config"]["prepare"]["compression"]["implement"]:
        context.parent.obj["config"]["paths"]["images"] = context.obj["config"]["paths"]["compressed_images"]
    dset = deepprofiler.dataset.image_dataset.read_dataset(context.obj["config"])
    deepprofiler.dataset.sampling.export_dataset(context.obj["config"], dset)
    print("Single-cell sampling complete.")


# Third tool: Train a network
@cli.command(help='train a model')
@click.option("--epoch", default=1)
@click.option("--seed", default=None)
@click.pass_context
def train(context, epoch, seed):
    if context.parent.obj["config"]["prepare"]["compression"]["implement"]:
        context.parent.obj["config"]["paths"]["images"] = context.obj["config"]["paths"]["compressed_images"]

    if context.parent.obj["config"]["train"]["model"]["crop_generator"] == 'crop_generator':
        dset = deepprofiler.dataset.image_dataset.read_dataset(context.obj["config"], mode='train')
        deepprofiler.learning.training.learn_model(context.obj["config"], dset, epoch, seed)
    else:
        deepprofiler.learning.training.learn_model(context.obj["config"], None, epoch, seed)


# Third tool (b): Train a network with TF dataset
@cli.command(help='train a model with TensorFlow 2 dataset')
@click.option("--epoch", default=1)
@click.pass_context
def traintf2(context, epoch):
    deepprofiler.learning.training.learn_model_v2(context.obj["config"], epoch)


# Fourth tool: Profile cells and extract features
@cli.command(help='run feature extraction')
@click.pass_context
@click.option("--part",
              help="Part of index to process",
              default=-1,
              type=click.INT)
def profile(context, part):
    if context.parent.obj["config"]["prepare"]["compression"]["implement"]:
        context.parent.obj["config"]["paths"]["images"] = context.obj["config"]["paths"]["compressed_images"]
    config = context.obj["config"]
    if part >= 0:
        partfile = "index-{0:03d}.csv".format(part)
        config["paths"]["index"] = context.obj["config"]["paths"]["index"].replace("index.csv", partfile)
    dset = deepprofiler.dataset.image_dataset.read_dataset(context.obj["config"], mode='profile')
    deepprofiler.learning.profiling.profile(context.obj["config"], dset)


# Auxiliary tool: Split index in multiple parts
@cli.command(help='split metadata into multiple parts')
@click.pass_context
@click.option("--parts",
              help="Number of parts to split the index",
              type=click.INT)
def split(context, parts):
    if context.parent.obj["config"]["prepare"]["compression"]["implement"]:
        context.parent.obj["config"]["paths"]["images"] = context.obj["config"]["paths"]["compressed_images"]
    deepprofiler.dataset.indexing.split_index(context.obj["config"], parts)


if __name__ == "__main__":
    cli(obj={})
