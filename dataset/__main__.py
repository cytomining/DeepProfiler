import json
import os

import click

import dataset.compression
import dataset.image_dataset
import dataset.indexing
import dataset.illumination_statistics
import dataset.metadata
import dataset.utils


# Main interaction point
@click.group()
@click.option("--config", prompt="Configuration file for the compression pipeline",
              help="The configuration file, written in JSON format",
              type=click.File('r'))
@click.pass_context
def cli(context, config):
    params = json.load(config)
    params["metadata"]["plate_maps"] = os.path.join(params["metadata"]["path"], params["metadata"]["plate_maps"])
    params["metadata"]["csv_list"] = os.path.join(params["metadata"]["path"], params["metadata"]["csv_list"])
    params["metadata"]["filename"] = os.path.join(params["metadata"]["path"], "metadata.csv")
    params["metadata"]["labels_file"] = os.path.join(params["metadata"]["path"], params["metadata"]["label_field"]+".csv")
    process = dataset.utils.Parallel(params)
    context.obj["config"] = params
    context.obj["process"] = process


# First dataset tool: Create metadata files
@cli.command()
@click.pass_context
def metadata(context):
    dataset.indexing.create_metadata_index(context.obj["config"])


# Second dataset tool: Compute illumination statistics
@cli.command()
@click.pass_context
def illumination(context):
    process = context.obj["process"]
    metadata = dataset.metadata.read_plates(context.obj["config"]["metadata"]["filename"])
    process.compute(dataset.illumination_statistics.calculate_statistics, metadata)


# Third dataset tool: Compress images
@cli.command()
@click.pass_context
def compression(context):
    process = context.obj["process"]
    metadata = dataset.metadata.read_plates(context.obj["config"]["metadata"]["filename"])
    process.compute(dataset.compression.compress_plate, metadata)
    dataset.indexing.write_compression_index(context.obj["config"])


# Fourth dataset tool: Find cell locations
@cli.command()
@click.pass_context
def locations(context):
    process = context.obj["process"]
    metadata = dataset.metadata.read_plates(context.obj["config"]["metadata"]["filename"])
    process.compute(dataset.metadata.create_cell_indices, metadata)


if __name__ == "__main__":
    cli(obj={})
