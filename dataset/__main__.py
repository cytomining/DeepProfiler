import json

import click

import dataset.compression
import dataset.image_dataset
import dataset.illumination_statistics
import dataset.metadata
import dataset.utils


# Main interaction point
@click.group()
@click.option("--config", prompt="Configuration file for the compression pipeline",
              help="The configuration file, written in JSON format",
              type=click.File('r'))  # TODO: use a file argument http://click.pocoo.org/5/arguments/#file-arguments
@click.pass_context
def cli(context, config):
    params = json.load(config)
    process = dataset.utils.Parallel(params)
    metadata = dataset.metadata.readPlates(params["metadata"]["filename"])
    context.obj["process"] = process
    context.obj["metadata"] = metadata


# First dataset tool: Compute illumination statistics
@cli.command()
@click.pass_context
def illumination(context):
    process = context.obj["process"]
    metadata = context.obj["metadata"]
    process.compute(dataset.illumination_statistics.calculate_statistics, metadata)


# Second dataset tool: Compress images
@cli.command()
@click.pass_context
def compression(context):
    process = context.obj["process"]
    metadata = context.obj["metadata"]
    process.compute(dataset.compression.compress_plate, metadata)


# Third dataset tool: Find cell locations
@cli.command()
@click.pass_context
def locations(context):
    process = context.obj["process"]
    metadata = context.obj["metadata"]
    process.compute(dataset.metadata.createCellFiles, metadata)


if __name__ == "__main__":
    cli(obj={})
