import json

import click

import data.dataset
import data.image_statistics
import data.metadata
import data.utils


@click.group()
@click.option("--config")  # TODO: use a file argument http://click.pocoo.org/5/arguments/#file-arguments
@click.pass_context
def cli(context, config):
    params = json.load(open(config))

    process = data.utils.Parallel(params)

    metadata = data.metadata.readPlates(params["metadata"]["filename"])

    context.obj["process"] = process

    context.obj["metadata"] = metadata


@cli.command()
@click.pass_context
def stats(context):
    process = context.obj["process"]

    metadata = context.obj["metadata"]

    process.compute(data.image_statistics.calculate_statistics, metadata)


@cli.command()
@click.pass_context
def compress(context):
    process = context.obj["process"]

    metadata = context.obj["metadata"]

    process.compute(data.compression.compress_plate, metadata)


if __name__ == "__main__":
    cli(obj={})
