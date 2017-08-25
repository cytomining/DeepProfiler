import json
import os

import click

import dataset.image_dataset
import learning.training
import learning.validation
import learning.profiling

# Main interaction point
@click.group()
@click.option("--config", prompt="Configuration file for learning",
              help="Configuration file, written in JSON format",
              type=click.File('r'))
@click.pass_context
def cli(context, config):
    params = json.load(config)
    params["image_set"]["index"] = os.path.join(params["image_set"]["metadata"], "index.csv")
    context.obj["config"] = params


# First learning tool: Training a network
@cli.command()
@click.pass_context
def training(context):
    images = dataset.image_dataset.read_dataset(context.obj["config"])
    learning.training.learn_model(context.obj["config"], images)


# Evaluate a network in the validation set
@cli.command()
@click.pass_context
def validation(context):
    images = dataset.image_dataset.read_dataset(context.obj["config"])
    learning.validation.validate(context.obj["config"], images)

# Profile cells and extract features
@cli.command()
@click.pass_context
@click.option("--part",
              help="Part of index to process", 
              default=-1, 
              type=click.INT)
def profiling(context, part):
    config = context.obj["config"]
    if part >= 0:
        partfile = "index-{0:03d}.csv".format(part)
        config["image_set"]["index"] = os.path.join(config["image_set"]["metadata"], partfile)
    metadata = dataset.image_dataset.read_dataset(context.obj["config"])
    learning.profiling.profile(context.obj["config"], metadata)


if __name__ == "__main__":
    cli(obj={})
