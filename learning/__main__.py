import json
import os

import click

import dataset.image_dataset
import learning.training
import learning.validation

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


if __name__ == "__main__":
    cli(obj={})
