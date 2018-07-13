from comet_ml import Experiment
import importlib
import keras.metrics


#################################################
## MAIN TRAINING ROUTINE
#################################################


def learn_model(config, dset, epoch=1, seed=None):
    model_module = importlib.import_module("plugins.models.{}".format(config['model']['name']))
    crop_module = importlib.import_module("plugins.crop_generators.{}".format(config['model']['crop_generator']))
    if 'metrics' in config['model'].keys():
        if type(config['model']['metrics']) not in [list, dict]:
            raise ValueError("Metrics should be a list or dictionary.")
        keras_metrics = [
            'accuracy',
            'binary_accuracy',
            'categorical_accuracy',
            'sparse_categorical_accuracy',
            'top_k_categorical_accuracy',
            'sparse_top_k_categorical_accuracy'
        ]
        if type(config['model']['metrics'] is list):
            metrics = list(map(lambda metric: not importlib.import_module(
                "plugins.metrics.{}".format(metric)).MetricClass(config).metric if metric not in keras_metrics else metric,
                          config['model']['metrics']))
        if type(config['model']['metrics'] is dict):
            metrics = {k: lambda metric: not importlib.import_module(
                "plugins.metrics.{}".format(metric)).MetricClass(config).metric if metric not in keras_metrics else metric
                       for k, v in config['model']['metrics'].items()}
    else:
        metrics = None
    importlib.invalidate_caches()

    crop_generator = crop_module.GeneratorClass
    model = model_module.ModelClass(config, dset, crop_generator)

    if seed:
        model.seed(seed)
    model.train(epoch, metrics)
