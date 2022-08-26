import importlib

#################################################
## MAIN TRAINING ROUTINE
#################################################


def learn_model(config, dset, epoch=1, seed=None, verbose=1):
    model_module = importlib.import_module("plugins.models.{}".format(config["train"]["model"]["name"]))
    crop_module = importlib.import_module("plugins.crop_generators.{}".format(config["train"]["model"]["crop_generator"]))
    if config["train"]["model"]["crop_generator"] == 'online_labels_cropgen':
        config["train"]["model"]["params"]["label_smoothing"] = 0

    if "metrics" in config["train"]["model"].keys():
        if type(config["train"]["model"]["metrics"]) not in [list, dict]:
            raise ValueError("Metrics should be a list or dictionary.")
        keras_metrics = [
            "accuracy",
            "binary_accuracy",
            "categorical_accuracy",
            "sparse_categorical_accuracy",
            "top_k_categorical_accuracy",
            "sparse_top_k_categorical_accuracy"
        ]
        if type(config["train"]["model"]["metrics"] is list):
            metrics = list(map(lambda metric: importlib.import_module(
                "plugins.metrics.{}".format(metric)).MetricClass(config, metric).f if metric not in keras_metrics else metric,
                          config["train"]["model"]["metrics"]))
        elif type(config["train"]["model"]["metrics"] is dict):
            metrics = {k: lambda metric: importlib.import_module(
                "plugins.metrics.{}".format(metric)).MetricClass(config, metric).f if metric not in keras_metrics else metric
                       for k, v in config["train"]["model"]["metrics"].items()}
    else:
        metrics = ["accuracy"]

    importlib.invalidate_caches()

    crop_generator = crop_module.GeneratorClass
    val_crop_generator = crop_module.SingleImageGeneratorClass
    model = model_module.model_factory(config, dset, crop_generator, val_crop_generator, True)

    if seed is not None:
        model.seed(seed)

    model.train(epoch, metrics)
    return None


def learn_model_v2(config, epoch=1):
    model_module = importlib.import_module("plugins.models.{}".format(config["train"]["model"]["name"]))
    model = model_module.model_factory(config, None, None, None, True)
    model.train(epoch)
