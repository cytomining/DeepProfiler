from comet_ml import Experiment
import tensorflow as tf

from deepprofiler.learning.model import DeepProfilerModel

tf.compat.v1.disable_v2_behavior()


def define_model(config, dset):
   
    # Load InceptionResnetV2 base architecture
    if config["profile"]["use_pretrained_input_size"]:
        input_tensor = tf.compat.v1.keras.layers.Input((299, 299, 3), name="input")
        model = tf.compat.v1.keras.applications.inception_resnet_v2.InceptionResNetV2(
            include_top=True,
            input_tensor=input_tensor,
            weights='imagenet',
            pooling="avg"
        )
        model.summary()
    else:
        input_tensor = tf.compat.v1.keras.layers.Input((
            config["dataset"]["locations"]["box_size"],  # height
            config["dataset"]["locations"]["box_size"],  # width
            len(config["dataset"]["images"]["channels"])  # channels
        ), name="input")
        base = tf.compat.v1.keras.applications.inception_resnet_v2.InceptionResNetV2(
            include_top=False,
            weights=None,
            input_tensor=input_tensor,
            pooling="avg",
            classes=dset.targets[0].shape[1]
        )
        # Create output embedding for each target
        class_outputs = []
        i = 0
        for t in dset.targets:
            y = tf.compat.v1.keras.layers.Dense(t.shape[1], activation="softmax", name=t.field_name)(base.output)
            class_outputs.append(y)
            i += 1
        # Define model
        model = tf.compat.v1.keras.Model(input_tensor, class_outputs)

    # Define optimizer and loss
    optimizer = tf.compat.v1.keras.optimizers.Adam(learning_rate=config["train"]["model"]["params"]["learning_rate"])
    loss = "categorical_crossentropy"

    return model, optimizer, loss


class ModelClass(DeepProfilerModel):
    def __init__(self, config, dset, generator, val_generator, is_training):
        super(ModelClass, self).__init__(config, dset, generator, val_generator, is_training)
        self.feature_model, self.optimizer, self.loss = define_model(config, dset)
