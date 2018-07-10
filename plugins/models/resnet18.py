import keras
import keras_resnet
import keras_resnet.models
import tensorflow as tf

from deepprofiler.learning.model import DeepProfilerModel


def define_model(config, dset):

    def make_regularizer(transforms, reg_lambda):
        loss = 0
        for i in range(len(transforms)):
            for j in range(i + 1, len(transforms)):
                loss += reg_lambda * tf.reduce_sum(
                    tf.abs(tf.matmul(transforms[i], transforms[j], transpose_a=True, transpose_b=False)))
        return loss

    # 1. Create ResNet architecture to extract features
    input_shape = (
        config["sampling"]["box_size"],  # height
        config["sampling"]["box_size"],  # width
        len(config["image_set"]["channels"])  # channels
    )
    input_image = keras.layers.Input(input_shape)

    model = keras_resnet.models.ResNet18(input_image, include_top=False)#, freeze_bn=not is_training)
    features = keras.layers.GlobalAveragePooling2D(name="pool5")(model.layers[-1].output)
    #features = keras.layers.core.Dropout(0.5)(features)

    # TODO: factorize the multi-target output model

    # 2. Create an output embedding for each target
    class_outputs = []

    i = 0
    for t in dset.targets:
        y = keras.layers.Dense(t.shape[1], activation="softmax", name=t.field_name)(features)
        class_outputs.append(y)
        i += 1

    # 3. Define the regularized loss function
    transforms = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if v.name.find("_embed") != -1]
    if len(transforms) > 1:
        regularizer = make_regularizer(transforms, config["training"]["reg_lambda"])
        def regularized_loss(y_true, y_pred):
            loss = keras.losses.categorical_crossentropy(y_true, y_pred) + regularizer
            return loss
        loss_func = ["categorical_crossentropy"]*(len(transforms)-1) + [regularized_loss]
    else:
        loss_func = ["categorical_crossentropy"]

    # 4. Create and compile model
    model = keras.models.Model(inputs=input_image, outputs=class_outputs)
    print(model.summary())
    print([t.shape for t in transforms])
    optimizer = keras.optimizers.Adam(lr=config["training"]["learning_rate"])
    model.compile(optimizer, loss_func, ["categorical_accuracy"])

    return model


class ModelClass(DeepProfilerModel):
    def __init__(self, config, dset, generator):
        super(ModelClass, self).__init__(config, dset, generator)
        self.model = define_model(config, dset)

