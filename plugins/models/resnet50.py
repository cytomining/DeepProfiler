import keras
import keras.applications
import keras.applications.imagenet_utils
import keras.applications.resnet50
import os
import warnings

from deepprofiler.learning.model import DeepProfilerModel

##################################################
# ResNet architecture as defined in "Deep Residual
# Learning for Image Recognition" by Kaiming He,
# Xiangyu Zhang, Shaoqing Ren, Jian Sun
# https://arxiv.org/abs/1512.03385
# Improved architecture according to Tong He et al.
# https://arxiv.org/abs/1812.01187
##################################################


def conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               strides=(2, 2),
               pool_size=(2, 2)):
    """A block that has a conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.
    # Returns
        Output tensor for the block.
    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    filters1, filters2, filters3 = filters
    if keras.backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = keras.layers.Conv2D(filters1, (1, 1), strides=(1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = keras.layers.Activation('relu')(x)

    x = keras.layers.Conv2D(filters2, kernel_size, strides=strides, padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = keras.layers.Activation('relu')(x)

    x = keras.layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c')(x)
    x = keras.layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = keras.layers.MaxPooling2D(pool_size=pool_size, padding='same')(input_tensor)
    shortcut = keras.layers.Conv2D(filters3, (1, 1), strides=(1, 1),
                             kernel_initializer='he_normal',
                             name=conv_name_base + '1')(x)
    shortcut = keras.layers.BatchNormalization(
        axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = keras.layers.add([x, shortcut])
    x = keras.layers.Activation('relu')(x)
    return x


def ResNet50(include_top=True,
             weights='imagenet',
             input_tensor=None,
             input_shape=None,
             pooling=None,
             classes=1000,
             **kwargs):
    """Instantiates the ResNet50 architecture.
    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.
    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 32.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional block.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional block, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    keras.applications.resnet50.resnet50.backend = keras.backend
    keras.applications.resnet50.resnet50.layers = keras.layers

    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    input_shape = keras.applications.imagenet_utils.imagenet_utils._obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=32,
                                      data_format=keras.backend.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)

    if input_tensor is None:
        img_input = keras.layers.Input(shape=input_shape)
    else:
        if not keras.backend.is_keras_tensor(input_tensor):
            img_input = keras.layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    if keras.backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = keras.layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)

    x = keras.layers.Conv2D(64, (7, 7), strides=(2, 2), padding='valid', kernel_initializer='he_normal', name='conv1_t1')(x)
    x = keras.layers.BatchNormalization(axis=bn_axis, name='bn_conv1_t1')(x)
    x = keras.layers.Activation('relu')(x)

    x = keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal', name='conv1_t2')(x)
    x = keras.layers.BatchNormalization(axis=bn_axis, name='bn_conv1_t2')(x)
    x = keras.layers.Activation('relu')(x)

    x = keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer='he_normal', name='conv1_t3')(x)
    x = keras.layers.BatchNormalization(axis=bn_axis, name='bn_conv1_t3')(x)
    x = keras.layers.Activation('relu')(x)

    x = keras.layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
    x = keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), pool_size=(1, 1))
    x = keras.applications.resnet50.resnet50.identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = keras.applications.resnet50.resnet50.identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = keras.applications.resnet50.resnet50.identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = keras.applications.resnet50.resnet50.identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = keras.applications.resnet50.resnet50.identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = keras.applications.resnet50.resnet50.identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = keras.applications.resnet50.resnet50.identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = keras.applications.resnet50.resnet50.identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = keras.applications.resnet50.resnet50.identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = keras.applications.resnet50.resnet50.identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = keras.applications.resnet50.resnet50.identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = keras.applications.resnet50.resnet50.identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    if include_top:
        x = keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x = keras.layers.Dense(classes, activation='softmax', name='fc1000')(x)
    else:
        if pooling == 'avg':
            x = keras.layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = keras.layers.GlobalMaxPooling2D()(x)
        else:
            warnings.warn('The output shape of `ResNet50(include_top=False)` '
                          'has been changed since Keras 2.2.0.')

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = keras.utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = keras.models.Model(inputs, x, name='resnet50')

    return model


def define_model(config, dset):
    # 1. Create ResNet architecture to extract features
    input_shape = (
        config["train"]["sampling"]["box_size"],  # height
        config["train"]["sampling"]["box_size"],  # width
        len(config["dataset"]["images"][
            "channels"])  # channels
    )
    input_image = keras.layers.Input(input_shape)
    model = ResNet50(include_top=False, weights=None, input_tensor=input_image)
    features = keras.layers.GlobalAveragePooling2D(name="pool5")(model.layers[-1].output)

    # 2. Create an output embedding for each target
    class_outputs = []

    i = 0
    for t in dset.targets:
        y = keras.layers.Dense(t.shape[1], activation="softmax", name=t.field_name)(features)
        class_outputs.append(y)
        i += 1

    # 3. Define the loss function
    loss_func = "categorical_crossentropy"

    # 4. Create and compile model
    model = keras.models.Model(inputs=input_image, outputs=class_outputs)
    ## Added weight decay following tricks reported in:
    ## https://github.com/keras-team/keras/issues/2717
    regularizer = keras.regularizers.l2(0.00001)
    for layer in model.layers:
        if hasattr(layer, "kernel_regularizer"):
            setattr(layer, "kernel_regularizer", regularizer)
    model = keras.models.model_from_json(model.to_json())
    optimizer = keras.optimizers.SGD(lr=config["train"]["model"]["params"]["learning_rate"], momentum=0.9, nesterov=True)
    # optimizer = keras.optimizers.Adam(lr=config["train"]["model"]["params"]["learning_rate"])

    return model, optimizer, loss_func


class ModelClass(DeepProfilerModel):
    def __init__(self, config, dset, generator, val_generator):
        super(ModelClass, self).__init__(config, dset, generator, val_generator)
        self.feature_model, self.optimizer, self.loss = define_model(config, dset)


