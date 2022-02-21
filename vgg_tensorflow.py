"""
Author: ZFTurbo (IPPM RAS)
Set of VGG models converted for TF from:
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vgg.py
"""

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model


def vgg11(
    inp_size=(224, 224, 3),
    inp_tensor=None,
    pretrained=True,
):
    if inp_tensor is not None:
        inp = inp_tensor
    else:
        inp = Input(inp_size)
    x = Conv2D(64, kernel_size=(3, 3), padding='same')(inp)
    x = ReLU()(x)
    x = MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')(x)
    x = Conv2D(128, kernel_size=(3, 3), padding='same')(x)
    x = ReLU()(x)
    x = MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')(x)
    x = Conv2D(256, kernel_size=(3, 3), padding='same')(x)
    x = ReLU()(x)
    x = Conv2D(256, kernel_size=(3, 3), padding='same')(x)
    x = ReLU()(x)
    x = MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')(x)
    x = Conv2D(512, kernel_size=(3, 3), padding='same')(x)
    x = ReLU()(x)
    x = Conv2D(512, kernel_size=(3, 3), padding='same')(x)
    x = ReLU()(x)
    x = MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')(x)
    x = Conv2D(512, kernel_size=(3, 3), padding='same')(x)
    x = ReLU()(x)
    x = Conv2D(512, kernel_size=(3, 3), padding='same')(x)
    x = ReLU()(x)
    x = MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')(x)

    model = Model(inputs=inp, outputs=x)
    return model


def vgg11_bn(
    inp_size=(224, 224, 3),
    inp_tensor=None,
    pretrained=True,
):
    if inp_tensor is not None:
        inp = inp_tensor
    else:
        inp = Input(inp_size)
    x = Conv2D(64, kernel_size=(3, 3), padding='same')(inp)
    x = BatchNormalization(epsilon=1e-5, momentum=0.1)(x)
    x = ReLU()(x)
    x = MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')(x)
    x = Conv2D(128, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization(epsilon=1e-5, momentum=0.1)(x)
    x = ReLU()(x)
    x = MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')(x)
    x = Conv2D(256, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization(epsilon=1e-5, momentum=0.1)(x)
    x = ReLU()(x)
    x = Conv2D(256, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization(epsilon=1e-5, momentum=0.1)(x)
    x = ReLU()(x)
    x = MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')(x)
    x = Conv2D(512, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization(epsilon=1e-5, momentum=0.1)(x)
    x = ReLU()(x)
    x = Conv2D(512, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization(epsilon=1e-5, momentum=0.1)(x)
    x = ReLU()(x)
    x = MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')(x)
    x = Conv2D(512, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization(epsilon=1e-5, momentum=0.1)(x)
    x = ReLU()(x)
    x = Conv2D(512, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization(epsilon=1e-5, momentum=0.1)(x)
    x = ReLU()(x)
    x = MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')(x)

    model = Model(inputs=inp, outputs=x)
    return model


def vgg13(
    inp_size=(224, 224, 3),
    inp_tensor=None,
    pretrained=True,
):
    if inp_tensor is not None:
        inp = inp_tensor
    else:
        inp = Input(inp_size)
    x = Conv2D(64, kernel_size=(3, 3), padding='same')(inp)
    x = ReLU()(x)
    x = Conv2D(64, kernel_size=(3, 3), padding='same')(x)
    x = ReLU()(x)
    x = MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')(x)
    x = Conv2D(128, kernel_size=(3, 3), padding='same')(x)
    x = ReLU()(x)
    x = Conv2D(128, kernel_size=(3, 3), padding='same')(x)
    x = ReLU()(x)
    x = MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')(x)
    x = Conv2D(256, kernel_size=(3, 3), padding='same')(x)
    x = ReLU()(x)
    x = Conv2D(256, kernel_size=(3, 3), padding='same')(x)
    x = ReLU()(x)
    x = MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')(x)
    x = Conv2D(512, kernel_size=(3, 3), padding='same')(x)
    x = ReLU()(x)
    x = Conv2D(512, kernel_size=(3, 3), padding='same')(x)
    x = ReLU()(x)
    x = MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')(x)
    x = Conv2D(512, kernel_size=(3, 3), padding='same')(x)
    x = ReLU()(x)
    x = Conv2D(512, kernel_size=(3, 3), padding='same')(x)
    x = ReLU()(x)
    x = MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')(x)

    model = Model(inputs=inp, outputs=x)
    return model


def vgg13_bn(
    inp_size=(224, 224, 3),
    inp_tensor=None,
    pretrained=True,
):
    if inp_tensor is not None:
        inp = inp_tensor
    else:
        inp = Input(inp_size)
    x = Conv2D(64, kernel_size=(3, 3), padding='same')(inp)
    x = BatchNormalization(epsilon=1e-5, momentum=0.1)(x)
    x = ReLU()(x)
    x = Conv2D(64, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization(epsilon=1e-5, momentum=0.1)(x)
    x = ReLU()(x)
    x = MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')(x)
    x = Conv2D(128, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization(epsilon=1e-5, momentum=0.1)(x)
    x = ReLU()(x)
    x = Conv2D(128, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization(epsilon=1e-5, momentum=0.1)(x)
    x = ReLU()(x)
    x = MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')(x)
    x = Conv2D(256, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization(epsilon=1e-5, momentum=0.1)(x)
    x = ReLU()(x)
    x = Conv2D(256, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization(epsilon=1e-5, momentum=0.1)(x)
    x = ReLU()(x)
    x = MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')(x)
    x = Conv2D(512, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization(epsilon=1e-5, momentum=0.1)(x)
    x = ReLU()(x)
    x = Conv2D(512, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization(epsilon=1e-5, momentum=0.1)(x)
    x = ReLU()(x)
    x = MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')(x)
    x = Conv2D(512, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization(epsilon=1e-5, momentum=0.1)(x)
    x = ReLU()(x)
    x = Conv2D(512, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization(epsilon=1e-5, momentum=0.1)(x)
    x = ReLU()(x)
    x = MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')(x)

    model = Model(inputs=inp, outputs=x)
    return model


def vgg16(
    inp_size=(224, 224, 3),
    inp_tensor=None,
    pretrained=True,
):
    if inp_tensor is not None:
        inp = inp_tensor
    else:
        inp = Input(inp_size)
    x = Conv2D(64, kernel_size=(3, 3), padding='same')(inp)
    x = ReLU()(x)
    x = Conv2D(64, kernel_size=(3, 3), padding='same')(x)
    x = ReLU()(x)
    x = MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')(x)
    x = Conv2D(128, kernel_size=(3, 3), padding='same')(x)
    x = ReLU()(x)
    x = Conv2D(128, kernel_size=(3, 3), padding='same')(x)
    x = ReLU()(x)
    x = MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')(x)
    x = Conv2D(256, kernel_size=(3, 3), padding='same')(x)
    x = ReLU()(x)
    x = Conv2D(256, kernel_size=(3, 3), padding='same')(x)
    x = ReLU()(x)
    x = Conv2D(256, kernel_size=(3, 3), padding='same')(x)
    x = ReLU()(x)
    x = MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')(x)
    x = Conv2D(512, kernel_size=(3, 3), padding='same')(x)
    x = ReLU()(x)
    x = Conv2D(512, kernel_size=(3, 3), padding='same')(x)
    x = ReLU()(x)
    x = Conv2D(512, kernel_size=(3, 3), padding='same')(x)
    x = ReLU()(x)
    x = MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')(x)
    x = Conv2D(512, kernel_size=(3, 3), padding='same')(x)
    x = ReLU()(x)
    x = Conv2D(512, kernel_size=(3, 3), padding='same')(x)
    x = ReLU()(x)
    x = Conv2D(512, kernel_size=(3, 3), padding='same')(x)
    x = ReLU()(x)
    x = MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')(x)

    model = Model(inputs=inp, outputs=x)
    return model


def vgg16_bn(
    inp_size=(224, 224, 3),
    inp_tensor=None,
    pretrained=True,
):
    if inp_tensor is not None:
        inp = inp_tensor
    else:
        inp = Input(inp_size)
    x = Conv2D(64, kernel_size=(3, 3), padding='same')(inp)
    x = BatchNormalization(epsilon=1e-5, momentum=0.1)(x)
    x = ReLU()(x)
    x = Conv2D(64, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization(epsilon=1e-5, momentum=0.1)(x)
    x = ReLU()(x)
    x = MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')(x)
    x = Conv2D(128, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization(epsilon=1e-5, momentum=0.1)(x)
    x = ReLU()(x)
    x = Conv2D(128, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization(epsilon=1e-5, momentum=0.1)(x)
    x = ReLU()(x)
    x = MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')(x)
    x = Conv2D(256, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization(epsilon=1e-5, momentum=0.1)(x)
    x = ReLU()(x)
    x = Conv2D(256, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization(epsilon=1e-5, momentum=0.1)(x)
    x = ReLU()(x)
    x = Conv2D(256, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization(epsilon=1e-5, momentum=0.1)(x)
    x = ReLU()(x)
    x = MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')(x)
    x = Conv2D(512, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization(epsilon=1e-5, momentum=0.1)(x)
    x = ReLU()(x)
    x = Conv2D(512, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization(epsilon=1e-5, momentum=0.1)(x)
    x = ReLU()(x)
    x = Conv2D(512, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization(epsilon=1e-5, momentum=0.1)(x)
    x = ReLU()(x)
    x = MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')(x)
    x = Conv2D(512, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization(epsilon=1e-5, momentum=0.1)(x)
    x = ReLU()(x)
    x = Conv2D(512, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization(epsilon=1e-5, momentum=0.1)(x)
    x = ReLU()(x)
    x = Conv2D(512, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization(epsilon=1e-5, momentum=0.1)(x)
    x = ReLU()(x)
    x = MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')(x)

    model = Model(inputs=inp, outputs=x)
    return model


def vgg19(
    inp_size=(224, 224, 3),
    inp_tensor=None,
    pretrained=True,
):
    if inp_tensor is not None:
        inp = inp_tensor
    else:
        inp = Input(inp_size)
    x = Conv2D(64, kernel_size=(3, 3), padding='same')(inp)
    x = ReLU()(x)
    x = Conv2D(64, kernel_size=(3, 3), padding='same')(x)
    x = ReLU()(x)
    x = MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')(x)
    x = Conv2D(128, kernel_size=(3, 3), padding='same')(x)
    x = ReLU()(x)
    x = Conv2D(128, kernel_size=(3, 3), padding='same')(x)
    x = ReLU()(x)
    x = MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')(x)
    x = Conv2D(256, kernel_size=(3, 3), padding='same')(x)
    x = ReLU()(x)
    x = Conv2D(256, kernel_size=(3, 3), padding='same')(x)
    x = ReLU()(x)
    x = Conv2D(256, kernel_size=(3, 3), padding='same')(x)
    x = ReLU()(x)
    x = Conv2D(256, kernel_size=(3, 3), padding='same')(x)
    x = ReLU()(x)
    x = MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')(x)
    x = Conv2D(512, kernel_size=(3, 3), padding='same')(x)
    x = ReLU()(x)
    x = Conv2D(512, kernel_size=(3, 3), padding='same')(x)
    x = ReLU()(x)
    x = Conv2D(512, kernel_size=(3, 3), padding='same')(x)
    x = ReLU()(x)
    x = Conv2D(512, kernel_size=(3, 3), padding='same')(x)
    x = ReLU()(x)
    x = MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')(x)
    x = Conv2D(512, kernel_size=(3, 3), padding='same')(x)
    x = ReLU()(x)
    x = Conv2D(512, kernel_size=(3, 3), padding='same')(x)
    x = ReLU()(x)
    x = Conv2D(512, kernel_size=(3, 3), padding='same')(x)
    x = ReLU()(x)
    x = Conv2D(512, kernel_size=(3, 3), padding='same')(x)
    x = ReLU()(x)
    x = MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')(x)

    model = Model(inputs=inp, outputs=x)
    return model


def vgg19_bn(
    inp_size=(224, 224, 3),
    inp_tensor=None,
    pretrained=True,
):
    if inp_tensor is not None:
        inp = inp_tensor
    else:
        inp = Input(inp_size)
    x = Conv2D(64, kernel_size=(3, 3), padding='same')(inp)
    x = BatchNormalization(epsilon=1e-5, momentum=0.1)(x)
    x = ReLU()(x)
    x = Conv2D(64, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization(epsilon=1e-5, momentum=0.1)(x)
    x = ReLU()(x)
    x = MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')(x)
    x = Conv2D(128, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization(epsilon=1e-5, momentum=0.1)(x)
    x = ReLU()(x)
    x = Conv2D(128, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization(epsilon=1e-5, momentum=0.1)(x)
    x = ReLU()(x)
    x = MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')(x)
    x = Conv2D(256, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization(epsilon=1e-5, momentum=0.1)(x)
    x = ReLU()(x)
    x = Conv2D(256, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization(epsilon=1e-5, momentum=0.1)(x)
    x = ReLU()(x)
    x = Conv2D(256, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization(epsilon=1e-5, momentum=0.1)(x)
    x = ReLU()(x)
    x = Conv2D(256, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization(epsilon=1e-5, momentum=0.1)(x)
    x = ReLU()(x)
    x = MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')(x)
    x = Conv2D(512, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization(epsilon=1e-5, momentum=0.1)(x)
    x = ReLU()(x)
    x = Conv2D(512, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization(epsilon=1e-5, momentum=0.1)(x)
    x = ReLU()(x)
    x = Conv2D(512, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization(epsilon=1e-5, momentum=0.1)(x)
    x = ReLU()(x)
    x = Conv2D(512, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization(epsilon=1e-5, momentum=0.1)(x)
    x = ReLU()(x)
    x = MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')(x)
    x = Conv2D(512, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization(epsilon=1e-5, momentum=0.1)(x)
    x = ReLU()(x)
    x = Conv2D(512, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization(epsilon=1e-5, momentum=0.1)(x)
    x = ReLU()(x)
    x = Conv2D(512, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization(epsilon=1e-5, momentum=0.1)(x)
    x = ReLU()(x)
    x = Conv2D(512, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization(epsilon=1e-5, momentum=0.1)(x)
    x = ReLU()(x)
    x = MaxPool2D(pool_size=(2, 2), strides=2, padding='valid')(x)

    model = Model(inputs=inp, outputs=x)
    return model