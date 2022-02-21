"""
Author: ZFTurbo (IPPM RAS)
Set of VGG models converted for TF from:
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vgg.py
"""

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.utils import get_file


WEIGHTS_VGG11 = 'https://github.com/ZFTurbo/Pretrained-VGG-neural-nets-in-TensorFlow/releases/download/v1.0/vgg11.h5'
WEIGHTS_VGG11_BN = 'https://github.com/ZFTurbo/Pretrained-VGG-neural-nets-in-TensorFlow/releases/download/v1.0/vgg11_bn.h5'
WEIGHTS_VGG13 = 'https://github.com/ZFTurbo/Pretrained-VGG-neural-nets-in-TensorFlow/releases/download/v1.0/vgg13.h5'
WEIGHTS_VGG13_BN = 'https://github.com/ZFTurbo/Pretrained-VGG-neural-nets-in-TensorFlow/releases/download/v1.0/vgg13_bn.h5'
WEIGHTS_VGG16 = 'https://github.com/ZFTurbo/Pretrained-VGG-neural-nets-in-TensorFlow/releases/download/v1.0/vgg16.h5'
WEIGHTS_VGG16_BN = 'https://github.com/ZFTurbo/Pretrained-VGG-neural-nets-in-TensorFlow/releases/download/v1.0/vgg16_bn.h5'
WEIGHTS_VGG19 = 'https://github.com/ZFTurbo/Pretrained-VGG-neural-nets-in-TensorFlow/releases/download/v1.0/vgg19.h5'
WEIGHTS_VGG19_BN = 'https://github.com/ZFTurbo/Pretrained-VGG-neural-nets-in-TensorFlow/releases/download/v1.0/vgg19_bn.h5'


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
    if pretrained is True:
        weights_path = get_file('vgg11.h5', WEIGHTS_VGG11)
        model.load_weights(weights_path)
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
    if pretrained is True:
        weights_path = get_file('vgg11_bn.h5', WEIGHTS_VGG11_BN)
        model.load_weights(weights_path)
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
    if pretrained is True:
        weights_path = get_file('vgg13.h5', WEIGHTS_VGG13)
        model.load_weights(weights_path)
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
    if pretrained is True:
        weights_path = get_file('vgg13_bn.h5', WEIGHTS_VGG13_BN)
        model.load_weights(weights_path)
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
    if pretrained is True:
        weights_path = get_file('vgg16.h5', WEIGHTS_VGG16)
        model.load_weights(weights_path)
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
    if pretrained is True:
        weights_path = get_file('vgg16_bn.h5', WEIGHTS_VGG16_BN)
        model.load_weights(weights_path)
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
    if pretrained is True:
        weights_path = get_file('vgg19.h5', WEIGHTS_VGG19)
        model.load_weights(weights_path)
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
    if pretrained is True:
        weights_path = get_file('vgg19_bn.h5', WEIGHTS_VGG19_BN)
        model.load_weights(weights_path)
    return model
