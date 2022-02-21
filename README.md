# Pretrained-VGG-neural-nets-in-TensorFlow
Set of VGG neural net models for TensorFlow. Weights converted from timm module (Pytorch).

## Introduction
Sometimes pretrained nets like VGG useful for segmentation problems. Unfortunately there is only pretrained VGG16 anf VGG19 without batchnorm available for tensorflow.
I made converter from Pytorch to VGG which allows to use all family of VGG models.

## Available models 
`vgg11`, `vgg11_bn`, `vgg13`, `vgg13_bn`, `vgg16`, `vgg16_bn`, `vgg19`, `vgg19_bn`

## Notes

* Converted only feature extraction part (no classification part available)
* Models tested against PyTorch version to give the same result (max diff < 1e-05)

## Usage

```python
from vgg_tensorflow import vgg11_bn

model = vgg11_bn(inp_size=(224, 224, 3), pretrained=True)
```