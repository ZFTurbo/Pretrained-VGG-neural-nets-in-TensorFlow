"""
Author: ZFTurbo
Set of VGG models converted for TF from:
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vgg.py
"""

if __name__ == '__main__':
    import os

    gpu_use = "4"
    print('GPU use: {}'.format(gpu_use))
    os.environ["KERAS_BACKEND"] = "tensorflow"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_use)


from vgg_tensorflow import *
import torch
import timm
import numpy as np
from torch.nn import Identity


def convert_weigths_vgg(type, bn, model_torch, model_tf):
    params = dict(model_torch.named_parameters())
    for name, param in model_torch.named_parameters():
        print(name, param.shape)
    st = model_torch.state_dict()

    if type == 'vgg11' and not bn:
        layers = [0, 3, 6, 8, 11, 13, 16, 18]
    elif type == 'vgg11' and bn:
        layers = [0, 4, 8, 11, 15, 18, 22, 25]
    elif type == 'vgg13' and not bn:
        layers = [0, 2, 5, 7, 10, 12, 15, 17, 20, 22]
    elif type == 'vgg13' and bn:
        layers = [0, 3, 7, 10, 14, 17, 21, 24, 28, 31]
    elif type == 'vgg16' and not bn:
        layers = [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28]
    elif type == 'vgg16' and bn:
        layers = [0, 3, 7, 10, 14, 17, 20, 24, 27, 30, 34, 37, 40]
    elif type == 'vgg19' and not bn:
        layers = [0, 2, 5, 7, 10, 12, 14, 16, 19, 21, 23, 25, 28, 30, 32, 34]
    elif type == 'vgg19' and bn:
        layers = [0, 3, 7, 10, 14, 17, 20, 23, 27, 30, 33, 36, 40, 43, 46, 49]
    else:
        print('Unknown: {} and {}'.format(type, bn))
        exit()

    w = dict()
    b = dict()
    for i in layers:
        w[i] = params['features.{}.weight'.format(i)].cpu().numpy().transpose((2, 3, 1, 0))
        b[i] = params['features.{}.bias'.format(i)].cpu().numpy()
        model_tf.layers[i + 1].set_weights((w[i], b[i]))
        if bn:
            weight = st['features.{}.weight'.format(i + 1)]
            bias = st['features.{}.bias'.format(i + 1)]
            running_mean = st['features.{}.running_mean'.format(i + 1)]
            running_var = st['features.{}.running_var'.format(i + 1)]
            model_tf.layers[i + 2].set_weights((weight, bias, running_mean, running_var))
    return model_tf


if __name__ == '__main__':
    for mod_tf, mod_name in [
        [vgg11, 'vgg11'],
        [vgg11_bn, 'vgg11_bn'],
        [vgg13, 'vgg13'],
        [vgg13_bn, 'vgg13_bn'],
        [vgg16, 'vgg16'],
        [vgg16_bn, 'vgg16_bn'],
        [vgg19, 'vgg19'],
        [vgg19_bn, 'vgg19_bn'],
    ]:
        print('Go for: {}'.format(mod_name))
        model_tf = mod_tf()
        print(model_tf.summary())
        model_torch = timm.create_model(
            mod_name,
            pretrained=True,
            num_classes=0,
            global_pool='',
        )
        bn = False
        if '_bn' in mod_name:
            bn = True
        model_torch.pre_logits = Identity()
        print(model_torch)
        with torch.no_grad():
            model_torch.eval()
            model_tf = convert_weigths_vgg(mod_name.replace('_bn', ''), bn, model_torch, model_tf)
        x1 = np.random.uniform(-1, 1, size=(10, 3, 224, 224)).astype(np.float32)
        x = torch.from_numpy(x1)
        with torch.no_grad():
            res = model_torch(x)
        res_torch = res.cpu().numpy()
        print(res_torch.shape)
        res_tf = model_tf.predict(x1.transpose((0, 2, 3, 1)))
        res_tf = res_tf.transpose((0, 3, 1, 2))
        print(res_tf.shape)
        diff = np.abs(res_torch - res_tf)
        print('Diff:', diff.mean(), diff.min(), diff.max())
        if diff.max() > 0.00001:
            print('Conversion error for {}!'.format(mod_name))
            exit()
        model_tf.save_weights(mod_name + '.h5')