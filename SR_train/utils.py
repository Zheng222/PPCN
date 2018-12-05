from tensorlayer.prepro import imresize, crop, rotation, flip_axis
import random
from scipy import misc
import numpy as np

def get_imgs_fn(file_name, path):
    return misc.imread(path + file_name, mode='RGB')

def augment(x, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip:
            img = flip_axis(img, axis=1)
        if vflip:
            img = flip_axis(img, axis=0)
        if rot90:
            img = rotation(img, rg=90)
        return img
    return _augment(x)

def crop_sub_imgs_fn(x, is_random=True):
    x = crop(x, wrg=384, hrg=384, is_random=is_random)
    x = augment(x)
    return x

def downsample_fn(x, scale=4):
    h, w = x.shape[0:2]
    hs, ws = h // scale, w // scale

    x = imresize(x, size=[hs, ws], interp='bicubic')
    x = imresize(x, size=[h, w], interp='bicubic')
    return x

def datatype(x):
    for i in range(len(x)):
        x[i] = x[i].astype(np.float32)
    return x

def datarange(x):
    for i in range(len(x)):
        x[i] = x[i] / 255.
    return x

def transpose(xs):
    for i in range(len(xs)):
        xs[i] = xs[i].transpose(2, 0, 1)
    return xs

def update_tensorboard(epoch, tb, img_idx, lr, sr, hr):  # tb--> tensorboard
    [lr, sr, hr] = transpose([lr, sr, hr])
    [lr, sr, hr] = datarange([lr, sr, hr])    # for visualizing correctly [0.0, 255.0] --> [0, 1]

    if epoch == 10:
        tb.add_image(str(img_idx) + '_LR', lr, 0)
        tb.add_image(str(img_idx) + '_HR', hr, 0)
    tb.add_image(str(img_idx) + '_SR', sr, epoch)
