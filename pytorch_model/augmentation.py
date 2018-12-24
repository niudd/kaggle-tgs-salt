import numpy as np
import pandas as pd
import cv2
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug import parameters as iap
import time
import os
import torch
import random
import math
from augmentation_huang import *


## mine
# SEQ = iaa.Sequential([
#     iaa.Fliplr(0.5), # horizontally flip
#     iaa.OneOf([
#         iaa.Noop(),
#         iaa.Noop(),
#         iaa.GaussianBlur(sigma=(0.0, 1.0)),
#         iaa.Multiply((0.5, 1.5)),
#     ]),
#     iaa.OneOf([
#         iaa.Noop(),
#         iaa.Noop(),
#         iaa.CropAndPad(percent=(-0.25, 0.25)),
#         iaa.Affine(rotate=(-10, 10), translate_percent={"x": (-0.25, 0.25)}, mode='symmetric', cval=(0)),
#     ]),
#     iaa.OneOf([
#         iaa.Noop(),
#         iaa.Noop(),
#         #iaa.Noop(),
#         iaa.PerspectiveTransform(scale=(0.04, 0.08)),
#         iaa.PiecewiseAffine(scale=(0.05, 0.1), mode='edge', cval=(0)),
#         #iaa.Affine(scale={"x": (1.0, 1.25), "y": (1.0, 1.25)}),
#     ]),
#     # More as you want ...
# ])

# # a simple kaggler one
# SEQ = iaa.Sequential([
#     iaa.Fliplr(0.5), # horizontally flip
#     iaa.OneOf([
#         iaa.Noop(),
#         iaa.GaussianBlur(sigma=(0.0, 1.0)),
#         iaa.Noop(),
#         iaa.Affine(rotate=(-10, 10), translate_percent={"x": (-0.25, 0.25)}, mode='symmetric', cval=(0)),
#         iaa.Noop(),
#         iaa.PerspectiveTransform(scale=(0.04, 0.08)),
#         iaa.Noop(),
#         iaa.PiecewiseAffine(scale=(0.05, 0.1), mode='edge', cval=(0)),
#     ]),
#     # More as you want ...
# ])

# SEQ = iaa.Sequential([
#     iaa.OneOf([
#     iaa.Fliplr(0.5), # horizontally flip
#     iaa.GaussianBlur(sigma=(0.0, 5.0))
#     ])
# ])


# Train augmentation (on a batch)
# def do_augmentation(x_train, y_train, seed=None):
#     """Endi's
#     """
#     #seq = reseed(SEQ, deterministic=False)
#     seq = SEQ
#     seq_det = seq.to_deterministic()
#     #seed = get_seed()
#     seq_img = seq_det#generate_augmentor(seed, is_img=True)
#     seq_mask = seq_det#generate_augmentor(seed, is_img=False)

#     # 'images' should be either a 4D numpy array of shape (N, height, width, channels)
#     # or a list of 3D numpy arrays, each having shape (height, width, channels).
#     # Grayscale images must have shape (height, width, 1) each.
#     # All images must have numpy's dtype uint8. Values are expected to be in
#     # range 0-255.
    
#     x_train_aug = seq_img.augment_image(x_train.reshape(101, 101, 1))
#     y_train_aug = seq_mask.augment_image(y_train.reshape(101, 101, 1))
#     return x_train_aug, y_train_aug
#     #x_train_aug, y_train_aug = seq_det.augment_images([x_train.reshape(128, 128, 1), y_train.reshape(128, 128, 1)])
#     #return x_train_aug.reshape(1, 128, 128), y_train_aug.reshape(1, 128, 128)

def do_augmentation(image, mask, idx=None):
    """Huang's
    """
    #seed = get_seed()
    #np.random.seed(seed)
    
    if np.random.rand() < 0.5:
        image, mask = do_horizontal_flip2(image, mask)
    
    if np.random.rand() < 0.5:
        c = np.random.choice(3)
        if c==0:
            image, mask = do_random_shift_scale_crop_pad2(image, mask, limit=0.125)
        if c==1:
            image, mask = do_elastic_transform2(image, mask, grid=10, distort=np.random.uniform(0, 0.1))
        if c==2:
            image, mask = do_shift_scale_rotate2(image, mask, dx=0, dy=0, scale=1, angle=np.random.uniform(0, 10))
    
    if np.random.rand() < 0.5:
        c = np.random.choice(3)
        if c==0:
            image = do_brightness_shift(image, np.random.uniform(-0.05, +0.05))
        if c==1:
            image = do_brightness_multiply(image, np.random.uniform(1-0.05, 1+0.05))
        if c==2:
            image = do_gamma(image, np.random.uniform(1-0.05, 1+0.05))

    #image, mask = do_center_pad_to_factor2(image, mask, factor=32)

    return image, mask#.astype(int)


# def reseed(augmenter, deterministic=True):
#     augmenter.random_state = ia.new_random_state(get_seed())
#     if deterministic:
#         augmenter.deterministic = True

#     for lists in augmenter.get_children_lists():
#         for aug in lists:
#             aug = reseed(aug, deterministic=True)
#     return augmenter

def get_seed():
    seed = int(time.time()) + int(os.getpid())
    return seed


# def generate_augmentor():
#     """SEQ = generate_augmentor()
#     """
#     np.random.seed(get_seed())
#     aug_list = []
#     param = np.random.choice([0.1, 10.0])
#     aug_list.append(iaa.GaussianBlur(sigma=(0.0, param)))
#     seq = iaa.Sequential(aug_list)
#     return seq

# def generate_augmentor(seed, is_img=True):
#     """SEQ = generate_augmentor()
#     is_img: True for image, False for mask
#     """
#     #np.random.seed(get_seed())
#     np.random.seed(seed)
    
#     aug_list = []
#     if np.random.binomial(1,0.5)==1:
#         aug_list.append(iaa.Fliplr(1.0))

#     if is_img:
#         if np.random.binomial(1,0.5)==1:
#             c = np.random.choice(2)
#             if c==0:
#                 param = np.random.uniform(0.5, 1.0)
#                 aug_list.append(iaa.GaussianBlur(sigma=(0.0, param)))
#             elif c==1:
#                 param = np.random.uniform(0.5, 1.5)
#                 aug_list.append(iaa.Multiply((param*0.95, param)))

#     if np.random.binomial(1,0.5)==1:
#         c = np.random.choice(2)
#         if c==0:
#             param = np.random.uniform(-0.25, 0.25)
#             aug_list.append(iaa.CropAndPad(percent=(param*0.95, param)))
#         elif c==1:
#             param0 = np.random.uniform(-10, 10)
#             param1 = np.random.uniform(-0.25, 0.25)
#             aug_list.append(iaa.Affine(rotate=(param0*0.95, param0), \
#                                        translate_percent={"x": (param1*0.95, param1)}, \
#                                        mode='symmetric', cval=(0)))
    
#     if np.random.binomial(1,0.5)==1:
#         c = np.random.choice(3)
#         if c==0:
#             param = np.random.uniform(0.04, 0.08)
#             aug_list.append(iaa.PerspectiveTransform(scale=(param*0.95, param)))
#         elif c==1:
#             param = np.random.uniform(0.05, 0.1)
#             aug_list.append(iaa.PiecewiseAffine(scale=(param*0.95, param), mode='edge', cval=(0)))
#         elif c==2:
#             param0 = np.random.uniform(0.8, 1.25)
#             param1 = np.random.uniform(0.8, 1.25)
#             aug_list.append(iaa.Affine(scale={"x": (param0*0.95, param0), "y": (param1*0.95, param1)}))
    
#     if len(aug_list)==0:
#         aug_list.append(iaa.Noop())
    
#     seq = iaa.Sequential(aug_list)
#     return seq


