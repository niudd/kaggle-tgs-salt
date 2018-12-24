import os
import sys
sys.path.append('..')
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# import cv2
from sklearn.model_selection import train_test_split

from tqdm import tqdm_notebook #, tnrange
#from itertools import chain
from skimage.io import imread, imshow #, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

from keras.models import Model, load_model, save_model
from keras.layers import Input,Dropout,BatchNormalization,Activation,Add
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K
from keras import optimizers
from keras.utils import multi_gpu_model

import tensorflow as tf

from keras.preprocessing.image import array_to_img, img_to_array, load_img#,save_img

import time
t_start = time.time()

#from unet2 import *
from unet_resnet34 import *
from metrics import *

# 指定需要使用的gpu，编号从0开始，多块gpu用逗号分隔
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"# "0", "0, 1"

# 占用百分之十的显存上限
import tensorflow as tf
import keras.backend as K
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95, allow_growth =True)#0.95
sess = tf.Session(config = tf.ConfigProto(gpu_options = gpu_options))
K.set_session(sess)

SEED = 1234
save_model_name = "model/0915_unet_seed%d_epoch100_lr005_size128.model"%SEED
#sub_name = "submissions/0910_unet080_seed%d_epoch50.csv.gz"%SEED

def load_data(img_size_ori, img_size_target, pickle=True):

    def upsample(img):# img_size_ori=img_size_target: identity
        if img_size_ori == img_size_target:
            return img
        return resize(img, (img_size_target, img_size_target), mode='constant', preserve_range=True)

    def downsample(img):
        if img_size_ori == img_size_target:
            return img
        return resize(img, (img_size_ori, img_size_ori), mode='constant', preserve_range=True)

    if not pickle:
        # Loading of training/testing ids and depths
        train_df = pd.read_csv("data/raw/train.csv", index_col="id", usecols=[0])
        depths_df = pd.read_csv("data/raw/depths.csv", index_col="id")
        train_df = train_df.join(depths_df)
        test_df = depths_df[~depths_df.index.isin(train_df.index)]

        print(len(train_df), len(test_df))

        # #Plotting the depth distributions
        # sns.distplot(train_df.z, label="Train")
        # sns.distplot(test_df.z, label="Test")
        # plt.legend()
        # plt.title("Depth distribution")

        # Loading images into numpy array
        train_df["images"] = [np.array(load_img("data/raw/train_images/images/{}.png".format(idx), color_mode = "grayscale")) / 255 for idx in tqdm_notebook(train_df.index)]
        train_df["masks"] = [np.array(load_img("data/raw/train_images/masks/{}.png".format(idx), color_mode = "grayscale")) / 255 for idx in tqdm_notebook(train_df.index)]


        train_df["coverage"] = train_df.masks.map(np.sum) / img_size_ori**2

        def cov_to_class(val):    
            for i in range(0, 11):
                if val * 10 <= i :
                    return i

        train_df["coverage_class"] = train_df.coverage.map(cov_to_class)


        # Create train/validation split stratified by salt coverage
        ids_train, ids_valid, x_train, x_valid, y_train, y_valid, cov_train, cov_test, depth_train, depth_test = train_test_split(
            train_df.index.values,
            np.array(train_df.images.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1), 
            np.array(train_df.masks.map(upsample).tolist()).reshape(-1, img_size_target, img_size_target, 1), 
            train_df.coverage.values,
            train_df.z.values,
            test_size=0.2, stratify=train_df.coverage_class, random_state=SEED)

        #Data augmentation
        x_train = np.append(x_train, [np.fliplr(x) for x in x_train], axis=0)
        y_train = np.append(y_train, [np.fliplr(x) for x in y_train], axis=0)
        print(x_train.shape)
        print(y_valid.shape)
    else:
        import pickle
        # with open('train.pkl', 'wb') as f:
        #     pickle.dump([x_train, x_valid, y_train, y_valid], f)

        with open('train.pkl', 'rb') as f:
            x_train, x_valid, y_train, y_valid = pickle.load(f)
        print(x_train.shape)
        print(y_valid.shape)
    return x_train, x_valid, y_train, y_valid


def train_model(unet0=True, unet1=True):
    img_size_ori = 101
    img_size_target = 128
    
    x_train, x_valid, y_train, y_valid = load_data(img_size_ori, img_size_target, pickle=True)
    
    if unet0:
        print('unet0')
        # model
        model1 = build_model(img_size_target=img_size_target)
        #model1 = multi_gpu_model(model1, gpus=2, cpu_merge=True, cpu_relocation=False)# multi-gpu version

        c = optimizers.sgd(lr = 0.1)#optimizers.adam(lr = 0.005)
        model1.compile(loss="binary_crossentropy", optimizer=c, metrics=[my_iou_metric])

        #model1.summary()

        #early_stopping = EarlyStopping(monitor='my_iou_metric', mode = 'max',patience=10, verbose=1)
        model_checkpoint = ModelCheckpoint(save_model_name,monitor='my_iou_metric', 
                                           mode = 'max', save_best_only=True, verbose=1)
        reduce_lr = ReduceLROnPlateau(monitor='my_iou_metric', mode = 'max',factor=0.5, patience=5, min_lr=0.0001, verbose=1)

        epochs = 100
        batch_size = 32
        history = model1.fit(x_train, y_train,
                            validation_data=[x_valid, y_valid], 
                            epochs=epochs,
                            batch_size=batch_size,
                            callbacks=[model_checkpoint,reduce_lr], 
                            verbose=1)

    if unet1:
        print('unet1')
        #model1 = load_model(save_model_name, custom_objects={'my_iou_metric': my_iou_metric})
        model1 = build_model(img_size_target=img_size_target)
        # remove laster activation layer and use losvasz loss
        input_x = model1.layers[0].input
        output_layer = model1.layers[-1].input
        model = Model(input_x, output_layer)
        
        c = optimizers.adam(lr = 0.005)
        # lovasz_loss need input range (-∞，+∞), so cancel the last "sigmoid" activation  
        # Then the default threshod for pixel prediction is 0 instead of 0.5, as in my_iou_metric_2.
        model.compile(loss=lovasz_loss, optimizer=c, metrics=[my_iou_metric_2])

        #model.summary()


        #
        early_stopping = EarlyStopping(monitor='val_my_iou_metric_2', mode = 'max',patience=20, verbose=1)
        model_checkpoint = ModelCheckpoint(save_model_name, monitor='val_my_iou_metric_2', 
                                           mode = 'max', save_best_only=True, verbose=1)
        reduce_lr = ReduceLROnPlateau(monitor='val_my_iou_metric_2', mode = 'max',factor=0.5, patience=5, min_lr=0.0001, verbose=1)
        epochs = 100#50
        batch_size = 32

        history = model.fit(x_train, y_train,
                            validation_data=[x_valid, y_valid], 
                            epochs=epochs,
                            batch_size=batch_size,
                            callbacks=[ model_checkpoint,reduce_lr,early_stopping], 
                            verbose=1)


if __name__ == "__main__":    

    train_model(unet0=True, unet1=False)

    print('Total seconds: ', time.time()-t_start)
