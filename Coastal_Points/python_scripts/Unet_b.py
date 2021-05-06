from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np

import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()
import tensorflow_probability as tfp
from tensorflow.python.client import device_lib


# %matplotlib inline
tfd = tfp.distributions
from tensorflow.keras import regularizers

import shutil
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore')
from scipy import interp
from scipy.stats import rankdata
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from sklearn import svm, datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.utils import resample
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold,KFold,GroupKFold
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error

import math
import matplotlib.pylab as plt
import random
import copy
import os
import glob
import sys
import pandas as pd
import time
import h5py
import time
import sys
import importlib
import gc
#check this later if broken

# #set seed for reproducibility:
# random.seed(1)

# def get_available_gpus():
#     local_device_protos = device_lib.list_local_devices()
#     return [x.name for x in local_device_protos if x.device_type == 'GPU']

# print(get_available_gpus())


# os.environ["CUDA_VISIBLE_DEVICES"]="0"

# if tf.test.gpu_device_name() != '/device:GPU:0':
#     print('WARNING: GPU device not found.')
# else:
#     print('SUCCESS: Found GPU: {}'.format(tf.test.gpu_device_name()))

# config_object = tf.compat.v1.ConfigProto()
# config_object.gpu_options.allow_growth = True
# session_object = tf.compat.v1.Session(config=config_object)
# tf.compat.v1.keras.backend.set_session(session_object)


from tensorflow.keras import regularizers
from tensorflow.keras.layers import BatchNormalization as BatchNorm
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Reshape, Activation
from tensorflow.keras.layers import Lambda, Flatten
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, ZeroPadding2D,Cropping2D
from tensorflow.keras.layers import MaxPooling2D, Dense
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import backend as K
from tensorflow.keras import layers

#clipped activation function: 
def clip_thresh(x,thresh):
    downmask = K.clip(x,-thresh,thresh)
    return downmask

def downsampling(x, level, filters, kernel_size, num_convs, conv_strides=1, activation = tf.keras.layers.LeakyReLU(alpha=0.2), batch_norm = False, pool_size = 2, pool_strides = 2, regularizer = None, regularizer_param = 0.001):
    if regularizer is not None:
        if regularizer == 'l2':
            reg = regularizers.l2(regularizer_param)
        elif regularizer == 'l1':
            reg = regularizers.l1(regularizer_param)
    else:
        reg = None
    for i in range(num_convs):
        x = Conv2D(filters=filters, kernel_size=kernel_size, strides=conv_strides, padding='same', kernel_regularizer=reg, bias_regularizer=reg, name = 'downsampling_' + str(level) + '_conv_' + str(i))(x)
        if batch_norm:
            x = BatchNorm(name = 'downsampling_' + str(level) + '_batchnorm_' + str(i))(x)
        x = Activation(activation, name = 'downsampling_' + str(level) + '_activation_' + str(i))(x)
    skip = x
    x = MaxPooling2D(pool_size=2, strides=2, padding='same')(x)
    return x, skip

def bottleneck_dilated(x, filters, kernel_size, num_convs = 6, activation = 'relu', batch_norm = False, last_activation = False, regularizer = None, regularizer_param = 0.001):
#     assert num_convs == len(conv_strides)
    if regularizer is not None:
        if regularizer == 'l2':
            reg = regularizers.l2(regularizer_param)
        elif regularizer == 'l1':
            reg = regularizers.l1(regularizer_param)
    else:
        reg = None
    skips = []
    for i in range(num_convs):
        x = Conv2D(filters=filters, kernel_size=kernel_size, strides=1, dilation_rate = 2 ** i, activation=tf.keras.layers.LeakyReLU(alpha=0.2), padding='same', kernel_regularizer=reg, bias_regularizer=reg, name = 'bottleneck_skip_' + str(i))(x)
        skips.append(x)
    x = layers.add(skips)
    if last_activation:
        x = Activation(tf.keras.layers.LeakyReLU(alpha=0.2))(x)
    return x
    
def bottleneck(x, filters, kernel_size, num_convs, conv_strides=1, activation = 'relu', batch_norm = False, pool_size = 2, pool_strides = 2, regularizer = None, regularizer_param = 0.001):
    if regularizer is not None:
        if regularizer == 'l2':
            reg = regularizers.l2(regularizer_param)
        elif regularizer == 'l1':
            reg = regularizers.l1(regularizer_param)
    else:
        reg = None
    for i in range(num_convs):
        x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', kernel_regularizer=reg, bias_regularizer=reg, name = 'bottleneck_' + str(i))(x)
        if batch_norm:
            x = BatchNorm()(x)
        x = Activation(activation)(x)
    return x

def upsampling(x, level, skip, filters, kernel_size, num_convs, conv_strides=1, activation = 'relu', batch_norm = False, conv_transpose = True, upsampling_size = 2, upsampling_strides = 2, regularizer = None, regularizer_param = 0.001):
    if regularizer is not None:
        if regularizer == 'l2':
            reg = regularizers.l2(regularizer_param)
        elif regularizer == 'l1':
            reg = regularizers.l1(regularizer_param)
    else:
        reg = None
    if conv_transpose:
        x = Conv2DTranspose(filters=filters, kernel_size = upsampling_size, strides=upsampling_strides, name = 'upsampling_' + str(level) + '_conv_trans_' + str(level))(x)
    else:
        x = UpSampling2D((upsampling_size), name = 'upsampling_' + str(level) + '_ups_' + str(i))(x)
    x = Concatenate()([x, skip])
    for i in range(num_convs):
        x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', kernel_regularizer=reg, bias_regularizer=reg, name = 'upsampling_' + str(level) + '_conv_' + str(i))(x)
        if batch_norm:
            print('Im batch norm upsample')
            x = BatchNorm(name = 'upsampling_' + str(level) + '_batchnorm_' + str(i))(x)
        x = Activation(activation, name = 'upsampling_' + str(level) + '_activation_' + str(i))(x)
    return x

def model_simple_unet_initializer(img_shape,num_classes, num_levels, num_layers = 2, num_bottleneck = 2, cost_func='mse',filter_size_start = 16, batch_norm = False, kernel_size = 3, bottleneck_dilation = False, bottleneck_sum_activation = False, regularizer = None, regularizer_param = 0.001):
    inputs = Input((img_shape))
    x = inputs
    x = inputs
    x = ZeroPadding2D(((0, 1), (1, 2)))(x)
    skips = []
    for i in range(num_levels):
        x, skip = downsampling(x, i, filter_size_start * (2 ** i), kernel_size, num_layers, batch_norm=True, regularizer= regularizer, regularizer_param=regularizer_param)
        skips.append(skip)
    if bottleneck_dilation:
        x = bottleneck_dilated(x, filter_size_start * (2 ** num_levels), kernel_size, num_bottleneck, batch_norm=True, last_activation=bottleneck_sum_activation, regularizer= regularizer, regularizer_param=regularizer_param)
    else:
        x = bottleneck(x, filter_size_start * (2 ** num_levels), kernel_size, num_bottleneck, batch_norm=True, regularizer=regularizer, regularizer_param=regularizer_param)
    
    for j in range(num_levels):
        x = upsampling(x, j, skips[num_levels - j - 1], filter_size_start * (2 ** (num_levels - j - 1)), kernel_size, num_layers, batch_norm=False, regularizer= regularizer, regularizer_param=regularizer_param)
    outputs = Conv2D(filters=num_classes, kernel_size=1, strides=1, padding='same', activation='linear',name = 'linear')(x)
        
    outputs = Cropping2D(cropping=((0, 1), (1, 2)))(outputs)
    
    model = Model(inputs = inputs, outputs = outputs)
    opt = optimizer=tf.optimizers.Adam(learning_rate=0.001)

    model.compile(optimizer='adam', loss=cost_func)
    model.summary()
    return model


def model_simple_CNN_initializer(num_classes, num_levels, num_layers = 4, num_bottleneck = 2, filter_size_start = 16, batch_norm = False, kernel_size = 3, bottleneck_dilation = False, bottleneck_sum_activation = False, regularizer = None, regularizer_param = 0.001):
    inputs = Input((img_shape))
    x = inputs
    x = inputs
    x = ZeroPadding2D(((0, 1), (1, 2)))(x)
    skips = []
    for i in range(num_levels):
        x, skip = downsampling(x, i, filter_size_start * (2 ** i), kernel_size, num_layers, batch_norm=False, regularizer= regularizer, regularizer_param=regularizer_param)
        skips.append(skip)
    if bottleneck_dilation:
        x = bottleneck_dilated(x, filter_size_start * (2 ** num_levels), kernel_size, num_bottleneck, batch_norm=False, last_activation=bottleneck_sum_activation, regularizer= regularizer, regularizer_param=regularizer_param)
    else:
        x = bottleneck(x, filter_size_start * (2 ** num_levels), kernel_size, num_bottleneck, batch_norm=False, regularizer=regularizer, regularizer_param=regularizer_param)

    outputs = Flatten()(x)
    outputs = Dense(targ_shape,activation='linear',)(outputs)
    
    model = Model(inputs = inputs, outputs = outputs)
    opt = optimizer=tf.optimizers.Adam(learning_rate=0.001)

    model.compile(optimizer='adam', loss='mse')
    model.summary()
    return model




# img_height = 64
# img_width = 33
# num_channels = 4
# img_shape = (img_height, img_width, num_channels)
# nummy_classes = 2

# UNET = model_simple_unet_initializer(num_classes=nummy_classes, num_levels=4, num_layers =3, num_bottleneck = 3, filter_size_start =16, batch_norm=None, kernel_size = 3, 
#                                      bottleneck_dilation = True, bottleneck_sum_activation =False)
