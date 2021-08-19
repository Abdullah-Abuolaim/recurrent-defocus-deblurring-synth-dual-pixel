"""
Codes for our ICCV 2021 paper: Learning to Reduce Defocus Blur by Realistically
Modeling Dual-Pixel Data.
GitHub: https://github.com/Abdullah-Abuolaim/recurrent-defocus-deblurring-synth-dual-pixel

This module has the convLSTM architecture and loss functions proposed for the
task of video defocus deblurring based on dual-pixel data.

Copyright (c) 2021-present, Abdullah Abuolaim
This source code is licensed under the license found in the LICENSE file in
the root directory of this source tree.

This code imports the modules and starts the implementation based on the
configurations in config.py module.

Note: this code is adapted from the code of "Defocus Deblurring Using Dual-
Pixel Data" paper. Link to GitHub repository:
https://github.com/Abdullah-Abuolaim/defocus-deblurring-dual-pixel

Email: abuolaim@eecs.yorku.ca | abdullah.abuolaim@gmail.com
"""
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler#, TensorBoard
from keras import backend
from config import *
import tensorflow as tf


#this contains both X and Y sobel filters in the format (3,3,1,2)
#size is 3 x 3, it considers 1 input channel and has two output channels: X and Y
sobelFilter_3 = backend.variable([[[[1.,  1.]], [[0.,  2.]], [[-1.,  1.]]],
                                  [[[2.,  0.]], [[0.,  0.]], [[-2.,  0.]]],
                                  [[[1., -1.]], [[0., -2.]], [[-1., -1.]]]])

sobelFilter_7 = backend.variable([[[[  -1.,   -1.]], [[  -4.,   -6.]], [[  -5.,  -15.]], [[   0.,  -20.]], [[   5.,  -15.]], [[   4.,   -6.]], [[   1.,   -1.]]],
                                  [[[  -6.,   -4.]], [[ -24.,  -24.]], [[ -30.,  -60.]], [[   0.,  -80.]], [[  30.,  -60.]], [[  24.,  -24.]], [[   6.,   -4.]]],
                                  [[[ -15.,   -5.]], [[ -60.,  -30.]], [[ -75.,  -75.]], [[   0., -100.]], [[  75.,  -75.]], [[  60.,  -30.]], [[  15.,   -5.]]],
                                  [[[ -20.,    0.]], [[ -80.,    0.]], [[-100.,    0.]], [[   0.,    0.]], [[ 100.,    0.]], [[  80.,    0.]], [[  20.,    0.]]],
                                  [[[ -15.,    5.]], [[ -60.,   30.]], [[ -75.,   75.]], [[   0.,  100.]], [[  75.,   75.]], [[  60.,   30.]], [[  15.,    5.]]],
                                  [[[  -6.,    4.]], [[ -24.,   24.]], [[ -30.,   60.]], [[   0.,   80.]], [[  30.,   60.]], [[  24.,   24.]], [[   6.,    4.]]],
                                  [[[  -1.,    1.]], [[  -4.,    6.]], [[  -5.,   15.]], [[   0.,   20.]], [[   5.,   15.]], [[   4.,    6.]], [[   1.,    1.]]]])

sobelFilter_11 = backend.variable([[[[-1.0000e+00, -1.0000e+00]], [[-8.0000e+00, -1.0000e+01]], [[-2.7000e+01, -4.5000e+01]], [[-4.8000e+01, -1.2000e+02]], [[-4.2000e+01, -2.1000e+02]], [[ 0.0000e+00, -2.5200e+02]], [[ 4.2000e+01, -2.1000e+02]], [[ 4.8000e+01, -1.2000e+02]], [[ 2.7000e+01, -4.5000e+01]], [[ 8.0000e+00, -1.0000e+01]], [[ 1.0000e+00, -1.0000e+00]]],
                                   [[[-1.0000e+01, -8.0000e+00]], [[-8.0000e+01, -8.0000e+01]], [[-2.7000e+02, -3.6000e+02]], [[-4.8000e+02, -9.6000e+02]], [[-4.2000e+02, -1.6800e+03]], [[ 0.0000e+00, -2.0160e+03]], [[ 4.2000e+02, -1.6800e+03]], [[ 4.8000e+02, -9.6000e+02]], [[ 2.7000e+02, -3.6000e+02]], [[ 8.0000e+01, -8.0000e+01]], [[ 1.0000e+01, -8.0000e+00]]],
                                   [[[-4.5000e+01, -2.7000e+01]], [[-3.6000e+02, -2.7000e+02]], [[-1.2150e+03, -1.2150e+03]], [[-2.1600e+03, -3.2400e+03]], [[-1.8900e+03, -5.6700e+03]], [[ 0.0000e+00, -6.8040e+03]], [[ 1.8900e+03, -5.6700e+03]], [[ 2.1600e+03, -3.2400e+03]], [[ 1.2150e+03, -1.2150e+03]], [[ 3.6000e+02, -2.7000e+02]], [[ 4.5000e+01, -2.7000e+01]]],
                                   [[[-1.2000e+02, -4.8000e+01]], [[-9.6000e+02, -4.8000e+02]], [[-3.2400e+03, -2.1600e+03]], [[-5.7600e+03, -5.7600e+03]], [[-5.0400e+03, -1.0080e+04]], [[ 0.0000e+00, -1.2096e+04]], [[ 5.0400e+03, -1.0080e+04]], [[ 5.7600e+03, -5.7600e+03]], [[ 3.2400e+03, -2.1600e+03]], [[ 9.6000e+02, -4.8000e+02]], [[ 1.2000e+02, -4.8000e+01]]],
                                   [[[-2.1000e+02, -4.2000e+01]], [[-1.6800e+03, -4.2000e+02]], [[-5.6700e+03, -1.8900e+03]], [[-1.0080e+04, -5.0400e+03]], [[-8.8200e+03, -8.8200e+03]], [[ 0.0000e+00, -1.0584e+04]], [[ 8.8200e+03, -8.8200e+03]], [[ 1.0080e+04, -5.0400e+03]], [[ 5.6700e+03, -1.8900e+03]], [[ 1.6800e+03, -4.2000e+02]], [[ 2.1000e+02, -4.2000e+01]]],
                                   [[[-2.5200e+02,  0.0000e+00]], [[-2.0160e+03,  0.0000e+00]], [[-6.8040e+03,  0.0000e+00]], [[-1.2096e+04,  0.0000e+00]], [[-1.0584e+04,  0.0000e+00]], [[ 0.0000e+00,  0.0000e+00]], [[ 1.0584e+04,  0.0000e+00]], [[ 1.2096e+04,  0.0000e+00]], [[ 6.8040e+03,  0.0000e+00]], [[ 2.0160e+03,  0.0000e+00]], [[ 2.5200e+02,  0.0000e+00]]],
                                   [[[-2.1000e+02,  4.2000e+01]], [[-1.6800e+03,  4.2000e+02]], [[-5.6700e+03,  1.8900e+03]], [[-1.0080e+04,  5.0400e+03]], [[-8.8200e+03,  8.8200e+03]], [[ 0.0000e+00,  1.0584e+04]], [[ 8.8200e+03,  8.8200e+03]], [[ 1.0080e+04,  5.0400e+03]], [[ 5.6700e+03,  1.8900e+03]], [[ 1.6800e+03,  4.2000e+02]], [[ 2.1000e+02,  4.2000e+01]]],
                                   [[[-1.2000e+02,  4.8000e+01]], [[-9.6000e+02,  4.8000e+02]], [[-3.2400e+03,  2.1600e+03]], [[-5.7600e+03,  5.7600e+03]], [[-5.0400e+03,  1.0080e+04]], [[ 0.0000e+00,  1.2096e+04]], [[ 5.0400e+03,  1.0080e+04]], [[ 5.7600e+03,  5.7600e+03]], [[ 3.2400e+03,  2.1600e+03]], [[ 9.6000e+02,  4.8000e+02]], [[ 1.2000e+02,  4.8000e+01]]],
                                   [[[-4.5000e+01,  2.7000e+01]], [[-3.6000e+02,  2.7000e+02]], [[-1.2150e+03,  1.2150e+03]], [[-2.1600e+03,  3.2400e+03]], [[-1.8900e+03,  5.6700e+03]], [[ 0.0000e+00,  6.8040e+03]], [[ 1.8900e+03,  5.6700e+03]], [[ 2.1600e+03,  3.2400e+03]], [[ 1.2150e+03,  1.2150e+03]], [[ 3.6000e+02,  2.7000e+02]], [[ 4.5000e+01,  2.7000e+01]]],
                                   [[[-1.0000e+01,  8.0000e+00]], [[-8.0000e+01,  8.0000e+01]], [[-2.7000e+02,  3.6000e+02]], [[-4.8000e+02,  9.6000e+02]], [[-4.2000e+02,  1.6800e+03]], [[ 0.0000e+00,  2.0160e+03]], [[ 4.2000e+02,  1.6800e+03]], [[ 4.8000e+02,  9.6000e+02]], [[ 2.7000e+02,  3.6000e+02]], [[ 8.0000e+01,  8.0000e+01]], [[ 1.0000e+01,  8.0000e+00]]],
                                   [[[-1.0000e+00,  1.0000e+00]], [[-8.0000e+00,  1.0000e+01]], [[-2.7000e+01,  4.5000e+01]], [[-4.8000e+01,  1.2000e+02]], [[-4.2000e+01,  2.1000e+02]], [[ 0.0000e+00,  2.5200e+02]], [[ 4.2000e+01,  2.1000e+02]], [[ 4.8000e+01,  1.2000e+02]], [[ 2.7000e+01,  4.5000e+01]], [[ 8.0000e+00,  1.0000e+01]], [[ 1.0000e+00,  1.0000e+00]]]])

# print(float(tf.reduce_max(sobelFilter_3))) # the max of 3x3 sobel filter = 2
# print(float(tf.reduce_max(sobelFilter_7))) # the max of 7x7 sobel filter = 100
# print(float(tf.reduce_max(sobelFilter_11))) ## the max of 11x11 sobel filter = 12096.0

def expanded_sobel(inputTensor, _ksize):
    #this considers data_format = 'channels_last'
    inputChannels = backend.reshape(backend.ones_like(inputTensor[0,0,0,:]),(1,1,-1,1))
    #if you're using 'channels_first', use inputTensor[0,:,0,0] above
    
    if _ksize == 3:
        return ((sobelFilter_3/2.0)* [[[[ms_edge_loss_weight_x,ms_edge_loss_weight_y]]]]) * inputChannels
    elif _ksize == 7:
        return ((sobelFilter_3/100.0)*[[[[ms_edge_loss_weight_x,ms_edge_loss_weight_y]]]]) * inputChannels
    elif _ksize == 11:
        return ((sobelFilter_11/12096.0)*[[[[ms_edge_loss_weight_x,ms_edge_loss_weight_y]]]]) * inputChannels


def sobel_sh_loss_3(y_true,y_pred):
    #get the sobel filter repeated for each input channel
    filt = expanded_sobel(y_true[0],3)

    #calculate the sobel filters for y_true and y_pred
    #this generates twice the number of input channels 
    #a X and Y channel for each input channel
    sobelTrue = backend.depthwise_conv2d(y_true[0],filt)
    sobelPred = backend.depthwise_conv2d(y_pred[0],filt)

    #now you just apply the mse:
    return backend.mean(backend.square(sobelTrue - sobelPred))

def sobel_sh_loss_7(y_true,y_pred):
    #get the sobel filter repeated for each input channel
    filt = expanded_sobel(y_true[0],7)

    #calculate the sobel filters for y_true and y_pred
    #this generates twice the number of input channels 
    #a X and Y channel for each input channel
    sobelTrue = backend.depthwise_conv2d(y_true[0],filt)
    sobelPred = backend.depthwise_conv2d(y_pred[0],filt)

    #now you just apply the mse:
    return backend.mean(backend.square(sobelTrue - sobelPred))

def sobel_sh_loss_11(y_true,y_pred):
    #get the sobel filter repeated for each input channel
    filt = expanded_sobel(y_true[0],11)

    #calculate the sobel filters for y_true and y_pred
    #this generates twice the number of input channels 
    #a X and Y channel for each input channel
    sobelTrue = backend.depthwise_conv2d(y_true[0],filt)
    sobelPred = backend.depthwise_conv2d(y_pred[0],filt)

    #now you just apply the mse:
    return backend.mean(backend.square(sobelTrue - sobelPred))

def custom_mse(y_true, y_pred):
     return backend.mean(backend.square(y_pred - y_true),axis=-1)
 
def unet(_in_data,_stateful=True):#lstm_stateful=True
    # inputs = Input(input_size)
    conv1 = TimeDistributed(Conv2D(32, 3, activation = 'relu', padding = 'same',
                                   kernel_initializer = 'he_normal'))(_in_data)
    conv1 = TimeDistributed(Conv2D(32, 3, activation = 'relu', padding = 'same',
                                   kernel_initializer = 'he_normal'))(conv1)
    pool1 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv1)
    conv2 = TimeDistributed(Conv2D(64, 3, activation = 'relu', padding = 'same',
                                   kernel_initializer = 'he_normal'))(pool1)
    conv2 = TimeDistributed(Conv2D(64, 3, activation = 'relu', padding = 'same',
                                   kernel_initializer = 'he_normal'))(conv2)
    pool2 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv2)
    conv3 = TimeDistributed(Conv2D(128, 3, activation = 'relu', padding = 'same', 
                                   kernel_initializer = 'he_normal'))(pool2)
    conv3 = TimeDistributed(Conv2D(128, 3, activation = 'relu', padding = 'same', 
                                   kernel_initializer = 'he_normal'))(conv3)
    pool3 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv3)
    conv4 = TimeDistributed(Conv2D(256, 3, activation = 'relu', padding = 'same', 
                                   kernel_initializer = 'he_normal'))(pool3)
    conv4 = TimeDistributed(Conv2D(256, 3, activation = 'relu', padding = 'same', 
                                   kernel_initializer = 'he_normal'))(conv4)
    pool4 = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(conv4)

    conv5 = TimeDistributed(Conv2D(512, 3, activation = 'relu', padding = 'same', 
                                   kernel_initializer = 'he_normal'))(pool4)
    conv5=ConvLSTM2D(filters=512, kernel_size=(3, 3), padding='same', 
                     kernel_initializer = 'he_normal', stateful=_stateful,
                     dropout=dropout_rate,return_sequences=True)(conv5) #go_backwards=True

    up6=TimeDistributed(UpSampling2D(size = (2,2)))(conv5)
    conv6 = TimeDistributed(Conv2D(256, 2, activation = 'relu', padding = 'same',
                                   kernel_initializer = 'he_normal'))(up6)
    merge6 = concatenate([conv4,conv6], axis = 4)
    conv6 = TimeDistributed(Conv2D(256, 3, activation = 'relu', padding = 'same',
                                   kernel_initializer = 'he_normal'))(merge6)
    conv6 = TimeDistributed(Conv2D(256, 3, activation = 'relu', padding = 'same',
                                   kernel_initializer = 'he_normal'))(conv6)

    up7=TimeDistributed(UpSampling2D(size = (2,2)))(conv6)
    conv7 = TimeDistributed(Conv2D(128, 2, activation = 'relu', padding = 'same',
                                   kernel_initializer = 'he_normal'))(up7)
    merge7 = concatenate([conv3,conv7], axis = 4)
    conv7 = TimeDistributed(Conv2D(128, 3, activation = 'relu', padding = 'same',
                                   kernel_initializer = 'he_normal'))(merge7)
    conv7 = TimeDistributed(Conv2D(128, 3, activation = 'relu', padding = 'same',
                                   kernel_initializer = 'he_normal'))(conv7)

    up8=TimeDistributed(UpSampling2D(size = (2,2)))(conv7)
    conv8 = TimeDistributed(Conv2D(64, 2, activation = 'relu', padding = 'same',
                                   kernel_initializer = 'he_normal'))(up8)
    merge8 = concatenate([conv2,conv8], axis = 4)
    conv8 = TimeDistributed(Conv2D(64, 3, activation = 'relu', padding = 'same',
                                   kernel_initializer = 'he_normal'))(merge8)
    conv8 = TimeDistributed(Conv2D(64, 3, activation = 'relu', padding = 'same',
                                   kernel_initializer = 'he_normal'))(conv8)

    up9=TimeDistributed(UpSampling2D(size = (2,2)))(conv8)
    conv9 = TimeDistributed(Conv2D(32, 2, activation = 'relu', padding = 'same',
                                   kernel_initializer = 'he_normal'))(up9)
    merge9 = concatenate([conv1,conv9], axis = 4)
    conv9 = TimeDistributed(Conv2D(32, 3, activation = 'relu', padding = 'same',
                                   kernel_initializer = 'he_normal'))(merge9)
    conv9 = TimeDistributed(Conv2D(32, 3, activation = 'relu', padding = 'same',
                                   kernel_initializer = 'he_normal'))(conv9)
    conv9 = TimeDistributed(Conv2D(3, 3, activation = 'relu', padding = 'same',
                                   kernel_initializer = 'he_normal'))(conv9)
    
    if linear_layer:
        conv10 = TimeDistributed(Conv2D(nb_ch_out, 1, activation = None))(conv9)
        conv10 = backend.clip(conv10, 0, 1)
    else:
        conv10 = TimeDistributed(Conv2D(nb_ch_out, 1, activation = 'sigmoid'))(conv9)

    if ms_edge_loss:
        return [conv10,conv10,conv10,conv10]
    else:
        return conv10