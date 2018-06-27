# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 15:23:21 2018

@author: lykha
"""

from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.convolutional import Convolution3D, MaxPooling3D, ZeroPadding3D
from keras.optimizers import SGD

def get_model(summary=False):
    """ Return the Keras model of the network
    """
    model = Sequential()
    # 1st layer group
    model.add(Convolution3D(3, 3, 3, 64, activation='relu', 
                            border_mode='same', name='conv1',
                            subsample=(1, 1, 1), 
                            input_shape=(3, 16, 112, 112)))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), 
                           border_mode='valid', name='pool1'))
    # 2nd layer group
    model.add(Convolution3D(3, 3, 3, 128, activation='relu', 
                            border_mode='same', name='conv2',
                            subsample=(1, 1, 1)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), 
                           border_mode='valid', name='pool2'))
    # 3rd layer group
    model.add(Convolution3D(3, 3, 3, 256, activation='relu', 
                            border_mode='same', name='conv3a',
                            subsample=(1, 1, 1)))
    model.add(Convolution3D(3, 3, 3, 256, activation='relu', 
                            border_mode='same', name='conv3b',
                            subsample=(1, 1, 1)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), 
                           border_mode='valid', name='pool3'))
    # 4th layer group
    model.add(Convolution3D(3, 3, 3, 512, activation='relu', 
                            border_mode='same', name='conv4a',
                            subsample=(1, 1, 1)))
    model.add(Convolution3D(3, 3, 3, 512, activation='relu', 
                            border_mode='same', name='conv4b',
                            subsample=(1, 1, 1)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), 
                           border_mode='valid', name='pool4'))
    # 5th layer group
    model.add(Convolution3D(3, 3, 3, 512, activation='relu', 
                            border_mode='same', name='conv5a',
                            subsample=(1, 1, 1)))
    model.add(Convolution3D(3, 3, 3, 512 activation='relu', 
                            border_mode='same', name='conv5b',
                            subsample=(1, 1, 1)))
   
    model.add(MaxPooling3D(pool_size=(2, 1, 1), strides=(2, 1, 1), 
                           border_mode='valid', name='pool5'))

    if summary:
        print(model.summary())
    return model

model = get_model(summary=True)

