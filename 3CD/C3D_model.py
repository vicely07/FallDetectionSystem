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
    model.add(Convolution3D(64, 3, 3, 3, activation='relu', 
                            border_mode='same', name='conv1',
                            subsample=(1, 1, 1), 
                            input_shape=(3, 16, 112, 112)))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), 
                           border_mode='valid', name='pool1'))
    # 2nd layer group
    model.add(Convolution3D(128, 3, 3, 3, activation='relu', 
                            border_mode='same', name='conv2',
                            subsample=(1, 1, 1)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), 
                           border_mode='valid', name='pool2'))
    # 3rd layer group
    model.add(Convolution3D(256, 3, 3, 3, activation='relu', 
                            border_mode='same', name='conv3a',
                            subsample=(1, 1, 1)))
    model.add(Convolution3D(256, 3, 3, 3, activation='relu', 
                            border_mode='same', name='conv3b',
                            subsample=(1, 1, 1)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), 
                           border_mode='valid', name='pool3'))
    # 4th layer group
    model.add(Convolution3D(512, 3, 3, 3, activation='relu', 
                            border_mode='same', name='conv4a',
                            subsample=(1, 1, 1)))
    model.add(Convolution3D(512, 3, 3, 3, activation='relu', 
                            border_mode='same', name='conv4b',
                            subsample=(1, 1, 1)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), 
                           border_mode='valid', name='pool4'))
    # 5th layer group
    model.add(Convolution3D(512, 3, 3, 3, activation='relu', 
                            border_mode='same', name='conv5a',
                            subsample=(1, 1, 1)))
    model.add(Convolution3D(512, 3, 3, 3, activation='relu', 
                            border_mode='same', name='conv5b',
                            subsample=(1, 1, 1)))
   
    model.add(MaxPooling3D(pool_size=(2, 1, 1), strides=(2, 1, 1), 
                           border_mode='valid', name='pool5'))

    if summary:
        print(model.summary())
    return model

model = get_model(summary=True)

import caffe_pb2 as caffe

import numpy as np



p = caffe.NetParameter()

p.ParseFromString(

    open('conv3d_deepnetA_sport1m_iter_1900000', 'rb').read()

)



def rot90(W):

    for i in range(W.shape[0]):

        for j in range(W.shape[1]):

            for k in range(W.shape[2]):

                W[i, j, k] = np.rot90(W[i, j, k], 2)

    return W



params = []

conv_layers_indx = [1, 4, 7, 9, 12, 14, 17, 19]

fc_layers_indx = [22, 25, 28]



for i in conv_layers_indx:

    layer = p.layers[i]

    weights_b = np.array(layer.blobs[1].data, dtype=np.float32)

    weights_p = np.array(layer.blobs[0].data, dtype=np.float32).reshape(

        layer.blobs[0].num, layer.blobs[0].channels, layer.blobs[0].length,

        layer.blobs[0].height, layer.blobs[0].width

    )

    weights_p = rot90(weights_p)

    params.append([weights_p, weights_b])

for i in fc_layers_indx:

    layer = p.layers[i]

    weights_b = np.array(layer.blobs[1].data, dtype=np.float32)

    weights_p = np.array(layer.blobs[0].data, dtype=np.float32).reshape(

        layer.blobs[0].num, layer.blobs[0].channels, layer.blobs[0].length,

        layer.blobs[0].height, layer.blobs[0].width)[0,0,0,:,:].T

    params.append([weights_p, weights_b])

    

model_layers_indx = [0, 2, 4, 5, 7, 8, 10, 11]  #conv 

for i, j in zip(model_layers_indx, range(11)):

    model.layers[i].set_weights(params[j])

 

import h5py



model.save_weights('sports1M_weights.h5', overwrite=True)

json_string = model.to_json()

with open('sports1M_model.json', 'w') as f:

    f.write(json_string)
