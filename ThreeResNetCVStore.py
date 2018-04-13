import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

config = tf.ConfigProto()  
config.gpu_options.allow_growth=True
session = tf.Session(config=config)

KTF.set_session(session)

#!/usr/bin/env
# coding:utf-8
"""
Created on 18/04/01 上午8:23

base Info
"""
__author__ = 'xiaochenwang94'
__version__ = '1.0'

import numpy as np
from keras import layers
from keras import regularizers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, \
    AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, Merge, concatenate
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.initializers import glorot_uniform
import scipy.misc
from matplotlib.pyplot import imshow
from sklearn.model_selection import StratifiedKFold

import keras.backend as K

K.set_image_data_format('channels_last')
K.set_learning_phase(1)

def identity_block(X, f, filters, stage, block):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters

    X_shortcut = X

    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid',
               kernel_initializer=glorot_uniform(seed=0), kernel_regularizer=regularizers.l2(0.01))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same',
               kernel_initializer=glorot_uniform(seed=0), kernel_regularizer=regularizers.l2(0.01))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid',
               kernel_initializer=glorot_uniform(seed=0), kernel_regularizer=regularizers.l2(0.01))(X)
    X = BatchNormalization(axis=3)(X)

    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X

def convolutional_block(X, f, filters, stage, block, s=2):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters

    X_shortcut = X

    X = Conv2D(F1, (1, 1), strides=(s, s), kernel_initializer=glorot_uniform(seed=0),
               kernel_regularizer=regularizers.l2(0.01))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)

    X = Conv2D(F2, (f, f), strides=(1, 1), padding='same',
               kernel_initializer=glorot_uniform(seed=0), kernel_regularizer=regularizers.l2(0.01))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)

    X = Conv2D(F3, (1, 1), strides=(1, 1), kernel_initializer=glorot_uniform(seed=0),
               kernel_regularizer=regularizers.l2(0.01))(X)
    X = BatchNormalization(axis=3)(X)

    X_shortcut = Conv2D(F3, (1, 1), strides=(s, s),
                        kernel_initializer=glorot_uniform(seed=0), kernel_regularizer=regularizers.l2(0.01))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3)(X_shortcut)

    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X

def ResNet50(input_shape=(64, 64, 3), layer_num=11):
    classes = 1

    X_input = Input(input_shape)
    X = ZeroPadding2D((3, 3))(X_input)

    X = Conv2D(64, (7, 7), strides=(1, 1), kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), strides=(1, 1))(X)

    X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    if layer_num >= 15:
        X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
        X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')

    if layer_num >= 20:
        X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
        X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')

    X = Flatten()(X)
    return X, X_input

def Three_ResNet(R1, R2, R3, X_inputs):

    X = concatenate([R1, R2, R3])
    X = Dense(1, name='fc', kernel_initializer=glorot_uniform(seed=0))(X)

    # Create model
    model = Model(inputs=X_inputs, outputs=X, name='ThreeResNet50')

    return model

from sklearn.cross_validation import train_test_split
stores = np.load('./ResNetData/stores.npy')
print(stores.shape)
stores = stores.reshape(stores.shape[0], 9, 9,-1)
pois = np.load('./ResNetData/pois.npy')
pois = pois.reshape(pois.shape[0],9,9,-1)
roads = np.load('./ResNetData/roads.npy')
roads = roads.reshape(roads.shape[0],9,9,-1)

y_data = np.load('./ResNetData/shopPower_y.npy')
origin_y = y_data.copy()
# data = data.reshape((24331, 9, 9, 52))
y_data = y_data/10

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(stores.reshape(stores.shape[0], -1))
stores = scaler.transform(stores.reshape(stores.shape[0], -1))
stores = stores.reshape(stores.shape[0], 9, 9, -1)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(pois.reshape(pois.shape[0], -1))
pois = scaler.transform(pois.reshape(pois.shape[0], -1))
pois = pois.reshape(pois.shape[0], 9, 9, -1)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(roads.reshape(roads.shape[0], -1))
roads = scaler.transform(roads.reshape(roads.shape[0], -1))
roads = roads.reshape(roads.shape[0], 9, 9, -1)

combine = np.concatenate((stores, pois, roads), axis=3)

def my_loss(y_true,y_pred):
    return K.sqrt(K.mean(K.square(y_true - y_pred), axis=-1))

seed = 7
np.random.seed(seed)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
cvscores = []

for train, test in kfold.split(stores, origin_y):
    model10_stores, instore = ResNet50(input_shape=stores.shape[1:])
    model10_roads, inroad = ResNet50(input_shape=stores.shape[1:])
    model10_pois, in3poi = ResNet50(input_shape=stores.shape[1:])
    
    model10_combine = Three_ResNet(model10_stores, model10_roads, model10_pois, [instore, inroad, in3poi])
    model10_combine.compile(optimizer='adam', loss='mse', metrics=['mse'])
    for _ in range(10):
        lines10_combine = model10_combine.fit([stores[train], stores[train], stores[train]], y_data[train], epochs = 5, batch_size = 16, validation_split=0.2)
        preds10_combine_ = model10_combine.evaluate([stores[test], stores[test], stores[test]], y_data[test])
        print ("model10_combine Loss = " + str(preds10_combine_[0]))
        print ("model10_combine Test Accuracy = " + str(preds10_combine_[1]))
        
    preds10_combine = model10_combine.evaluate([stores[test], stores[test], stores[test]], y_data[test])
    print ("model10_combine Loss = " + str(preds10_combine[0]))
    print ("model10_combine Test Accuracy = " + str(preds10_combine[1]))
    print(preds10_combine)
    cvscores.append(preds10_combine)
    
print('result = {}'.format(cvscores))