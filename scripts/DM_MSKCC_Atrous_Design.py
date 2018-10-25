# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 10:16:25 2018

@author: hirsch
"""

from keras.models import Model
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.convolutional import Conv3D
from keras.initializers import he_normal
from keras.initializers import Orthogonal
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1_l2
from keras.layers import Input, Flatten, Reshape, Permute
from keras.layers.merge import Concatenate
from keras.layers import MaxPooling3D
from keras.layers import AveragePooling3D
from keras.layers.convolutional import Cropping3D
from keras.layers import UpSampling3D
from keras.layers import concatenate
from keras.layers.advanced_activations import PReLU
from keras.utils import print_summary
from keras import regularizers
from keras.optimizers import RMSprop
import keras.backend as K
from keras.optimizers import Adam
import numpy as np
from keras.activations import softmax
from keras.engine import InputLayer

#------------------------------------------------------------------------------------------


def dice_coef(y_true, y_pred):
    smooth = 1e-6
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def Generalised_dice_coef_multilabel2(y_true, y_pred, numLabels=2):
    dice=0
    for index in range(numLabels):
        dice -= dice_coef(y_true[:,:,:,:,index], y_pred[:,:,:,:,index])
    return numLabels + dice

def dice_coef_multilabel6(y_true, y_pred, numLabels=6):
    dice=0
    for index in range(numLabels):
        dice -= dice_coef(y_true[:,:,:,:,index], y_pred[:,:,:,:,index])
    return numLabels + dice
def w_dice_coef(y_true, y_pred, PENALTY):
    smooth = 1e-6
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f) * PENALTY
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_multilabel0(y_true, y_pred):
    index = 0
    dice = dice_coef(y_true[:,:,:,:,index], y_pred[:,:,:,:,index])
    return dice
def dice_coef_multilabel1(y_true, y_pred):
    index = 1
    dice = dice_coef(y_true[:,:,:,:,index], y_pred[:,:,:,:,index])
    return dice
def dice_coef_multilabel2(y_true, y_pred):
    index = 2
    dice = dice_coef(y_true[:,:,:,:,index], y_pred[:,:,:,:,index])
    return dice



dpatch = 15,91,91
output_classes = 2
conv_features_downsample = [10,10,10,10,10,10,10,10,10]
conv_features = [50,50,50,50,50,50,50,50,50,70,70,70,70]#[20,20,20,20,30,30,40,40,60,50,50,50,50,50,50] #[50, 50, 50, 50, 50, 100, 100, 100]
fc_features = [100,100,150]
d_factor = 2  # downsampling factor = stride in downsampling pathway
num_channels = 1
L2 = 0.0001
dropout = [0,0]
learning_rate = 0.01

mod1      = Input((dpatch[0],dpatch[1],dpatch[2], num_channels))

#############   Downsampled pathway   ##################   

#x2        = AveragePooling3D(pool_size=(d_factor[0],d_factor[1],d_factor[2]), padding="same")(mod1)

x3        = Conv3D(filters = 30, 
                   kernel_size = (3,3,3), 
                   dilation_rate = (1,1,1),
                   padding = 'same',
                   kernel_initializer=Orthogonal(),
                   kernel_regularizer=regularizers.l2(L2))(mod1)
x3        = BatchNormalization()(x3)
x3        = Activation('relu')(x3)
#x3        = PReLU()(x3)

x3        = Conv3D(filters = 30, 
                   kernel_size = (3,3,3), 
                   dilation_rate = (1,1,1),
                   padding = 'same',
                   kernel_initializer=Orthogonal(),
                   kernel_regularizer=regularizers.l2(L2))(x3)
x3        = BatchNormalization()(x3)
x3        = Activation('relu')(x3)

# -------------------   Dilation Pyramid

x3        = Conv3D(filters = 40, 
                   kernel_size = (3,3,3), 
                   dilation_rate = (1,2,2),
                   padding = 'valid',
                   #kernel_initializer=he_normal(seed=seed),
                   kernel_initializer=Orthogonal(),
                   kernel_regularizer=regularizers.l2(L2))(x3)
x3        = BatchNormalization()(x3)
x3        = Activation('relu')(x3)

x3        = Conv3D(filters = 40, 
                   kernel_size = (3,3,3), 
                   dilation_rate = (1,4,4),
                   padding = 'valid',
                   #kernel_initializer=he_normal(seed=seed),
                   kernel_initializer=Orthogonal(),
                   kernel_regularizer=regularizers.l2(L2))(x3)
x3        = BatchNormalization()(x3)
x3        = Activation('relu')(x3)

x3        = Conv3D(filters = 50, 
                   kernel_size = (3,3,3), 
                   dilation_rate = (1,8,8),
                   padding = 'valid',
                   #kernel_initializer=he_normal(seed=seed),
                   kernel_initializer=Orthogonal(),
                   kernel_regularizer=regularizers.l2(L2))(x3)
x3        = BatchNormalization()(x3)
x3        = Activation('relu')(x3)

x3        = Conv3D(filters = 50, 
                   kernel_size = (3,3,3), 
                   dilation_rate = (1,16,16),
                   padding = 'valid',
                   #kernel_initializer=he_normal(seed=seed),
                   kernel_initializer=Orthogonal(),
                   kernel_regularizer=regularizers.l2(L2))(x3)
x3        = BatchNormalization()(x3)
x3        = Activation('relu')(x3)

x3        = Conv3D(filters = 50, 
                   kernel_size = (3,3,3), 
                   dilation_rate = (1,10,10),
                   padding = 'valid',
                   #kernel_initializer=he_normal(seed=seed),
                   kernel_initializer=Orthogonal(),
                   kernel_regularizer=regularizers.l2(L2))(x3)
x3        = BatchNormalization()(x3)
x3        = Activation('relu')(x3)


#############   High res pathway   ##################  

x1        = Cropping3D(cropping = ((0,0),(30,30),(30,30)), input_shape=(dpatch[0],dpatch[1],dpatch[2],  num_channels))(mod1)

for feature in conv_features[0:10]:  
    x1        = Conv3D(filters = feature, 
                       kernel_size = (2,3,3), 
                       #kernel_initializer=he_normal(seed=seed),
                       kernel_initializer=Orthogonal(),
                       kernel_regularizer=regularizers.l2(L2))(x1)
    x1        = BatchNormalization()(x1)
    x1        = Activation('relu')(x1)


#############   Fully connected layers   ################## 

x        = concatenate([x1,x3])



#   Fully convolutional variant

for feature in (conv_features[0:2]):  
    x        = Conv3D(filters = feature, 
                       kernel_size = (3,1,1), 
                       #kernel_initializer=he_normal(seed=seed),
                       kernel_initializer=Orthogonal(),
                       kernel_regularizer=regularizers.l2(L2))(x)
    x        = BatchNormalization()(x)
    x        = Activation('relu')(x)
    #x        = PReLU()(x)
x        = BatchNormalization()(x)
x        = Dropout(rate = dropout[0])(x)


x        = Conv3D(filters = feature, 
                   kernel_size = (1,3,3), 
                   kernel_initializer=Orthogonal(),
                   kernel_regularizer=regularizers.l2(L2))(x)
x        = Activation('relu')(x)



for fc_filters in fc_features:
	    x        = Conv3D(filters = fc_filters, 
		       kernel_size = (1,1,1), 
		           #kernel_initializer=he_normal(seed=seed),
		       kernel_initializer=Orthogonal(),
		       kernel_regularizer=regularizers.l2(L2))(x)
	    x        = BatchNormalization()(x)        
	    x        = Activation('relu')(x)

coords_x = Input((1,9,9,1))  # or 3,9,9,1 ? 
coords_y = Input((1,9,9,1))
coords_z = Input((1,9,9,1))

x = concatenate([x, coords_x, coords_y, coords_z])

	# Final Softmax Layer
x        = Conv3D(filters = output_classes, 
           kernel_size = (1,1,1), 
           kernel_initializer=Orthogonal(),
           kernel_regularizer=regularizers.l2(L2))(x)
x        = Activation(softmax)(x)


model     = Model(inputs=[mod1,coords_x,coords_y,coords_z], outputs=x)
model.summary()