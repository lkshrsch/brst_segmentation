# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 10:53:33 2018

@author: hirsch
"""


#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 10:26:01 2017

@author: lukas
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
from keras.layers import LeakyReLU
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



class DeepMedic():
    
    def __init__(self, dpatch, output_classes, num_channels, L2, dropout, learning_rate, optimizer_decay, loss_function):
        
        self.dpatch = dpatch
        self.output_classes = output_classes
        self.conv_features_downsample = [30,30,30,30,30,30,30,30,30]
        self.conv_features = [30,30,30,30,50,50,50,50,50,50,100,100,100]# #[50, 50, 50, 50, 50, 100, 100, 100] #[20,20,20,20,30,30,40,40,60,50,50,50,50,50,50] 
        self.fc_features = [150,150,200]
        self.d_factor = 2  # downsampling factor = stride in downsampling pathway
        self.num_channels = num_channels
        self.L2 = L2
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.optimizer_decay = optimizer_decay
        self.loss_function = loss_function
        #self.w_initializer=w_initializer, # initialization of layer parameters? Needed here?
        #self.w_regularizer=w_regularizer,
        #self.b_initializer=b_initializer, # initialization of bias parameters? Needed here?
        #self.b_regularizer=b_regularizer,
        #self.acti_func=acti_func


    
    def createModel(self):
        '''Creates model architecture
        Input: Data input dimensions, eventually architecture specifications parsed from a config file? (activations, costFunction, hyperparameters (nr layers), dropout....)
        Output: Keras Model'''
    
        #seed = 1337
        #mod1 = Input(input_shape=(None,None,None, self.num_channels))
        
        mod1      = Input((self.dpatch[0],self.dpatch[1],self.dpatch[2], self.num_channels))
        
        #############   Downsampled pathway   ##################   
        
        #x2        = AveragePooling3D(pool_size=(d_factor[0],d_factor[1],d_factor[2]), padding="same")(mod1)
        
        x3        = Conv3D(filters = 30, 
                           kernel_size = (3,3,3), 
                           dilation_rate = (1,1,1),
                           padding = 'same',
                           kernel_initializer=Orthogonal(),
                           kernel_regularizer=regularizers.l2(self.L2))(mod1)
        x3        = BatchNormalization()(x3)
        #x3        = Activation('relu')(x3)
        x3        = LeakyReLU()(x3)
        
        x3        = Conv3D(filters = 30, 
                           kernel_size = (3,3,3), 
                           dilation_rate = (1,1,1),
                           padding = 'same',
                           kernel_initializer=Orthogonal(),
                           kernel_regularizer=regularizers.l2(self.L2))(x3)
        x3        = BatchNormalization()(x3)
        #x3        = Activation('relu')(x3)
        x3        = LeakyReLU()(x3)
        # -------------------   Dilation Pyramid
        
        x3        = Conv3D(filters = 40, 
                           kernel_size = (3,3,3), 
                           dilation_rate = (1,2,2),
                           padding = 'valid',
                           #kernel_initializer=he_normal(seed=seed),
                           kernel_initializer=Orthogonal(),
                           kernel_regularizer=regularizers.l2(self.L2))(x3)
        x3        = BatchNormalization()(x3)
        #x3        = Activation('relu')(x3)
	x3        = LeakyReLU()(x3)        

        x3        = Conv3D(filters = 40, 
                           kernel_size = (3,3,3), 
                           dilation_rate = (1,4,4),
                           padding = 'valid',
                           #kernel_initializer=he_normal(seed=seed),
                           kernel_initializer=Orthogonal(),
                           kernel_regularizer=regularizers.l2(self.L2))(x3)
        x3        = BatchNormalization()(x3)
        #x3        = Activation('relu')(x3)
	x3        = LeakyReLU()(x3)       
        
        x3        = Conv3D(filters = 50, 
                           kernel_size = (3,3,3), 
                           dilation_rate = (1,8,8),
                           padding = 'valid',
                           #kernel_initializer=he_normal(seed=seed),
                           kernel_initializer=Orthogonal(),
                           kernel_regularizer=regularizers.l2(self.L2))(x3)
        x3        = BatchNormalization()(x3)
        #x3        = Activation('relu')(x3)
	x3        = LeakyReLU()(x3)       
        
        x3        = Conv3D(filters = 50, 
                           kernel_size = (3,3,3), 
                           dilation_rate = (1,16,16),
                           padding = 'valid',
                           #kernel_initializer=he_normal(seed=seed),
                           kernel_initializer=Orthogonal(),
                           kernel_regularizer=regularizers.l2(self.L2))(x3)
        x3        = BatchNormalization()(x3)
        #x3        = Activation('relu')(x3)
	x3        = LeakyReLU()(x3)       
        
       
        
        #############   High res pathway   ##################  
        
        x1        = Cropping3D(cropping = ((0,0),(22,22),(22,22)), input_shape=(self.dpatch[0],self.dpatch[1],self.dpatch[2],  self.num_channels))(mod1)
        
        for feature in self.conv_features[0:8]:  
            x1        = Conv3D(filters = feature, 
                               kernel_size = (2,3,3), 
                               #kernel_initializer=he_normal(seed=seed),
                               kernel_initializer=Orthogonal(),
                               kernel_regularizer=regularizers.l2(self.L2))(x1)
            x1        = BatchNormalization()(x1)
            #x1        = Activation('relu')(x1)
	    x1        = LeakyReLU()(x1)       
        
        
        #############   Fully connected layers   ################## 
        
        x        = concatenate([x1,x3])
        
        #   Fully convolutional variant
        
        for feature in (self.conv_features[10:12]):  
            x        = Conv3D(filters = feature, 
                               kernel_size = (3,3,3), 
                               #kernel_initializer=he_normal(seed=seed),
                               kernel_initializer=Orthogonal(),
                               kernel_regularizer=regularizers.l2(self.L2))(x)
            x        = BatchNormalization()(x)
            #x        = Activation('relu')(x)
            x        = LeakyReLU()(x)

        #x        = BatchNormalization()(x)
        #x        = Dropout(rate = self.dropout[0])(x)
        
        
        x        = Conv3D(filters = 100, 
                           kernel_size = (1,3,3), 
                           kernel_initializer=Orthogonal(),
                           kernel_regularizer=regularizers.l2(self.L2))(x)
        #x        = Activation('relu')(x)
        x        = LeakyReLU()(x)
        #coords_x = Input((1,9,9,1))  

        coords_y = Input((1,9,9,1))
        coords_z = Input((1,9,9,1))
        
        #x = concatenate([x, coords_y, coords_z])

   
        for fc_filters in self.fc_features:
        	    x        = Conv3D(filters = fc_filters, 
        		       kernel_size = (1,1,1), 
        		           #kernel_initializer=he_normal(seed=seed),
        		       kernel_initializer=Orthogonal(),
        		       kernel_regularizer=regularizers.l2(self.L2))(x)
        	    x        = BatchNormalization()(x)        
        	    #x        = Activation('relu')(x)
        	    x        = LeakyReLU()(x)
 
        
        	# Final Softmax Layer
        x        = Conv3D(filters = self.output_classes, 
                   kernel_size = (1,1,1), 
                   kernel_initializer=Orthogonal(),
                   kernel_regularizer=regularizers.l2(self.L2))(x)
        x        = Activation(softmax)(x)
        
        model     = Model(inputs=[mod1,coords_y,coords_z], outputs=x)
        model.compile(loss=Generalised_dice_coef_multilabel2, optimizer=Adam(lr=self.learning_rate), metrics=[dice_coef_multilabel0,dice_coef_multilabel1])
                                  
        return model
        
