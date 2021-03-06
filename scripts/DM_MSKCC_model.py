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
from keras.layers import LeakyReLU
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



class DeepMedic():
    
    def __init__(self, dpatch, output_classes, num_channels, L2, dropout, learning_rate, optimizer_decay, loss_function):
        
        self.dpatch = dpatch[0]
        self.output_classes = output_classes
	self.conv_features_downsample = [30,30,30,30,30,30,30,30,30]
        self.conv_features = [50,50,50,50,50,50,50,70,70,70,70,100,100]#[20,20,20,20,30,30,40,40,60,50,50,50,50,50,50] #[50, 50, 50, 50, 50, 100, 100, 100]
        self.fc_features = [100,100,100,150]
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
        mod1      = Input((self.dpatch,self.dpatch,self.dpatch, self.num_channels))
        
        #############   High res pathway   ##################  
        x1        = Cropping3D(cropping = ((8,8),(8,8),(8,8)), input_shape=(None,None,None, self.num_channels))(mod1)
        #x1        = Cropping3D(cropping = ((8,8),(8,8),(8,8)), input_shape=(self.dpatch,self.dpatch,self.dpatch, self.num_channels))(mod1)
        
        for feature in self.conv_features[0:8]:  
            x1        = Conv3D(filters = feature, 
                               kernel_size = (3,3,3), 
                               #kernel_initializer=he_normal(seed=seed),
                               kernel_initializer=Orthogonal(),
                               kernel_regularizer=regularizers.l2(self.L2))(x1)
            x1        = BatchNormalization()(x1)
            x1        = Activation('relu')(x1)
            #x1        = LeakyReLU()(x1)
            #x1        = BatchNormalization()(x1)
          
            
            
        #############   Downsampled pathway   ##################   
        x2        = MaxPooling3D(pool_size=(self.d_factor,self.d_factor,self.d_factor), padding="same")(mod1)
        
        x3        = Conv3D(filters = feature, 
                           kernel_size = (3,3,3), 
                           #kernel_initializer=he_normal(seed=seed),
                           kernel_initializer=Orthogonal(),
                           kernel_regularizer=regularizers.l2(self.L2))(mod1)
        x3        = BatchNormalization()(x3)
        x3        = Activation('relu')(x3)
        #x3        = LeakyReLU()(x3)


	for feature in (self.conv_features_downsample[0:9]):  
            x3        = Conv3D(filters = feature, 
                               kernel_size = (3,3,3), 
                               #kernel_initializer=he_normal(seed=seed),
                               kernel_initializer=Orthogonal(),
                               kernel_regularizer=regularizers.l2(self.L2))(x3)
            x3        = BatchNormalization()(x3)
            x3        = Activation('relu')(x3)
            #x3        = LeakyReLU()(x3)


        #x2        = AveragePooling3D(pool_size=(self.d_factor,self.d_factor,self.d_factor), padding="same")(mod1)
        
	x2        = concatenate([x2,x3])

        for feature in (self.conv_features[0:6]):    
            x2        = Conv3D(filters = feature, 
                               kernel_size = (3,3,3), 
                               #kernel_initializer=he_normal(seed=seed),
                               kernel_initializer=Orthogonal(),
                               kernel_regularizer=regularizers.l2(self.L2))(x2)
            x2        = BatchNormalization()(x2)
            x2        = Activation('relu')(x2)
            #x2        = LeakyReLU()(x2)
            #x2        = BatchNormalization()(x2)
        
        #x2        = UpSampling3D(size=(9,9,9))(x2)
        
        #############   Fully connected layers   ################## 
        
        x        = concatenate([x1,x2])
        
        #   Fully convolutional variant
        
        for feature in (self.conv_features[8:10]):  
            x        = Conv3D(filters = feature, 
                               kernel_size = (5,1,1), 
                               #kernel_initializer=he_normal(seed=seed),
                               kernel_initializer=Orthogonal(),
                               kernel_regularizer=regularizers.l2(self.L2))(x)
            #x        = BatchNormalization()(x)
            #x        = Activation('relu')(x)
            x        = LeakyReLU()(x)
        x        = BatchNormalization()(x)
        x        = Dropout(rate = self.dropout[0])(x)
       
        coords = Input((1,9,9,1))

        x = concatenate([x, coords])    
 
	for fc_filters in self.fc_features:
	    x        = Conv3D(filters = fc_filters, 
		       kernel_size = (1,1,1), 
		           #kernel_initializer=he_normal(seed=seed),
		       kernel_initializer=Orthogonal(),
		       kernel_regularizer=regularizers.l2(self.L2))(x)
    	    x        = BatchNormalization()(x)        
	    x        = LeakyReLU()(x)

	# Final Softmax Layer
        x        = Conv3D(filters = self.output_classes, 
                   kernel_size = (1,1,1), 
                   #kernel_initializer=he_normal(seed=seed),
                   kernel_initializer=Orthogonal(),
                   kernel_regularizer=regularizers.l2(self.L2))(x)
        #x        = BatchNormalization()(x)
        x        = Activation(softmax)(x)
        #x        = Dense(units = fc_features[2], activation = 'softmax', name = 'softmax')(x)
        
        model     = Model(inputs=[mod1,coords], outputs=x)
        model.compile(loss=Generalised_dice_coef_multilabel2, optimizer=Adam(lr=self.learning_rate), metrics=[dice_coef_multilabel0,dice_coef_multilabel1])
                                  
        return model
        
        






