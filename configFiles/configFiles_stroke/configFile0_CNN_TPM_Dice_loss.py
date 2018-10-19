#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 10:34:06 2018

@author: lukas
"""

import os
wd = os.getcwd()

###################   parameters // replace with config files ########################


#availabledatasets :'ATLAS17','CustomATLAS', 'BRATS15', 'BRATS15_TEST', 'BRATS15_wholeNormalized' ,BRATS15_ENTIRE', 'CustomBRATS15' (for explicitly giving channels)
dataset = 'fullHeadSegmentation'

############################## Load dataset #############################
 
#TPM_channel = '/home/hirsch/Documents/projects/TPM/correct_labels_TPM_padded.nii'
    
TPM_channel = '/logTPM_padded.nii'
    
trainChannels = ['/CV_folds/stroke/MRIs_train_set0.txt']
trainLabels = '/CV_folds/stroke/labels_train_set0.txt'
    
testChannels = ['/CV_folds/stroke/MRIs_test_set0.txt']
testLabels = '/CV_folds/stroke/labels_test_set0.txt'

validationChannels = testChannels
validationLabels = testLabels
    
output_classes = 6
test_subjects = 7
    
  
#-------------------------------------------------------------------------------------------------------------

# Parameters 

######################################### MODEL PARAMETERS
# Models : 'CNN_TPM' , 'DeepMedic'
model = 'DeepMedic'
dpatch=51
L2 = 0.0001
# Loss functions: 'Dice', 'wDice', 'Multinomial'
loss_function = 'Dice6'

load_model = True
path_to_model = '/home/hirsch/Documents/projects/brainSegmentation/DeepPriors/training_sessions/CNN_TPM_fullHeadSegmentation_configFile0_CNN_TPM_Dice_loss_2018-08-13_2311/models/HeadSegmentation_configFile0_CNN_TPM_Dice_loss_2018-08-13_2311.log_epoch5.h5'
session =  path_to_model.split('/')[-3]

num_channels = len(trainChannels)
dropout = [0,0]  # dropout for last two fully connected layers
learning_rate = 2e-04
optimizer_decay = 0

########################################## TRAIN PARAMETERS
num_iter = 5
epochs = 80
samplingMethod_train = 0

n_patches = 50000
n_subjects = 28# Check that this is not larger than subjects in training file
size_minibatches = 800 # Check that this value is not larger than the ammount of patches per subject per class

quickmode = False # Train without validation. Full segmentation often but only report dice score (whole)
n_patches_val = 500
n_subjects_val =  7# Check that this is not larger than subjects in validation file
size_minibatches_val = 500 # Check that this value is not larger than the ammount of patches per subject per class
samplingMethod_val = 0

########################################### TEST PARAMETERS
quick_segmentation = True
test_subjects = 7
n_fullSegmentations = test_subjects
#list_subjects_fullSegmentation = []
epochs_for_fullSegmentation = range(0,epochs)
size_test_minibatches = 500
saveSegmentation = False

threshold_EARLY_STOP = 0.01

import numpy as np
penalty_MATRIX = np.array([[ 1, -1, -1, -1,  0,  0],
                           [-1,  1,  0,  0, -1, -1],
                           [-1,  0,  1,  0, -1, -1],
                           [-1,  0,  0,  1,  0, -1],
                           [ 0, -1, -1,  0,  1,  0],
                           [ 0, -1, -1, -1,  0,  1]], dtype='float32')

penalty_MATRIX[penalty_MATRIX < 0 ] = 0

comments = ''

