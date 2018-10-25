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
    
TPM_channel = ''
    
trainChannels = ['/CV_folds/MRIs_train_post_set0.txt'] #,'/CV_folds/MRIs_train_pre_set0.txt']
trainLabels = '/CV_folds/labels_train_set0.txt'
    
testChannels = ['/CV_folds/MRIs_test_post_set0.txt'] #,'/CV_folds/MRIs_test_pre_set0.txt']
testLabels = '/CV_folds/labels_test_set0.txt'

validationChannels = testChannels
validationLabels = testLabels
    
output_classes = 2
test_subjects = 5
    
  
#-------------------------------------------------------------------------------------------------------------

# Parameters 

######################################### MODEL PARAMETERS
# Models : 'CNN_TPM' , 'DeepMedic'
model = 'DeepMedic'
dpatch=[41,41,41]
L2 = 0.0001
# Loss functions: 'Dice', 'wDice', 'Multinomial'
loss_function = 'Dice2'

load_model = False
path_to_model = '/home/hirsch/Documents/projects/brainSegmentation/DeepPriors/training_sessions/CNN_TPM_fullHeadSegmentation_configFile0_CNN_TPM_Dice_loss_2018-08-13_2311/models/HeadSegmentation_configFile0_CNN_TPM_Dice_loss_2018-08-13_2311.log_epoch5.h5'
session =  path_to_model.split('/')[-3]

num_channels = len(trainChannels)
dropout = [0,0]  # dropout for last two fully connected layers
learning_rate = 2e-05
optimizer_decay = 0

########################################## TRAIN PARAMETERS
num_iter = 10
epochs = 50
samplingMethod_train = 1
samplingMethod_val = 1

n_patches = 7060
n_subjects = 706# Check that this is not larger than subjects in training file
size_minibatches = 2000 # Check that this value is not larger than the ammount of patches per subject per class


quickmode = False # Train without validation. Full segmentation often but only report dice score (whole)
n_patches_val = 1000
n_subjects_val =  50# Check that this is not larger than subjects in validation file
size_minibatches_val = 1000 # Check that this value is not larger than the ammount of patches per subject per class


########################################### TEST PARAMETERS
quick_segmentation = True
n_fullSegmentations = 2
list_subjects_fullSegmentation = []
epochs_for_fullSegmentation = range(1,epochs)
size_test_minibatches = 1500
saveSegmentation = True

threshold_EARLY_STOP = 0.0001

import numpy as np
penalty_MATRIX = np.array([[ 1,  0],
			   [ 0,  1]], dtype='float32')


comments = ''

