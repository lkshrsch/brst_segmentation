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
    
segmentChannels = ['/CV_folds/stroke/MRIs_test_set0.txt']
segmentLabels = '/CV_folds/stroke/labels_test_set0.txt'

output_classes = 6
    
#-------------------------------------------------------------------------------------------------------------

# Parameters 

######################################### MODEL PARAMETERS
# Models : 'CNN_TPM' , 'DeepMedic'
model = 'CNN_TPM'
dpatch=51

path_to_model = '/home/hirsch/Documents/projects/brainSegmentation/DeepPriors/training_sessions/CNN_TPM_fullHeadSegmentation_configFile0_CNN_TPM_Dice_loss_2018-08-13_2311/models/HeadSegmentation_configFile0_CNN_TPM_Dice_loss_2018-08-13_2311.log_epoch5.h5'
session =  path_to_model.split('/')[-3]

########################################### TEST PARAMETERS
quick_segmentation = True
test_subjects = 10
n_fullSegmentations = test_subjects
size_test_minibatches = 500
saveSegmentation = True

import numpy as np
penalty_MATRIX = np.array([[ 1, -1, -1, -1,  0,  0],
                           [-1,  1,  0,  0, -1, -1],
                           [-1,  0,  1,  0, -1, -1],
                           [-1,  0,  0,  1,  0, -1],
                           [ 0, -1, -1,  0,  1,  0],
                           [ 0, -1, -1, -1,  0,  1]], dtype='float32')

penalty_MATRIX[penalty_MATRIX < 0 ] = 0

comments = ''

