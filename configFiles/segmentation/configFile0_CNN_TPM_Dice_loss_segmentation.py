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
dataset = 'ATLAS'

############################## Load dataset #############################
 
TPM_channel = ''

segmentChannels = ['/CV_folds/MRIs_train_post_set0.txt']
segmentLabels = '/CV_folds/labels_train_set0.txt'

output_classes = 2
    
#-------------------------------------------------------------------------------------------------------------

# Parameters 

######################################### MODEL PARAMETERS
# Models : 'CNN_TPM' , 'DeepMedic'
model = 'DeepMedic'
dpatch=41

path_to_model = '/home/hirsch/Documents/projects/Breast_segmentation/DeepPriors_package/training_sessions/DeepMedic_fullHeadSegmentation_configFile_DM_2018-10-17_2053/models/llHeadSegmentation_configFile_DM_2018-10-17_2053.log_epoch7.h5'
session =  path_to_model.split('/')[-3]

########################################### TEST PARAMETERS
quick_segmentation = True
test_subjects = 10
n_fullSegmentations = test_subjects
size_test_minibatches = 1000
saveSegmentation = True

import numpy as np
penalty_MATRIX = np.array([[ 1,  0],
			   [ 0,  1]], dtype='float32')

comments = ''

