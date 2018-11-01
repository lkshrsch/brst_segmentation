#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 11:35:14 2017

@author: lukas
"""
import os
import sys
import nibabel as nib
import numpy as np
import time
import random
from numpy.random import seed
import keras 
from keras import backend as K
from keras.utils import to_categorical
from tensorflow import set_random_seed
from sklearn import metrics
import matplotlib.pyplot as plt
from shutil import copyfile
from random import shuffle
from random import sample
from keras.callbacks import ModelCheckpoint
from matplotlib.patches import Rectangle
import pandas as pd

seed(1)
set_random_seed(2)


def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def generateRandomIndexesSubjects(n_subjects, total_subjects):
    indexSubjects = random.sample(xrange(total_subjects), n_subjects)
    return indexSubjects

def getSubjectChannels(subjectIndexes, channel):
    "With the channels (any modality) and the indexes of the selected subjects, return the addresses of the subjects channels"
    fp = open(channel)
    # read file, per subject index extract patches given the indexesPatch
    lines = fp.readlines()
    selectedSubjects = [lines[i][:-1] for i in subjectIndexes]
    fp.close()
    return selectedSubjects

def getSubjectShapes(subjectIndexes, n_patches, channelList):
    # Need to open every nifty file and get the shapes
    fp = open(channelList)
    # read file, per subject index extract patches given the indexesPatch
    lines = fp.readlines()
    selectedSubjects = [lines[i] for i in subjectIndexes]
    fp.close()
    shapes = []
    # Get shapes of all subjects to sample from. Can be a separate function (cause apparently I am needing this everywhere)
    for subjectChannel in selectedSubjects:
        subjectChannel = str(subjectChannel)[:-1]
        proxy_img = nib.load(subjectChannel)
        shapes.append(proxy_img.shape)
    return shapes      

def generateVoxelIndexes(subjectIndexes, shapes, patches_per_subject, dpatch, n_patches, groundTruthChannel_list, mriChannel, samplingMethod, output_classes, allForegroundVoxels = ""):
    "Alternative improved function of the same named one above."
    "Here extract the channels from the subject indexes, and loop over them. Then in second loop extract as many needed voxel coordinates per subject."
    methods =["Random sampling","Equal sampling background/foreground","Equal sampling all classes center voxel","Equal sampling background/foreground with exhaustive foreground samples"]
    print("Generating voxel indexes with method: " + methods[samplingMethod])
    channels = getSubjectChannels(subjectIndexes, groundTruthChannel_list)
    channel_mri = getSubjectChannels(subjectIndexes, mriChannel)
    allVoxelIndexes = []    
    
    if samplingMethod == 0:
        for i in xrange(0, len(shapes)):
            voxelIndexesSubj = []
            #loop over voxels per subject
            for j in range(0,patches_per_subject[i]):
                # unform sampling
                # central voxel of final 9x9x9 label cube
                #voxelIndexesSubj.append((np.random.randint(4, shapes[i][0]-5),np.random.randint(4, shapes[i][1]-5),np.random.randint(4, shapes[i][2]-5)))
                voxelIndexesSubj.append((np.random.randint(0+dpatch[0]/2, shapes[i][0]-(dpatch[0]/2)-1),np.random.randint(0+dpatch[1]/2, shapes[i][1]-(dpatch[1]/2)-1),np.random.randint(0+dpatch[2]/2, shapes[i][2]-(dpatch[2]/2)-1)))
            allVoxelIndexes.append(voxelIndexesSubj)            
        random.shuffle(allVoxelIndexes[i])
        return allVoxelIndexes   
        
    elif samplingMethod == 1:
        "This samples equally background/foreground. Assumption that foreground is very seldom: Only foreground voxels are sampled, and background voxels are just random samples which are then proofed against foreground ones"
        "Still need to proof for repetition. Although unlikely and uncommon"
        for i in range(0,len(channels)): 
            print('Generating {} voxel indexes for subject {}/{}'.format(patches_per_subject[i],i,len(shapes)))
            voxelIndexesSubj = []
            backgroundVoxels = [] 
            #sagittal_background_slices = []     
            assert len(channels) == len(shapes)
            #print('\n Getting foreground Voxels from {} \n with shape {}'.format(channels[i], shapes[i]))
            fg = getForegroundBackgroundVoxels(channels[i], dpatch) # This function returns only foreground voxels
            #print('\n Found {} foreground voxels in {}'.format(len(fg),channels[i]))
            if len(fg) > 0:
        		#print("Extracting foreground voxels " + str(patches_per_subject[i]/2 + patches_per_subject[i]%2) +' from ' + str(len(fg)) + " from channel " + str(channels[i]) + " with index " + str(i))
              
                # Half of patches are foreground (standard method)
              #foregroundVoxels = fg[random.sample(xrange(0,len(fg)), min(len(fg),patches_per_subject[i]/2 + patches_per_subject[i]%2))].tolist()       
                # All patches are foreground.
              foregroundVoxels = fg[random.sample(xrange(0,len(fg)), min(len(fg),patches_per_subject[i]))].tolist()       
        		# Images oriented to RAS have tumor on sagittal (first axis)            
        		#create a window 0f -5:+5 around foreground voxels from which no background voxels can be sampled... This is to avoid mislabels on tumor. 
        		#try: sagittal_background_slices = range(foregroundVoxels[0][0] - 5) + range(foregroundVoxels[0][0] + 5, shapes[i][0]-5)
        		#except: 
        		#    print('\n Could not get background slices from tumor slice {}'.format(foregroundVoxels[0][0] ))
        		#    break
            
            elif len(fg) == 0:
              #print('No foreground voxels found! Channel: {}'.format(channels[i]))
              bV = getBodyVoxels(channel_mri[i], dpatch)
              #print('\n Found {} body voxels in {}'.format(len(bV),channel_mri[i]))
              foregroundVoxels = []
              foregroundVoxels = bV[random.sample(xrange(0,len(bV)), min(len(bV),patches_per_subject[i]))].tolist()   
              
              #for j in range(0,patches_per_subject[i]):
               # foregroundVoxels.append((np.random.randint(4, shapes[i][0]-5) , np.random.randint(4, shapes[i][1]-5) , np.random.randint(4, shapes[i][2]-5) ))  
              
              #sagittal_background_slices = range(4, shapes[i][0]-5)
              #for j in range(0,min(len(fg),patches_per_subject[i]/2 + patches_per_subject[i]%2)):



            #for j in range(0,patches_per_subject[i]/2):
            #    backgroundVoxels.append((random.choice(sagittal_background_slices),np.random.randint(4, shapes[i][1]-5),np.random.randint(4, shapes[i][2]-5)))               
            #Replace the ones that by chance are foreground voxels (not so many in tumor data)

            #while any([e for e in foregroundVoxels if e in backgroundVoxels]):
            #    ix = [e for e in foregroundVoxels if e in backgroundVoxels]
            #    for index in ix:
            #        newVoxel = [np.random.randint(4, shapes[i][0]-5),np.random.randint(4, shapes[i][1]-5),np.random.randint(4, shapes[i][2]-5)]
            #        backgroundVoxels[backgroundVoxels.index(index)] = newVoxel
            allVoxelIndexes.append(foregroundVoxels) #+ backgroundVoxels)
            random.shuffle(allVoxelIndexes[i])
        return allVoxelIndexes    
        
    elif samplingMethod == 2:
        "sample from each class equally"        
        "use function getAllForegroundClassesVoxels to get coordinates from all classes (not including background)"
        for i in range(0,len(channels)):  # iteration over subjects
            voxelIndexesSubj = []
            backgroundVoxels = []
            fg = getAllForegroundClassesVoxels(channels[i], dpatch, output_classes) # This function returns only foreground voxels            
            # WATCH OUT, FG IS A LIST OF LISTS. fIRST DIMENSION IS THE CLASS, SECOND IS THE LIST OF VOXELS OF THAT CLASS
            foregroundVoxels = []
            patches_to_sample = [patches_per_subject[i]/output_classes] * output_classes #
            extra = random.sample(range(output_classes),1)
            patches_to_sample[extra[0]] = patches_to_sample[extra[0]] + patches_per_subject[i]%output_classes            
            for c in range(0, output_classes-1):
                foregroundVoxels.extend(fg[c][random.sample(xrange(0,len(fg[c])), min(patches_to_sample[c],len(fg[c])))].tolist())
            # get random voxel coordinates
            for j in range(0,patches_per_subject[i]/output_classes):
                backgroundVoxels.append([np.random.randint(0+dpatch[0]/2, shapes[i][0]-(dpatch[0]/2)-1),np.random.randint(0+dpatch[1]/2, shapes[i][1]-(dpatch[1]/2)-1),np.random.randint(0+dpatch[2]/2, shapes[i][2]-(dpatch[2]/2)-1)])
                #backgroundVoxels.extend([0])
            #Replace the ones that by chance are foreground voxels (not so many in tumor data)
            while any([e for e in foregroundVoxels if e in backgroundVoxels]):
                ix = [e for e in foregroundVoxels if e in backgroundVoxels]
                for index in ix:
                    newVoxel = [np.random.randint(dpatch[0]/2, shapes[i][0]-(dpatch[0]/2)-1),np.random.randint(dpatch[1]/2, shapes[i][1]-(dpatch[1]/2)-1),np.random.randint(dpatch[2]/2, shapes[i][2]-(dpatch[2]/2)-1)]
                    backgroundVoxels[backgroundVoxels.index(index)] = newVoxel
            allVoxelIndexes.append(foregroundVoxels + backgroundVoxels)
            random.shuffle(allVoxelIndexes[i])
        return allVoxelIndexes
    
def getAllForegroundClassesVoxels(groundTruthChannel, dpatch, output_classes):
    '''Get vector of voxel coordinates for all voxel values for all freground classes'''
    "e.g. groundTruthChannel = '/home/hirsch/Documents/projects/ATLASdataset/native_part2/c0011/c0011s0006t01/c0011s0006t01_LesionSmooth_Binary.nii.gz'"
    "NOTE: img in MRICRON starts at (1,1,1) and this function starts at (0,0,0), so points do not match when comparing in MRICRON. Add 1 to all dimensions to match in mricron. Function works properly though"
    img = nib.load(groundTruthChannel)
    data = np.array(img.dataobj[dpatch[0]/2:img.shape[0]-(dpatch[0]/2)-1, dpatch[1]/2:img.shape[1]-(dpatch[1]/2)-1, dpatch[2]/2:img.shape[2]-(dpatch[2]/2)-1],dtype='int16') # Get a cropped image, to avoid CENTRAL foreground voxels that are too near to the border. These will still be included, but not as central voxels. As long as they are in the 9x9x9 volume (-+ 4 voxels from the central, on a segment size of 25x25x25) they will still be included in the training.
    img.uncache()    
    voxels = []
    for c in range(1,output_classes):
        coords = np.argwhere(data==c)
        coords = [sum(x) for x in zip(coords , [x/2 for x in dpatch])]
        voxels.append(coords)
    return voxels  # This is a List! Use totuple() to convert if this makes any trouble
            
def getSubjectsToSample(channelList, subjectIndexes):
    "Actually returns channel of the subjects to sample"
    fp = open(channelList)
    lines = fp.readlines()
    subjects = [lines[i] for i in subjectIndexes]
    fp.close()
    return subjects

def extractLabels(groundTruthChannel_list, subjectIndexes, voxelCoordinates, dpatch):
    print('extracting labels from ' + str(len(subjectIndexes))+ ' subjects.')
    subjects = getSubjectsToSample(groundTruthChannel_list,subjectIndexes)
    labels = []
    #coordinates = []
    if (len(subjectIndexes) > 1):
        for i in xrange(0,len(voxelCoordinates)):
            subject = str(subjects[i])[:-1]
            print('extracting labels from subject index [{}] with path : {}'.format(subjectIndexes[i],subject))
            proxy_label = nib.load(subject)
            label_data = np.array(proxy_label.get_data(),dtype='int8')
            if np.sum(label_data) == 0:
              for j in xrange(0,len(voxelCoordinates[i])):
                labels.append(np.zeros((9,9),dtype='int8'))
            else:
              for j in xrange(0,len(voxelCoordinates[i])):
                D1,D2,D3 = voxelCoordinates[i][j]
                print('Extracting labels from \n subject {} with shape {} and coords {},{},{}'.format(subjects[i], label_data.shape ,D1,D2,D3))
                labels.append(label_data[D1,D2-4:D2+5,D3-4:D3+5])    # changed for breast data with sagittal 2D labels     
                if len(labels[-1])==0:
                  labels[-1] = np.zeros((9,9),dtype='int8')
            proxy_label.uncache()
            del label_data
        return labels
    elif(len(subjectIndexes) == 1):
        subject = str(subjects[0])[:-1]
        proxy_label = nib.load(subject)
        label_data = np.array(proxy_label.get_data(),dtype='int8')
        for i in xrange(0,len(voxelCoordinates[0])):
            if np.sum(label_data) == 0:
              labels.append(np.zeros((9,9),dtype='int8'))
            else:
              D1,D2,D3 = voxelCoordinates[0][i]
              labels.append(label_data[D1,D2-4:D2+5,D3-4:D3+5])  # changed for breast data with sagittal 2D labels
            #print("Extracted labels " + str(i))
        proxy_label.uncache()
        del label_data
        return labels
    
    
def extractCoordinates(channel, subjectIndexes, voxelCoordinates):
    print('extracting coordinates from ' + str(len(subjectIndexes))+ ' subjects.')
    subjects = getSubjectsToSample(channel, subjectIndexes)
    all_coordinates = []
    if (len(subjectIndexes) > 1):
        for i in xrange(0,len(voxelCoordinates)):
            subject = str(subjects[i])[:-1]
            img = nib.load(subject)
            for j in xrange(0,len(voxelCoordinates[i])):     
                D1,D2,D3 = voxelCoordinates[i][j]
                all_coordinates.append(get_Coordinates_from_target_patch(img.shape,D1,D2,D3))                    
            img.uncache()
        return np.array(all_coordinates)
    elif(len(subjectIndexes) == 1):
        subject = str(subjects[0])[:-1]
        img = nib.load(subject)
        for i in xrange(0,len(voxelCoordinates[0])):
            D1,D2,D3 = voxelCoordinates[0][i]
            all_coordinates.append(get_Coordinates_from_target_patch(img.shape,D1,D2,D3))
        img.uncache()
        return np.array(all_coordinates)
       
def get_Coordinates_from_target_patch(img_shape,D1,D2,D3) :
    dpatch = [1,9,9]
    x = []
    y = []
    z = []    
    
    X = [D1]*dpatch[0]*dpatch[1]*dpatch[2]
    x = np.array(X)
    x = x.reshape(dpatch[1],dpatch[2])
    
    Y = range(D2-(dpatch[1]//2),D2+((dpatch[1]//2)+1))
    for i in range(dpatch[1]):
      y.append(Y)
    y = np.transpose(y)
    
    
    Z = range(D3-(dpatch[2]//2),D3+((dpatch[2]//2)+1))
    for i in range(dpatch[2]):
     z.append(Z)
    z = np.array(z)
    
    coords = np.array([x,y,z])
    # Vectorized version:
    #dpatch = [1,9,9]
    #coords = []
    #for i in range(dpatch[1]):
    #    j = i - dpatch[1]//2
        # Get the row by mltiplying the row-coordinate * column-length, then adding the column-coordinate. Loop over row-coordinates to get all rows.
        # Get the correct slice by adding every coordinate + dim2*dim3 times the 'selected slice' = D1.
    #    coords.append((np.array(range((D2+j)*img_shape[2] + D3-(dpatch[1]//2) , (D2+j)*img_shape[2] + (D3+(dpatch[1]//2)+1)),dtype='float64') + float(D1*img_shape[1]*img_shape[2]))*1e-4)
    return np.array(coords)  
    
       
def get_patches_per_subject( n_patches, n_subjects):
    patches_per_subject = [n_patches/n_subjects]*n_subjects
    randomAdd = random.sample(range(0,len(patches_per_subject)),k=n_patches%n_subjects)
    randomAdd.sort()
    for index in randomAdd:
        patches_per_subject[index] = patches_per_subject[index] + 1
    return patches_per_subject

def extractImagePatch(channel, subjectIndexes, patches, voxelCoordinates, dpatch, debug=False):
    subjects = getSubjectsToSample(channel, subjectIndexes)
    n_patches = 0
    badCoords = 0
    myFlag = False
    for i in range(len(voxelCoordinates)):
        n_patches += len(voxelCoordinates[i])
    print('Starting extraction of {} patches from {} subjects.'.format(n_patches,len(voxelCoordinates)))
    vol = np.ones((n_patches,dpatch[0],dpatch[1],dpatch[2]),dtype='float32')
    k = 0
    if (len(subjectIndexes) > 1):
        print('\n MULTIPLE subjects')
        # Loop over subjects
        for i in xrange(0,len(voxelCoordinates)):
            print('Extracting {} image patches for subject with index [{}] and channel {}'.format(len(voxelCoordinates[i]),subjectIndexes[i],str(subjects[i])[:-1]))    
            subject = str(subjects[i])[:-1]
            print('Subject with path: {}'.format(subject))
            proxy_img = nib.load(subject)            
            img_data = np.array(proxy_img.get_data(),dtype='float32')
            # Loop over voxelCoordinates tuples of subject i
            for j in xrange(0,len(voxelCoordinates[i])):   
                #print(voxelCoordinates[i][j] )     
                D1,D2,D3 = voxelCoordinates[i][j]           
                if any([(D1 > img_data.shape[0]) , (D2 > img_data.shape[1]) , (D3 > img_data.shape[2])]):
                    print('Bad Coordinates')
                    badCoords += 1
                    continue
                if (voxelCoordinates[i][j][0] - (dpatch[0]/2) < 0 ) or (voxelCoordinates[i][j][0] + (dpatch[0]/2 + 1) > img_data.shape[0]) : myFlag = True
                if (voxelCoordinates[i][j][1] - (dpatch[1]/2) < 0 ) or (voxelCoordinates[i][j][1] + (dpatch[1]/2 + 1) > img_data.shape[1]) : myFlag = True
                if (voxelCoordinates[i][j][2] - (dpatch[2]/2) < 0 ) or (voxelCoordinates[i][j][2] + (dpatch[2]/2 + 1) > img_data.shape[2]) : myFlag = True
                if myFlag:
                    
                    x_range = [x for x in range(D1-(dpatch[0]/2),D1+(dpatch[0]/2)+1) if (x>=0) and (x < img_data.shape[0]) ]                              
                    y_range = [x for x in range(D2-(dpatch[1]/2),D2+(dpatch[1]/2)+1) if (x>=0) and (x < img_data.shape[1]) ]     
                    z_range = [x for x in range(D3-(dpatch[2]/2),D3+(dpatch[2]/2)+1) if (x>=0) and (x < img_data.shape[2]) ]                   
                    print('From subject: {}'.format(subjects[i][:-1]))
                    print('With voxel coordinates: {}, {}, {}'.format(D1,D2,D3))
                    print('Getting subpatch from {}:{}, {}:{}, {}:{}'.format(x_range[0],x_range[-1],y_range[0],y_range[-1],z_range[0],z_range[-1]))
                    subpatch = img_data[x_range[0]:x_range[len(x_range)-1]+1, y_range[0]:y_range[len(y_range)-1]+1, z_range[0]:z_range[len(z_range)-1]+1]
                    if x_range[0] == 0:
                        start_x = dpatch[0] - len(x_range)    
                    else:
                        start_x = 0
                    end_x = start_x + len(x_range) 
                    if y_range[0] == 0:
                        start_y = dpatch[1] - len(y_range)            
                    else:
                        start_y = 0
                    end_y = start_y + len(y_range) 
                    if z_range[0] == 0:
                        start_z = dpatch[2] - len(z_range)            
                    else:
                        start_z = 0
                    end_z = start_z + len(z_range) 
                    vol[k,:,:,:] = vol[k,:,:,:] * np.min(img_data)
                    vol[k,start_x:end_x,start_y:end_y,start_z:end_z] = subpatch       
                else:
                    vol[k,:,:,:] = img_data[D1-(dpatch[0]/2):D1+(dpatch[0]/2)+1,D2-(dpatch[1]/2):D2+(dpatch[1]/2)+1,D3-(dpatch[2]/2):D3+(dpatch[2]/2)+1]
                k = k+1
            proxy_img.uncache()
            del img_data
            if debug: print('extracted [' + str(len(voxelCoordinates[i])) + '] patches from subject ' + str(i) +'/'+ str(len(subjectIndexes)) +  ' with index [' + str(subjectIndexes[i]) + ']')
        print('\n In this batch found {} Bad Coordinates \n'.format(badCoords))
        return vol

    elif(len(subjectIndexes) == 1):
        #print("\n \n ONLY subject " + str(subjects))
        subject = str(subjects[0])[:-1]
        proxy_img = nib.load(subject)
        img_data = np.array(proxy_img.get_data(),dtype='float32')
        # Loop over the voxel coordinates tuples.
        for i in xrange(0,len(voxelCoordinates[0])):          
            D1,D2,D3 = voxelCoordinates[0][i] 
            #print('Extracting image patch from {}'.format(voxelCoordinates[0][i] ) )
            if (voxelCoordinates[0][i][0] - (dpatch[0]/2) < 0 ) or (voxelCoordinates[0][i][0] + (dpatch[0]/2 + 1) > img_data.shape[0]) : myFlag = True
            if (voxelCoordinates[0][i][1] - (dpatch[1]/2) < 0 ) or (voxelCoordinates[0][i][1] + (dpatch[1]/2 + 1) > img_data.shape[1]) : myFlag = True
            if (voxelCoordinates[0][i][2] - (dpatch[2]/2) < 0 ) or (voxelCoordinates[0][i][2] + (dpatch[2]/2 + 1) > img_data.shape[2]) : myFlag = True
            if myFlag:
                #print('Border voxel')
                x_range = [x for x in range(D1-(dpatch[0]/2),D1+(dpatch[0]/2)+1) if (x>=0) and (x < img_data.shape[0]) ]                              
                y_range = [x for x in range(D2-(dpatch[1]/2),D2+(dpatch[1]/2)+1) if (x>=0) and (x < img_data.shape[1]) ]     
                z_range = [x for x in range(D3-(dpatch[2]/2),D3+(dpatch[2]/2)+1) if (x>=0) and (x < img_data.shape[2]) ] 
                #print('{}, {}, {}'.format(len(x_range),len(y_range),len(z_range)))
                #print('Getting subpatch from \n{}:{},\n{}:{},\n{}:{}'.format(x_range[0],x_range[-1],y_range[0],y_range[-1],z_range[0],z_range[-1]))                 
                subpatch = img_data[x_range[0]:x_range[len(x_range)-1]+1, y_range[0]:y_range[len(y_range)-1]+1, z_range[0]:z_range[len(z_range)-1]+1]		                    
                if x_range[0] == 0:
                    start_x = dpatch[0] - len(x_range)    
                else:
                   start_x = 0
                end_x = start_x + len(x_range) 
                if y_range[0] == 0:
                   start_y = dpatch[1] - len(y_range)            
                else:
                   start_y = 0
                end_y = start_y + len(y_range) 
                if z_range[0] == 0:
                   start_z = dpatch[2] - len(z_range)            
                else:
                   start_z = 0
                end_z = start_z + len(z_range) 
                vol[k,:,:,:] = vol[k,:,:,:] * np.min(img_data)
                vol[k,start_x:end_x,start_y:end_y,start_z:end_z] = subpatch                                                           
            else:
                vol[k,:,:,:] = img_data[D1-(dpatch[0]/2):D1+(dpatch[0]/2)+1,D2-(dpatch[1]/2):D2+(dpatch[1]/2)+1,D3-(dpatch[2]/2):D3+(dpatch[2]/2)+1]
            k = k+1        
            if debug: print('extracted [' + str(i) + '] patches from subject ')
        proxy_img.uncache()
        del img_data
    	return vol

def sampleTrainData(trainChannels, trainLabels, n_patches, n_subjects, dpatch, output_classes, samplingMethod, logfile):
    '''output is a batch containing n-patches and their labels'''
    '''main function, called in the training process'''  
    num_channels = len(trainChannels)
    start = time.time()
    patches_per_subject = get_patches_per_subject( n_patches, n_subjects)    
    labelsFile = open(trainLabels,"r")    
    total_subjects = file_len(trainLabels)
    labelsFile.close()    
    subjectIndexes = generateRandomIndexesSubjects(n_subjects, total_subjects) 
    print('Extracting patches from subjects index: {}'.format(subjectIndexes))
    shapes = getSubjectShapes(subjectIndexes, n_patches, trainChannels[0])
    voxelCoordinates = generateVoxelIndexes(subjectIndexes, shapes, patches_per_subject, dpatch, n_patches, trainLabels, trainChannels[0], samplingMethod, output_classes)    
    
        
    #print('Sampling Training Data: \n With {} patches'.format(len(voxelCoordinates)))
    #print(voxelCoordinates)
    #print(subjectIndexes)
    # Get real number of patches to sample (as counted by the voxelCoordinates extracted, which is <= n_patches, as some classes are sparse)
    real_n_patches = 0
    for i in range(len(voxelCoordinates)):
        real_n_patches += len(voxelCoordinates[i])    
    patches = np.zeros((real_n_patches,dpatch[0],dpatch[1],dpatch[2],num_channels),dtype='float32')    
    for i in xrange(0,len(trainChannels)):
        patches[:,:,:,:,i] = extractImagePatch(trainChannels[i], subjectIndexes, patches, voxelCoordinates, dpatch, debug=False)           
    labels_list = extractLabels(trainLabels, subjectIndexes, voxelCoordinates, dpatch)
    #print(type(labels_list))
    #print(len(labels_list))
    #print(labels_list)
    labels = np.array(labels_list,dtype='int8')
    labels = np.array(to_categorical(labels.astype(int),output_classes),dtype='int8')
    if(samplingMethod == 2):
        patches = patches[0:len(labels)]  # when using equal sampling (samplingMethod 2), because some classes have very few voxels in a head, there are fewer patches as intended. Patches is initialized as the maximamum value, so needs to get cut to match labels.
    all_coordinates = extractCoordinates(trainChannels[0], subjectIndexes, voxelCoordinates)
    end = time.time()
    my_logger("Finished extracting " + str(real_n_patches) + " patches, from "  + str(n_subjects) + " subjects and " + str(num_channels) + " channels. Timing: " + str(round(end-start,2)) + "s", logfile)
    return patches, labels, all_coordinates
    
def generateAllForegroundVoxels(groundTruthChannel_list, dpatch):
    "Gets called once, outside whole training iterations" 
    labelsFile = open(groundTruthChannel_list,"r")    
    total_subjects = file_len(groundTruthChannel_list)
    labelsFile.close()    
    subjectIndexes = range(0,total_subjects)
    channels = getSubjectChannels(subjectIndexes, groundTruthChannel_list)
    allForegroundVoxels = []    
    for index in subjectIndexes:
        fg = list(getForegroundBackgroundVoxels(channels[index], dpatch))
        allForegroundVoxels.append(fg)
    return allForegroundVoxels

def getForegroundBackgroundVoxels(groundTruthChannel, dpatch):
    '''Get vector of voxel coordinates for all voxel values > 0'''
    "e.g. groundTruthChannel = '/home/hirsch/Documents/projects/ATLASdataset/native_part2/c0011/c0011s0006t01/c0011s0006t01_LesionSmooth_Binary.nii.gz'"
    "NOTE: img in MRICRON starts at (1,1,1) and this function starts at (0,0,0), so points do not match when comparing in MRICRON. Add 1 to all dimensions to match in mricron. Function works properly though"
    img = nib.load(groundTruthChannel)
    #data = np.array(img.dataobj[4:img.shape[0]-5,4:img.shape[1]-5,4:img.shape[2]-5],dtype='int16') # Get a cropped image, to avoid CENTRAL foreground voxels that are too near to the border. These will still be included, but not as central voxels. As long as they are in the 9x9x9 volume (-+ 4 voxels from the central, on a segment size of 25x25x25) they will still be included in the training.
    data = img.get_data()    
    img.uncache()    
    foregroundVoxels = np.argwhere(data>0)
    #foregroundVoxels = foregroundVoxels  # need to add this, as the cropped image starts again at (0,0,0)
    #backgroundVoxels = np.argwhere(data==0)
    return foregroundVoxels#, backgroundVoxels  # This is a List! Use totuple() to convert if this makes any trouble
      
def getBodyVoxels(channel, dpatch):
    '''Get vector of voxel coordinates for all voxel values > 0'''
    "e.g. groundTruthChannel = '/home/hirsch/Documents/projects/ATLASdataset/native_part2/c0011/c0011s0006t01/c0011s0006t01_LesionSmooth_Binary.nii.gz'"
    "NOTE: img in MRICRON starts at (1,1,1) and this function starts at (0,0,0), so points do not match when comparing in MRICRON. Add 1 to all dimensions to match in mricron. Function works properly though"
    img = nib.load(channel)
    #data = np.array(img.dataobj[4:img.shape[0]-5,4:img.shape[1]-5,4:img.shape[2]-5],dtype='int16') # Get a cropped image, to avoid CENTRAL foreground voxels that are too near to the border. These will still be included, but not as central voxels. As long as they are in the 9x9x9 volume (-+ 4 voxels from the central, on a segment size of 25x25x25) they will still be included in the training.
    data = img.get_data()    
    img.uncache()    
    bodyVoxels = np.argwhere(data>-0.5)
    #foregroundVoxels = foregroundVoxels  # need to add this, as the cropped image starts again at (0,0,0)
    #backgroundVoxels = np.argwhere(data==0)
    return bodyVoxels#, backgroundVoxels  # This is a List! Use totuple() to convert if this makes any trouble      
      
      
def totuple(a):
    "Returns tuple with tuples"
    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a

def totuple2(a):
    "Returns list with tuples"
    try:
        return list(totuple(i) for i in a)
    except TypeError:
        return a


def f1(y_true, y_pred):
    "Only makes sense in multiclass problems somehow. When binary classes, have to change average = 'binary' somewhere somehow "
    "see https://stackoverflow.com/questions/43001014/precision-recall-fscore-support-returns-same-values-for-accuracy-precision-and"
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall))

def dice(y_true, y_pred):
    # Symbolically compute the intersection
    y_int = y_true*y_pred
    # Technically this is the negative of the Sorensen-Dice index. This is done for
    # minimization purposes
    return (2*K.sum(y_int) / (K.sum(y_true) + K.sum(y_pred)))

def ytrue(y_true, y_pred):
    return(y_true)

def ypred(y_true, y_pred):
    return(y_pred)
    

def fbeta(y_true, y_pred, threshold_shift=0):
    beta = 2

    # just in case of hipster activation at the final layer
    y_pred = K.clip(y_pred, 0, 1)

    # shifting the prediction threshold from .5 if needed
    y_pred_bin = K.round(y_pred + threshold_shift)

    tp = K.sum(K.round(y_true * y_pred_bin), axis=1) + K.epsilon()
    fp = K.sum(K.round(K.clip(y_pred_bin - y_true, 0, 1)), axis=1)
    fn = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)), axis=1)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    beta_squared = beta ** 2
    return K.mean((beta_squared + 1) * (precision * recall) / (beta_squared * precision + recall + K.epsilon()))

def evaluation_metrics(class_pred, prediction, output_classes, miniTestbatch_labels ):

    # Add classes to fullfill requisites for F1 
    #tmp = list(class_pred[-1][-1][-1][0:output_classes]) # store original values
    #class_pred[-1][-1][-1][0:output_classes] = [u for u in range(output_classes)]
    newlist = [u for u in class_pred for u in u]
    newlist = [u for u in newlist for u in u]
    newlist = [u for u in newlist for u in u]
    newlist = np.array(newlist)
    y_pred = newlist
    
    ytrueALL = (np.argmax(miniTestbatch_labels, axis=4))
    newlist = [u for u in ytrueALL for u in u]
    newlist = [u for u in newlist for u in u]
    newlist = [u for u in newlist for u in u]
    newlist = np.array(newlist)
    y_true = newlist
    
    P,N,TP,TN,FP,FN,ACC = stats(y_pred, y_true, output_classes)
    # if normalize = True, same as hamming_score. Just a summarized accuracy for all labels.
    acc = metrics.accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)
    
    # add one class to be able to compute AUC ROC
    #y_true[-output_classes:len(y_true)] = [u for u in range(output_classes)] # replace with classes.
    newlist = [u for u in prediction for u in u]
    newlist = [u for u in newlist for u in u]
    newlist = [u for u in newlist for u in u]
    newlist = np.array(newlist)
    y_score = newlist
    y_true = to_categorical(y_true.astype(int),output_classes)
    "roc_auc_score needs inputs as arrays of shape (n_samples , n_classes), y_score is the output probs of the softmax layer, and both y_true and y_score are one-hot-encoded"
    "There need to be at least one present class in the sample. Not defined if a class is never present in the sample"
    try:
        roc = metrics.roc_auc_score(y_true, y_score, average=None, sample_weight=None)
    except ValueError:
        roc = np.array([np.nan]*(output_classes))
    #coverage = metrics.coverage_error(y_true, y_score, sample_weight=None)
    #label_ranking_loss = metrics.label_ranking_loss(y_true, y_score, sample_weight=None)
    
    # restore to original values, to avoid weird grid patterns in segmentation output. As this alterns globally the outout.
    #class_pred[-1][-1][-1][0:output_classes] = tmp
    
    return P,N,TP,TN,FP,FN,ACC,acc,roc#, coverage, label_ranking_loss


def stats(y_pred, y_true, output_classes):
    P = []
    N = []
    TP = []
    TN = []
    FP = []
    FN = []
    TPR = []
    SPC = []
    DSC = []
    ACC = []

    for c in range(output_classes):
        tmp_pred = y_pred == c
        tmp_true = y_true == c
        p = sum(tmp_true)
        n = sum(np.invert(tmp_true))
        tp = sum(tmp_pred * tmp_true)
        tn = sum(np.invert(tmp_pred)*np.invert(tmp_true))
        fp = sum(tmp_pred * np.invert(tmp_true))
        fn = sum(np.invert(tmp_pred) * tmp_true)
        a = sum(tmp_true == tmp_pred)/float(len(tmp_pred))

        P.append(p)
        N.append(n)
        TP.append(tp)
        TN.append(tn)
        FP.append(fp)
        FN.append(fn)
        #TPR.append(tpr)
        #SPC.append(spc)
        #DSC.append(dsc)
        ACC.append(a)
        
        # Foreground
    tmp_pred = y_pred > 0
    tmp_true = y_true > 0
    p = sum(tmp_true)
    n = sum(np.invert(tmp_true))
    tp = sum(tmp_pred * tmp_true)
    tn = sum(np.invert(tmp_pred)*np.invert(tmp_true))
    fp = sum(tmp_pred * np.invert(tmp_true))
    fn = sum(np.invert(tmp_pred) * tmp_true)
    a = sum(tmp_true == tmp_pred)/float(len(tmp_pred))

    P.append(p)
    N.append(n)
    TP.append(tp)
    TN.append(tn)
    FP.append(fp)
    FN.append(fn)
    #TPR.append(tpr)
    #SPC.append(spc)
    #DSC.append(dsc)
    ACC.append(a)
    
    return P,N,TP,TN,FP,FN,ACC#TPR,SPC,DSC,ACC
    

def hamming_score(y_true, y_pred, normalize=True, sample_weight=None):
    '''
    Compute the Hamming score (a.k.a. label-based accuracy) for the multi-label case
    https://stackoverflow.com/q/32239577/395857
    '''
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set( np.where(y_true[i])[0] )
        set_pred = set( np.where(y_pred[i])[0] )
        #print('\nset_true: {0}'.format(set_true))
        #print('set_pred: {0}'.format(set_pred))
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred))/\
                    float( len(set_true.union(set_pred)) )
        #print('tmp_a: {0}'.format(tmp_a))
        acc_list.append(tmp_a)
    return np.mean(acc_list)


def my_logger(string, logfile):
    f = open(logfile,'a')
    f.write('\n' + str(string))
    f.close()
    print(string)
    
def movingAverageConv(a, window_size=1) :
    if not a : return a
    window = np.ones(int(window_size))
    result = np.convolve(a, window, 'full')[ : len(a)] # Convolve full returns array of shape ( M + N - 1 ).
    slotsWithIncompleteConvolution = min(len(a), window_size-1)
    result[slotsWithIncompleteConvolution:] = result[slotsWithIncompleteConvolution:]/float(window_size)
    if slotsWithIncompleteConvolution > 1 :
        divisorArr = np.asarray(range(1, slotsWithIncompleteConvolution+1, 1), dtype=float)
        result[ : slotsWithIncompleteConvolution] = result[ : slotsWithIncompleteConvolution] / divisorArr
    return result
        
        
def dice_completeImages(img1,img2):
    return(2*np.sum(np.multiply(img1>0,img2>0))/float(np.sum(img1>0)+np.sum(img2>0)))

def normalizeMRI(data):
    #img = nib.load('/home/hirsch/Documents/projects/public_SegmentationData/BRATS2015_Training/HGG/brats_2013_pat0001_1/VSD.Brain.XX.O.MR_Flair.54512/VSD.Brain.XX.O.MR_Flair.54512.nii')
    #data = img.get_data()
    data1 = np.ma.masked_array(data, data==0)
    m = data1.mean()
    s = data1.std()
    data1 = (data1 - m)/s
    data1 = np.ma.getdata(data1)
    return(data1)
    
   
def sampleTestData(testChannels, testLabels, subjectIndex, output_classes, dpatch,logfile):
    "For usage with fullheadsegmentation()"
    "output should be a batch containing all (non-overlapping) image patches of the whole head, and the labels"
    "Actually something like sampleTraindata, thereby inputting extractImagePatch with all voxels of a subject"
    "Voxel coordinates start at index [26] and then increase by 17 in all dimensions."    
    
    if len(testLabels) == 0:  # No GT

        labelsFile = open(testChannels[0],"r")   
        ch = labelsFile.readlines()
        subjectGTchannel = ch[subjectIndex[0]][:-1]
        my_logger('Segmenting subject with channel: ' + str(subjectGTchannel), logfile)
        labelsFile.close()      
        proxy_img = nib.load(subjectGTchannel)
        shape = proxy_img.shape
        affine = proxy_img.affine
            

	# make borders larger, so it takes less to segment, and also because we know there are no interesting things on borders...
        xend = shape[0]-5 # earlier : 5
        yend = shape[1]-10 # earlier : 5
        zend = shape[2]-10 # earlier : 5
    
        voxelCoordinates = []
        for x in range(4,xend):        
            for y in range(8,yend,9):   # earlier 4
                for z in range(8,zend,9):  # earlier 4
                    voxelCoordinates.append([x,y,z])
        labels = []
        all_coordinates = extractCoordinates(testLabels, subjectIndex, [voxelCoordinates])
        #print("Finished extracting " + str(n_patches) + " patches, from "  + str(n_subjects) + " subjects and " + str(num_channels) + " channels. Timing: " + str(round(end-start,2)) + "s")
        return labels, voxelCoordinates, all_coordinates, shape, affine
        
    else:

        labelsFile = open(testChannels[0],"r")   
        ch = labelsFile.readlines()
        subjectGTchannel = ch[subjectIndex[0]][:-1]
        my_logger('Segmenting subject with channel: ' + str(subjectGTchannel), logfile)
        labelsFile.close()      
        proxy_img = nib.load(subjectGTchannel)
        shape = proxy_img.shape
        affine = proxy_img.affine
            
        xend = shape[0]-5
        yend = shape[1]-10
        zend = shape[2]-10
    
        voxelCoordinates = []
        for x in range(4,xend):
            for y in range(8,yend,9):
                for z in range(8,zend,9):
                    voxelCoordinates.append([x,y,z])        
        labels = np.array(extractLabels(testLabels, subjectIndex, [voxelCoordinates], dpatch))
        all_coordinates = extractCoordinates(testLabels, subjectIndex, [voxelCoordinates])
        labels = to_categorical(labels.astype(int),output_classes)
        #print("Finished extracting " + str(n_patches) + " patches, from "  + str(n_subjects) + " subjects and " + str(num_channels) + " channels. Timing: " + str(round(end-start,2)) + "s")
        return labels, voxelCoordinates, all_coordinates, shape, affine
        
def generalized_dice_completeImages(img1,img2):
    assert img1.shape == img2.shape, 'Images of different size!'
    #assert (np.unique(img1) == np.unique(img2)).all(), 'Images have different classes!'
    classes = np.array(np.unique(img1), dtype='int8')   
    dice = []
    for i in classes:
        dice.append(2*np.sum(np.multiply(img1==i,img2==i))/float(np.sum(img1==i)+np.sum(img2==i)))   
    return np.sum(dice)/len(classes), [round(x,2) for x in dice]
    

def fullHeadSegmentation(wd, penalty_MATRIX, dice_compare, dsc, model, testChannels, testLabels, subjectIndex, output_classes, dpatch, size_minibatches,logfile, epoch, saveSegmentation = False, full_evaluation = False):    
    
    subjectIndex = [subjectIndex]
    flairCh = getSubjectsToSample(testChannels[0], subjectIndex)
    subID = flairCh[0].split('.')[0].split('/')[-1]

    # Open subject MRI to extract patches dynamically
    num_channels = len(testChannels)
    labelsFile = open(testChannels[0],"r")   
    ch = labelsFile.readlines()
    subjectGTchannel = ch[subjectIndex[0]][:-1]
    my_logger('Segmenting subject with channel: ' + str(subjectGTchannel), logfile)
    labelsFile.close()      
    proxy_img = nib.load(subjectGTchannel)
    shape = proxy_img.shape
    affine = proxy_img.affine
    
    labels, voxelCoordinates, all_coordinates, shape, affine = sampleTestData(testChannels, testLabels, subjectIndex, output_classes, dpatch,logfile)
    print("Extracted image patches for full head segmentation")
        
    if full_evaluation == False:
        
        start = 0
        n_minibatches = len(labels)/size_minibatches
        indexes = []
        for j in range(0,n_minibatches):
            print("Segmenting minibatch " +str(j)+ "/" + str(n_minibatches))
            end = start + size_minibatches
            # get indexes of already openend MRI, get patch in real time, use for segmentation, then delete.
            # Only store prediction, which is int8
            patches = np.zeros((size_minibatches,dpatch[0],dpatch[1],dpatch[2],num_channels),dtype='float32')
            minibatch_voxelCoordinates = voxelCoordinates[start:end]
            for i in xrange(0,len(testChannels)):
                patches[:,:,:,:,i] = extractImagePatch(testChannels[i], subjectIndex, patches, [minibatch_voxelCoordinates], dpatch, debug=False)
            minibatch_coords = all_coordinates[start:end,:,:]
            minibatch_coords = np.reshape(minibatch_coords,(minibatch_coords.shape[0],3,9,9,1))
            #minibatch_coords_X = minibatch_coords[:,0,:,:,:]
            minibatch_coords_Y = minibatch_coords[:,1,:,:,:]
            minibatch_coords_Z = minibatch_coords[:,2,:,:,:]
            #minibatch_coords_X = np.reshape(minibatch_coords_X,(minibatch_coords_X.shape[0],1,9,9,1))
            minibatch_coords_Y = np.reshape(minibatch_coords_Y,(minibatch_coords_Y.shape[0],1,9,9,1))
            minibatch_coords_Z = np.reshape(minibatch_coords_Z,(minibatch_coords_Z.shape[0],1,9,9,1))
	    #prediction = model.predict(patches, verbose=0)
            prediction = model.predict([patches,minibatch_coords_Y,minibatch_coords_Z], verbose=0)
            #print(prediction[:,:,:,:,1])
            class_pred = prediction[:,:,:,:,1]            # Just output the probability maps for Tumor.
            #class_pred = np.argmax(prediction, axis=4)
            indexes.extend(class_pred)        
            start = end
            
        #last one
        size_last_minibatch = (len(voxelCoordinates)-n_minibatches*size_minibatches)
        end = start + size_last_minibatch
        patches = np.zeros((size_last_minibatch,dpatch[0],dpatch[1],dpatch[2],num_channels),dtype='float32')
        minibatch_voxelCoordinates = voxelCoordinates[start:end]
        for i in xrange(0,len(testChannels)):
            patches[:,:,:,:,i] = extractImagePatch(testChannels[i], subjectIndex, patches, [minibatch_voxelCoordinates], dpatch, debug=False)
        minibatch_coords = all_coordinates[start:end,:,:]            
        minibatch_coords = np.reshape(minibatch_coords,(minibatch_coords.shape[0],3,9,9,1))
        #minibatch_coords_X = minibatch_coords[:,0,:,:,:]
        minibatch_coords_Y = minibatch_coords[:,1,:,:,:]
        minibatch_coords_Z = minibatch_coords[:,2,:,:,:]
        #minibatch_coords_X = np.reshape(minibatch_coords_X,(minibatch_coords_X.shape[0],1,9,9,1))
        minibatch_coords_Y = np.reshape(minibatch_coords_Y,(minibatch_coords_Y.shape[0],1,9,9,1))
        minibatch_coords_Z = np.reshape(minibatch_coords_Z,(minibatch_coords_Z.shape[0],1,9,9,1))
	#prediction = model.predict(patches, verbose=0)   # WHEN LOADING A MODEL THAT IN CODE RECIEVES MULTIPLE INPUTS BUT IN PRACTICE ONLY ONE, IT WILL ONLY ACCEPT ONE INPUT.
        prediction = model.predict([patches,minibatch_coords_Y,minibatch_coords_Z], verbose=0)
        class_pred = prediction[:,:,:,:,1]  
        #class_pred = np.argmax(prediction, axis=4)
        indexes.extend(class_pred)     
        del patches       

        #test_performance.append(model.evaluate(miniTestbatch, miniTestbatch_labels, verbose=0))
        head = np.zeros(shape, dtype='float32')  # same size as input head, start index for segmentation start at 26,26,26, rest filled with zeros....
        i = 0
        for x,y,z in voxelCoordinates:
            head[x,y-4:y+5,z-4:z+5] = indexes[i]
            i = i+1
        img = nib.Nifti1Image(head, affine)
            
        if dice_compare:           
            labelsFile = open(testLabels,"r")   
            ch = labelsFile.readlines()
            subjectGTchannel = ch[subjectIndex[0]][:-1]
            GT = nib.load(subjectGTchannel)
            if img.shape != GT.shape:
              print('Images of different size! \n{}, \n{}'.format(subjectGTchannel,testChannels[0]))
	      #print(np.sum(GT.get_data()))
              if np.sum(GT.get_data()) == 0:
		#print('creating dummy ground truth')
                GT = np.zeros(shape=img.shape)
                score = weighted_generalized_dice_completeImages(GT, img.get_data(), penalty_MATRIX)
                #score = generalized_dice_completeImages(img.get_data(), GT.get_data())
                dsc.append(score[0])
                print(dsc[-1])
                print('per class dice score: {}'.format(score[1]))
                print('mean DCS so far:' + str(np.mean(dsc)))
            else:  
              score = weighted_generalized_dice_completeImages(GT.get_data(), img.get_data(), penalty_MATRIX)
              #score = generalized_dice_completeImages(img.get_data(), GT.get_data())
              dsc.append(score[0])
              print(dsc[-1])
              print('per class dice score: {}'.format(score[1]))
              print('mean DCS so far:' + str(np.mean(dsc)))
            
        if(saveSegmentation):
            segmentationName = '/predictions/' + subID + str(epoch)
            output = wd +'/' + segmentationName + '.nii.gz'
            nib.save(img, output)
            my_logger('Saved segmentation of subject at: ' + output, logfile)

    else:
       
        positives = []
        negatives = []
        truePositives = []
        trueNegatives = []
        falsePositives = []
        falseNegatives = []
        accuracy = []  # per class
        total_accuracy = []  # as a whole
        auc_roc = []
        
        start = 0
        n_minibatches = len(labels)/size_minibatches
        indexes = []
        for j in range(0,n_minibatches):
            print("Segmenting patch " +str(j)+ "/" + str(n_minibatches))
            end = start + size_minibatches
            patches = np.zeros((size_minibatches,dpatch[0],dpatch[1],dpatch[2],num_channels),dtype='float32')
            minibatch_voxelCoordinates = voxelCoordinates[start:end]
            for i in xrange(0,len(testChannels)):
                patches[:,:,:,:,i] = extractImagePatch(testChannels[i], subjectIndex, patches, [minibatch_voxelCoordinates], dpatch, debug=False)
                        
            miniTestbatch_labels = labels[start:end,:,:,:,:]    
            prediction = model.predict([patches], verbose=0) 
            class_pred = np.argmax(prediction, axis=4)
            indexes.extend(class_pred)        
            #test_performance.append(model.evaluate(miniTestbatch, miniTestbatch_labels, verbose=0))
            P,N,TP,TN,FP,FN,ACC,acc,roc =  evaluation_metrics(class_pred, prediction, output_classes, miniTestbatch_labels )
            
            positives.append(P)
            negatives.append(N)
            truePositives.append(TP)
            trueNegatives.append(TN)
            falsePositives.append(FP)
            falseNegatives.append(FN)
            accuracy.append(ACC)
            total_accuracy.append(acc)
            auc_roc.append(roc)
            start = end
            
        
        #last one
        size_last_minibatch = (len(voxelCoordinates)-n_minibatches*size_minibatches)
        end = start + size_last_minibatch
        patches = np.zeros((size_last_minibatch,dpatch[0],dpatch[1],dpatch[2],num_channels),dtype='float32')
        minibatch_voxelCoordinates = voxelCoordinates[start:end]
        for i in xrange(0,len(testChannels)):
            patches[:,:,:,:,i] = extractImagePatch(testChannels[i], subjectIndex, patches, [minibatch_voxelCoordinates], dpatch, debug=False)
                           
        miniTestbatch_labels = labels[start:end,:,:,:,:]    
        prediction = model.predict([patches], verbose=0) 
        #prediction = model.predict(miniTestbatch, verbose=0)
        class_pred = np.argmax(prediction, axis=4)
        indexes.extend(class_pred)            
        #test_performance.append(model.evaluate(miniTestbatch, miniTestbatch_labels, verbose=0))
        P,N,TP,TN,FP,FN,ACC,acc,roc =  evaluation_metrics(class_pred, prediction, output_classes, miniTestbatch_labels )
        positives.append(P)
        negatives.append(N)
        truePositives.append(TP)
        trueNegatives.append(TN)
        falsePositives.append(FP)
        falseNegatives.append(FN)
        #sens.append(TPR)
        #spec.append(SPC)
        #Dice.append(DSC)
        accuracy.append(ACC)  # per class
        total_accuracy.append(acc)
        auc_roc.append(roc)
    
        mean_acc = np.average(accuracy, axis=0)
        mean_total_accuracy = np.average(total_accuracy, axis=0)
        mean_AUC_ROC = np.nanmean(auc_roc, axis=0)
        
        
        sumTP = np.sum(truePositives,0)
        sumTN = np.sum(trueNegatives,0)
        sumP = np.sum(positives,0)
        sumN = np.sum(negatives,0)
        sumFP = np.sum(falsePositives,0)
        sumFN = np.sum(falseNegatives,0)    
    
        total_sens = np.divide(np.array(sumTP,dtype='float32'),np.array(sumP,dtype='float32'))
        total_spec = np.divide(np.array(sumTN,dtype='float32'),np.array(sumN,dtype='float32'))
        total_precision = np.divide(np.array(sumTP,dtype='float32'),(np.array(sumTP,dtype='float32') + np.array(sumFP,dtype='float32')))
        total_DSC = np.divide(2*np.array(sumTP,dtype='float64'),(2 * np.array(sumTP,dtype='float64') + np.array(sumFP,dtype='float64') + np.array(sumFN,dtype='float64')))
        
        if(saveSegmentation):
        
            head = np.zeros(shape, dtype=np.int16)  # same size as input head, start index for segmentation start at 26,26,26, rest filled with zeros....
            i = 0
            for x,y,z in voxelCoordinates:
                
                head[x-4:x+5,y-4:y+5,z-4:z+5] = indexes[i]
                i = i+1
    
            img = nib.Nifti1Image(head, affine)
            
            segmentationName =  '/predictions/' + subID + str(epoch)
            output = wd  +'/'+ segmentationName + '.nii.gz'
            nib.save(img, output)
            my_logger('Saved segmentation of subject at: ' + output, logfile)
            
        return total_sens, total_spec, total_DSC, mean_acc, mean_total_accuracy, mean_AUC_ROC, total_precision

def start_training_session_logger(logfile,threshold_EARLY_STOP, TPM_channel, load_model,saveSegmentation,path_to_model,model,dropout, trainChannels, trainLabels, validationChannels, validationLabels, testChannels, testLabels, num_iter, epochs, n_patches, n_patches_val, n_subjects, samplingMethod_train, size_minibatches, n_fullSegmentations, epochs_for_fullSegmentation, size_test_minibatches):
    my_logger('#######################################  NEW TRAINING SESSION  #######################################', logfile)    
    my_logger(trainChannels, logfile)
    my_logger(trainLabels, logfile)
    my_logger(validationChannels, logfile)        
    my_logger(validationLabels, logfile)  
    my_logger(testChannels, logfile) 
    my_logger(testLabels, logfile)
    my_logger('TPM channel (if given):', logfile)
    my_logger(TPM_channel, logfile)
    my_logger('Session parameters: ', logfile)
    my_logger('[num_iter, epochs, n_patches, n_patches_val, n_subjects, samplingMethod_train, size_minibatches, n_fullSegmentations, epochs_for_fullSegmentation, size_test_minibatches]', logfile)
    my_logger([num_iter, epochs, n_patches, n_patches_val, n_subjects, samplingMethod_train, size_minibatches, n_fullSegmentations, epochs_for_fullSegmentation, size_test_minibatches], logfile)
    my_logger('Dropout for last two fully connected layers: ' + str(dropout), logfile)
    my_logger('Model loss function: ' + str(model.loss), logfile)
    my_logger('Model number of parameters: ' + str(model.count_params()), logfile)
    my_logger('Optimizer used: ' +  str(model.optimizer.from_config), logfile)
    my_logger('Optimizer parameters: ' + str(model.optimizer.get_config()), logfile)
    my_logger('Save full head segmentation of subjects: ' + str(saveSegmentation), logfile)
    my_logger('EARLY STOP Threshold last 3 epochs: ' + str(threshold_EARLY_STOP), logfile)
    if load_model:
        my_logger("USING PREVIOUSLY SAVED MODEL -  Model retrieved from: " + path_to_model, logfile)


class LossHistory_multiDice6(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.dice = []
        self.metrics = []

    def on_batch_end(self, batch, logs={}):
        self.dice = []
        self.losses.append(logs.get('loss'))
        self.dice.append(logs.get('dice_coef_multilabel0'))
        self.dice.append(logs.get('dice_coef_multilabel1'))
        self.dice.append(logs.get('dice_coef_multilabel2'))
        self.dice.append(logs.get('dice_coef_multilabel3'))
        self.dice.append(logs.get('dice_coef_multilabel4'))
        self.dice.append(logs.get('dice_coef_multilabel5'))
        self.metrics.append(self.dice)



class LossHistory_multiDice2(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.dice = []
        self.metrics = []

    def on_batch_end(self, batch, logs={}):
        self.dice = []
        self.losses.append(logs.get('loss'))
        self.dice.append(logs.get('dice_coef_multilabel0'))
        self.dice.append(logs.get('dice_coef_multilabel1'))
        self.metrics.append(self.dice)


def classesInSample(minibatch_labels, output_classes):
	label_numbers = []
	print(minibatch_labels.shape)	
	minibatch_labels = np.argmax(minibatch_labels, axis=3)
	for c in range(output_classes):
		label_numbers.append(np.sum(minibatch_labels == c))
	#return label_numbers
	return np.sum(minibatch_labels, axis=4)

def train_model_on_batch(model,batch,labels,coords,size_minibatches,history,losses,metrics,l,logfile, output_classes):
    start = 0
    n_minibatches = len(batch)/size_minibatches
    for j in range(0,n_minibatches):
        print("training on minibatch " +str(j+1)+ "/" + str(n_minibatches))
        end = start + size_minibatches
        minibatch = batch[start:end,:,:,:,:]    
        minibatch_labels = labels[start:end,:,:,:]   
        minibatch_labels = np.reshape(minibatch_labels,(minibatch_labels.shape[0],1,9,9,2))
        minibatch_coords = coords[start:end,:,:,:]
        minibatch_coords = np.reshape(minibatch_coords,(minibatch_coords.shape[0],3,9,9,1))
        minibatch_coords_X = minibatch_coords[:,0,:,:,:]
        minibatch_coords_Y = minibatch_coords[:,1,:,:,:]
        minibatch_coords_Z = minibatch_coords[:,2,:,:,:]
        #minibatch_coords_X = np.reshape(minibatch_coords_X,(minibatch_coords_X.shape[0],1,9,9,1))
        minibatch_coords_Y = np.reshape(minibatch_coords_X,(minibatch_coords_Y.shape[0],1,9,9,1))
        minibatch_coords_Z = np.reshape(minibatch_coords_X,(minibatch_coords_Z.shape[0],1,9,9,1))
        print("Sampled following number of classes in training minibatch: {}".format(np.sum(np.sum(np.sum(np.sum(minibatch_labels, axis=3),axis=2),axis=1),axis=0)))
        #freq = classesInSample(minibatch_labels, output_classes)
        #my_logger("Sampled following number of classes in training MINIBATCH: " + str(freq), logfile)
        #print(minibatch_labels.shape)
        #print(minibatch_labels)
        #print(getClassProportions(freq))
        model.fit([minibatch,minibatch_coords_Y,minibatch_coords_Z], minibatch_labels,  verbose = 0, callbacks = [history])
        losses.extend(history.losses)
        metrics.extend(history.metrics)
        start = end
        my_logger('Train cost and metrics     ' + str(losses[-1]) + ' ' + str(metrics[-1]),logfile)
    del batch
    del labels

    
def plot_training(session,losses, metrics,val_performance,full_segm_DICE, smooth=50, loss_name = ['Multiclass Dice'], class_names = ['Air','GM','WM','CSF','Bone','Skin']):

    losses_df = pd.DataFrame(losses)
    losses_df.columns=loss_name
    
    losses_mv_avg = losses_df.rolling(smooth,center=False).mean()
    metrics_df = pd.DataFrame(metrics)
    metrics_df.columns = class_names
    color_dict = {'Air':'black','GM':'blue','WM':'green','CSF':'yellow','Bone':'orange','Skin':'red'}
    metrics_mv_avg = metrics_df.rolling(smooth,center=False).mean()
    
    n_plots = 2 + np.sum([int(x) for x in [2*(len(val_performance) > 0), len(full_segm_DICE) > 0]])
            
    f, axarr = plt.subplots(n_plots, sharex=False, figsize=(8,10))
    losses_mv_avg.plot(ax=axarr[0])
    axarr[0].set_title(session)
    metrics_mv_avg.plot(ax=axarr[1], color=[color_dict.get(x, '#333333') for x in metrics_mv_avg.columns])
    #axarr[1].plot(metrics_mv_avg)
    #axarr[1].set_title('Single Class Dice Loss')
    axarr[1].set_xlabel('Training Iterations')
    axarr[1].legend(loc='upper left')
       
    if len(val_performance) > 0  :
    
        loss_val = [x[0] for x in val_performance]
        metrics_val = [x[1:len(x)] for x in val_performance]
        
        loss_val_df = pd.DataFrame(loss_val)
        loss_val_df.columns=loss_name
        #loss_val_df = loss_val_df.rolling(smooth,center=False).mean()
        metrics_val_df = pd.DataFrame(metrics_val)
        metrics_val_df.columns = class_names
        #metrics_val_df = metrics_val_df.rolling(smooth,center=False).mean()
        loss_val_df.plot(ax=axarr[2])
        #axarr[2].set_title(loss_name[0])
        metrics_val_df.plot(ax=axarr[3], color=[color_dict.get(x, '#333333') for x in metrics_mv_avg.columns])
        #axarr[1].plot(metrics_mv_avg)
        #axarr[3].set_title('Single Class Dice Loss')
        #axarr[3].set_xlabel('Training Iterations')
        
        axarr[3].legend(loc='upper left')
    
    if len(full_segm_DICE) > 0:
        
        full_segm_DICE = pd.DataFrame(full_segm_DICE)
        full_segm_DICE.columns=['Full Segmentation DICE']
        full_segm_DICE.plot(ax=axarr[n_plots-1],style='-o',color='green')
        axarr[n_plots-1].legend(loc='lower right')
        
def validation_on_batch_quick(model, valbatch, size_minibatches_val, vallabels, valcoords, output_classes, logfile):
    print('\n Validation on Batch. Len: {}'.format(len(valbatch)))
    batch_performance = []
    start = 0
    if len(valbatch)<=size_minibatches_val:
        size_minibatches_val = len(valbatch)
    n_minibatches = len(valbatch)/size_minibatches_val
    for j in range(0,n_minibatches):
        print("validation on minibatch " +str(j+1)+ "/" + str(n_minibatches))
        end = start + size_minibatches_val
        minivalbatch = valbatch[start:end,:,:,:,:]    
        minivalbatch_labels = vallabels[start:end,:,:,:]
        minivalbatch_labels = np.reshape(minivalbatch_labels,(minivalbatch_labels.shape[0],1,9,9,2))    
        minibatch_valcoord = valcoords[start:end,:,:]
        minibatch_valcoord = np.reshape(minibatch_valcoord,(minibatch_valcoord.shape[0],3,9,9,1))  
        #minibatch_coords_X = minibatch_valcoord[:,0,:,:,:]
        minibatch_coords_Y = minibatch_valcoord[:,1,:,:,:]
        minibatch_coords_Z = minibatch_valcoord[:,2,:,:,:]
        #minibatch_coords_X = np.reshape(minibatch_coords_X,(minibatch_coords_X.shape[0],1,9,9,1))
        minibatch_coords_Y = np.reshape(minibatch_coords_Y,(minibatch_coords_Y.shape[0],1,9,9,1))
        minibatch_coords_Z = np.reshape(minibatch_coords_Z,(minibatch_coords_Z.shape[0],1,9,9,1))
    batch_performance.append(model.evaluate([minivalbatch,minibatch_coords_Y,minibatch_coords_Z], minivalbatch_labels, verbose=0))
    val_performance = np.mean(batch_performance, 0)
    my_logger('Validation cost and accuracy ' + str(val_performance),logfile)        
    del valbatch
    del vallabels       
    return list(val_performance)
    
def weighted_generalized_dice_completeImages(img1,img2,penalty_MATRIX):

    # Penalty matrix is just inverse of label compatibility function given in the CRF:
    # 0 = Air, 1 = GM, 2 = WM, 3 = CSF, 4 = Bone, 5 = Skin
    # grap forms two partitions: brain and non-brain
    # Improves Dice score if label is the same (diagonal)
    # does not change Dice score if label is mistaken by same label group = normal penalty
    # actively decreases Dice score if label mistaken by label in other group = active penalty
    # makes dice score now go in range [-1, 1]
    # Saldy it works in O(n^2), but still small enough to be fast.    
    
    
    #assert (np.unique(img1) == np.unique(img2)).all(), 'Images have different classes!'
    classes = np.array(range(0,len(penalty_MATRIX)), dtype='int8')   
    dice = []
    
    for i in classes:
        dice_2 = []
        #DICE = 2*np.sum(np.multiply(img1==i,img2==i))/float(np.sum(img1==i)+np.sum(img2==i))
        for j in classes:
            wDice = 2*np.sum(np.multiply(img1==i,img2==j) * penalty_MATRIX[i,j] )/float(np.sum(img1==i)+np.sum(img2==j))
            dice_2.append(wDice)
        dice.append(np.sum(dice_2)) 
    return np.sum(dice)/len(classes), [round(x,2) for x in dice]

def getVarFromFile(filename):
    import imp
    print('import using {}'.format(filename))
    f = open(filename)
    global cfg
    cfg = imp.load_source('cfg', '', f)
    f.close()
    
############################## create or load model ###########################################

def make_model(model_configFile):
    
    path = '/'.join(model_configFile.split('/')[:-1])
    model_configFileName = model_configFile.split('/')[-1][:-3]   
    sys.path.append(path)

    cfg = __import__(model_configFileName)
    
    if cfg.model == 'CNN_TPM':
        from multiscale_CNN_TPM import multiscale_CNN_TPM            
        dm = multiscale_CNN_TPM(cfg.dpatch, cfg.output_classes, cfg.num_channels, cfg.L2, cfg.dropout, cfg.learning_rate, cfg.optimizer_decay, cfg.loss_function)
        model = dm.createModel()        
    
    elif cfg.model == 'DeepMedic':
        from DeepMedic_model import DeepMedic
        dm = DeepMedic(cfg.dpatch, cfg.output_classes, cfg.num_channels, cfg.L2, cfg.dropout, cfg.learning_rate, cfg.optimizer_decay, cfg.loss_function)
        model = dm.createModel()

    else: 
        print('ERROR: No model selected.')
        return 0
        
    print(model.summary())

    from keras.utils import plot_model
    plot_model(model, to_file=cfg.workingDir+'/models/'+cfg.model +'.png', show_shapes=True)
    model_path = cfg.workingDir+'/models/'+cfg.model +'.h5'
    model.save(model_path)
    print('Saved model at {}'.format(model_path))

############################## training session ###########################################

configFile = '/home/hirsch/Documents/projects/Breast_segmentation/DeepPriors_package/configFiles/configFiles_stroke/configFile_DM.py'
workingDir ='/home/hirsch/Documents/projects/Breast_segmentation/DeepPriors_package/'

def train_test_model(configFile, workingDir):

    # import configuration file and create working environment

    print(configFile)
    path = '/'.join(configFile.split('/')[:-1])
    print(path)
    configFileName = configFile.split('/')[-1][:-3]   
    sys.path.append(path)
    cfg = __import__(configFileName)


    cfg.TPM_channel = workingDir + cfg.TPM_channel
    cfg.trainChannels = [workingDir + x for x in cfg.trainChannels]
    cfg.trainLabels = workingDir +cfg.trainLabels 
    cfg.testChannels = [workingDir + x for x in cfg.testChannels]
    cfg.testLabels = workingDir + cfg.testLabels
    cfg.validationChannels = [workingDir + x for x in cfg.validationChannels]
    cfg.validationLabels = workingDir +cfg.validationLabels

    # Create or load CNN model

    if cfg.load_model == False:
        if cfg.model == 'CNN_TPM':
            from multiscale_CNN_TPM import multiscale_CNN_TPM            
            dm = multiscale_CNN_TPM(cfg.dpatch, cfg.output_classes, cfg.num_channels, cfg.L2, cfg.dropout, cfg.learning_rate, cfg.optimizer_decay, cfg.loss_function)
            model = dm.createModel()        
        
        elif cfg.model == 'DeepMedic':
            from DM_MSKCC_model import DeepMedic
            dm = DeepMedic(cfg.dpatch, cfg.output_classes, cfg.num_channels, cfg.L2, cfg.dropout, cfg.learning_rate, cfg.optimizer_decay, cfg.loss_function)
            model = dm.createModel()

            
        elif cfg.model == 'Atrous':
            from DM_MSKCC_Atrous_model import DeepMedic
            dm = DeepMedic(cfg.dpatch, cfg.output_classes, cfg.num_channels, cfg.L2, cfg.dropout, cfg.learning_rate, cfg.optimizer_decay, cfg.loss_function)
            model = dm.createModel()
            

        else: 
            print('ERROR: No model selected.')
            return 0
            
        start_epoch = 0
        os.chdir(workingDir + '/training_sessions/')
        session = cfg.model + '_' + cfg.dataset + '_' + configFileName + '_' + time.strftime("%Y-%m-%d_%H%M") 
        wd = workingDir + '/training_sessions/' +session
        if not os.path.exists(wd):    
            os.mkdir(session)
            os.mkdir(session + '/models')
            os.mkdir(session + '/predictions')
        os.chdir(wd)
    
        logfile = session +'.log'
            
        print(model.summary())
        val_performance = []
        from keras.utils import plot_model
        plot_model(model, to_file=wd+'/multiscale_TPM.png', show_shapes=True)
        model_path = workingDir+'/Output/models/'+session+'.h5'
        with open(wd+'/model_summary.txt','w') as fh:
        # Pass the file handle in as a lambda function to make it callable
            model.summary(print_fn=lambda x: fh.write(x + '\n'))
        if len(cfg.comments) > 0:
            f = open('Comments.txt','w')
            f.write(str(cfg.comments))
            f.close()
        
    elif cfg.load_model == True:
        from keras.models import load_model  
        if cfg.loss_function == 'Dice6':
            from DM_MSKCC_Atrous_model import dice_coef_multilabel6, dice_coef_multilabel0,dice_coef_multilabel1,dice_coef_multilabel2,dice_coef_multilabel3,dice_coef_multilabel4,dice_coef_multilabel5
            my_custom_objects = {'dice_coef_multilabel6':dice_coef_multilabel6,
				     'dice_coef_multilabel0':dice_coef_multilabel0,
				     'dice_coef_multilabel1':dice_coef_multilabel1,
				     'dice_coef_multilabel2':dice_coef_multilabel2,
				     'dice_coef_multilabel3':dice_coef_multilabel3,
				     'dice_coef_multilabel4':dice_coef_multilabel4,
				     'dice_coef_multilabel5':dice_coef_multilabel5}
            #custom_metrics = [dice_coef_multilabel6,dice_coef_multilabel0,dice_coef_multilabel1,dice_coef_multilabel2,dice_coef_multilabel3,dice_coef_multilabel4,dice_coef_multilabel5]
        elif cfg.loss_function == 'wDice6':
            from DM_MSKCC_Atrous_model import w_dice_coef_multilabel6, dice_coef_multilabel0,dice_coef_multilabel1,dice_coef_multilabel2,dice_coef_multilabel3,dice_coef_multilabel4,dice_coef_multilabel5
            my_custom_objects = {'w_dice_coef_multilabel6':w_dice_coef_multilabel6,
				     'dice_coef_multilabel0':dice_coef_multilabel0,
				     'dice_coef_multilabel1':dice_coef_multilabel1,
				     'dice_coef_multilabel2':dice_coef_multilabel2,
				     'dice_coef_multilabel3':dice_coef_multilabel3,
				     'dice_coef_multilabel4':dice_coef_multilabel4,
				     'dice_coef_multilabel5':dice_coef_multilabel5}
        elif cfg.loss_function == 'Dice2':
            from DM_MSKCC_Atrous_model import Generalised_dice_coef_multilabel2, dice_coef_multilabel0,dice_coef_multilabel1
            my_custom_objects = {'Generalised_dice_coef_multilabel2':Generalised_dice_coef_multilabel2,
				     'dice_coef_multilabel0':dice_coef_multilabel0,
				     'dice_coef_multilabel1':dice_coef_multilabel1}
        elif cfg.loss_function == 'wDice2':
            from DM_MSKCC_Atrous_model import w_dice_coef_multilabel2, dice_coef_multilabel0,dice_coef_multilabel1
            my_custom_objects = {'w_dice_coef_multilabel2':w_dice_coef_multilabel2,
				     'dice_coef_multilabel0':dice_coef_multilabel0,
				     'dice_coef_multilabel1':dice_coef_multilabel1}
 
        model = load_model(cfg.path_to_model, custom_objects = my_custom_objects )
        print('LOADED MODEL FROM SESSION {}'.format(cfg.session))
        session = cfg.session


        #copyfile(cfg.configFile_address, wd + '/configFile_continued')
        start_epoch = int(cfg.path_to_model.split('.')[-2][cfg.path_to_model.split('.')[-2].find('epoch') + 5 : ]) + 1
        cfg.epochs_for_fullSegmentation = range(start_epoch+1, cfg.epochs)
        os.chdir(workingDir + '/training_sessions/')
        wd = workingDir + '/training_sessions/' +session
        if not os.path.exists(wd):    
            os.mkdir(session)
            os.mkdir(session + '/models')
            os.mkdir(session + '/predictions')
        os.chdir(wd)
    
        logfile = session +'.log'


    #################################################################################################
    #                                                                                               #
    #                                         START SESSION                                         #
    #                                                                                               #
    #################################################################################################
    
    # OUTCOMMENTED SO I CAN KEEP USING SAME TRAINING DATA FOR SAME MODEL.
    val_performance = []
    full_segm_DICE = []
    losses = []
    metrics = []
    np.set_printoptions(precision=3)
    l = 0
    start_training_session_logger(logfile, cfg.threshold_EARLY_STOP, cfg.TPM_channel, cfg.load_model, cfg.saveSegmentation, cfg.path_to_model, model, \
        cfg.dropout, cfg.trainChannels, cfg.trainLabels, cfg.validationChannels, cfg.validationLabels, \
        cfg.testChannels, cfg.testLabels, cfg.num_iter, cfg.epochs, cfg.n_patches, cfg.n_patches_val, cfg.n_subjects, cfg.samplingMethod_train, \
        cfg.size_minibatches, cfg.n_fullSegmentations, cfg.epochs_for_fullSegmentation, cfg.size_test_minibatches)
    # Callback history    
    if cfg.output_classes == 2:
        history = LossHistory_multiDice2() 
    elif cfg.output_classes == 6:
        history = LossHistory_multiDice6()
    
    EARLY_STOP = False
    
    for epoch in xrange(start_epoch,cfg.epochs):
    
        t1 = time.time()
        my_logger("######################################################",logfile)
        my_logger("                   TRAINING EPOCH " + str(epoch+1) + "/" + str(cfg.epochs),logfile)
        my_logger("######################################################",logfile)
                
        ####################### FULL HEAD SEGMENTATION ##############################
                
        if cfg.quick_segmentation:
             if epoch in cfg.epochs_for_fullSegmentation:
                my_logger("------------------------------------------------------", logfile)
                my_logger("                 FULL HEAD SEGMENTATION", logfile)
                my_logger("------------------------------------------------------", logfile)
                dice_compare = True
                dsc = []
                subjectIndex = 0
                with open(cfg.validationLabels) as vl:
                    n_valSubjects = len(vl.readlines())
                if cfg.test_subjects > n_valSubjects:
                    print("Given number of subjects for test set (" + str(cfg.test_subjects) +") is larger than the amount of \
                    subjects in test set (" +str(n_valSubjects)+ ")")
                    cfg.test_subjects = n_valSubjects
                    print('Using {} number of test subjects'.format(n_valSubjects))
                if len(cfg.list_subjects_fullSegmentation) == 0:
                    list_subjects_fullSegmentation = sample(range(cfg.test_subjects), cfg.n_fullSegmentations)
                else:
                    list_subjects_fullSegmentation = cfg.list_subjects_fullSegmentation
                for subjectIndex in list_subjects_fullSegmentation: 
                    
                    if (cfg.model == 'DeepMedic') or (cfg.model == 'Atrous'):
                        fullHeadSegmentation(wd, cfg.penalty_MATRIX, dice_compare, dsc, model, cfg.testChannels, cfg.testLabels, subjectIndex, \
                        cfg.output_classes, cfg.dpatch, cfg.size_test_minibatches, logfile, epoch, cfg.saveSegmentation)
                    else:
                        continue
                    my_logger('--------------- TEST EVALUATION ---------------', logfile)
                    my_logger('          Full segmentation evaluation of subject' + str(subjectIndex), logfile)
                    if dice_compare: my_logger('DCS ' + str(dsc[-1]),logfile)
                my_logger('         FULL SEGMENTATION SUMMARY STATISTICS ', logfile)
                full_segm_DICE.append(np.mean(dsc))
                my_logger('Overall DCS:   ' + str(full_segm_DICE[-1]),logfile)
                
                # Function to define if STOP flag goes to True or not, based on difference between last three or two segmentations.
                if len(full_segm_DICE) > 5:                        
                    if np.max(np.abs(np.diff([full_segm_DICE[-3], full_segm_DICE[-2], full_segm_DICE[-1]] ))) < cfg.threshold_EARLY_STOP:
                        EARLY_STOP = True
                    #elif np.max(np.abs(np.diff([full_segm_DICE[-5],full_segm_DICE[-4],full_segm_DICE[-3], full_segm_DICE[-2], full_segm_DICE[-1]] ))) < 0.03:
                    #    EARLY_STOP = True

		# Save model if best results achieved
		if len(full_segm_DICE) > 0:
	  		if np.max(full_segm_DICE) <= full_segm_DICE[-1]:
				my_logger('###### SAVING TRAINED MODEL AT : ' + wd +'/Output/models/'+logfile[12:]+'.h5', logfile)
				model.save(wd+'/models/'+logfile[12:]+'_epoch' + str(epoch+1) + '.h5')
                
        #################################################################################################
        #                                                                                               #
        #                               Training and Validation                                         #
        #                                                                                               #
        #################################################################################################
        if EARLY_STOP:
           my_logger('Convergence criterium met. Stopping training.',logfile)
           break           
               
        for i in range(0, cfg.num_iter):
            my_logger("                   Batch " + str(i+1) + "/" + str(cfg.num_iter) ,logfile)
            my_logger("###################################################### ",logfile)
                      
            ####################### VALIDATION ON BATCHES ############################                      
                      
            if not cfg.quickmode:   
                with open(cfg.validationLabels) as vl:
                    n_valSubjects = len(vl.readlines())

                if cfg.n_subjects_val > n_valSubjects:
                    print("Given number of subjects for test set (" + str(cfg.n_subjects_val) +") is larger than the amount of \
                    subjects in test set (" +str(n_valSubjects)+ ")")
                    cfg.n_subjects_val = n_valSubjects
                    print('Using {} number of test subjects'.format(n_valSubjects))
                    
                valbatch, vallabels, valcoords = sampleTrainData(cfg.validationChannels, cfg.validationLabels, cfg.n_patches_val, cfg.n_subjects_val, cfg.dpatch, cfg.output_classes, cfg.samplingMethod_val, logfile)
                #print(valcoords.shape)
                print(valcoords[0])
                print(valcoords[0].dtype)
                print(valcoords.shape)
                print(vallabels.shape)
                val_performance.append(validation_on_batch_quick(model, valbatch, cfg.size_minibatches_val, vallabels, valcoords, cfg.output_classes, logfile))
                del valbatch, vallabels
          
            ####################### TRAINING ON BATCHES ##############################
            
            with open(cfg.trainLabels) as vl:
                    n_trainSubjects = len(vl.readlines())                
            if cfg.n_subjects > n_trainSubjects:
                print("Given number of subjects for test set (" + str(cfg.n_subjects) +") is larger than the amount of \
                subjects in test set (" +str(n_trainSubjects)+ ")")
                cfg.n_subjects = n_trainSubjects
                print('Using {} number of test subjects'.format(n_trainSubjects))
         
            print('sampling {} patches'.format(cfg.n_patches))
            batch = 0
            labels = 0
            batch, labels, coords = sampleTrainData(cfg.trainChannels,cfg.trainLabels, cfg.n_patches, cfg.n_subjects, cfg.dpatch, cfg.output_classes, cfg.samplingMethod_train, logfile) 
            assert not np.any(np.isnan(batch)), 'nan found in the input data!'   
            print('TRAINING BATCH')
            print(batch.shape)
            print(labels.shape)
            print(coords.shape)
            shuffleOrder = np.arange(batch.shape[0])
            np.random.shuffle(shuffleOrder)
            batch = batch[shuffleOrder]
            labels = labels[shuffleOrder]  
            coords = coords[shuffleOrder]
            #freq = classesInSample(labels, cfg.output_classes)
            #my_logger("Sampled following number of classes in training batch: " + str(freq), logfile)
            #print(getClassProportions(freq))

            train_model_on_batch(model,batch,labels, coords,cfg.size_minibatches,history,losses,metrics,l,logfile,cfg.output_classes)  
            del batch, labels
            # For large datasets, save model after every 'epoch'
            model.save(wd+'/models/'+logfile[12:]+'_epoch' + str(epoch+1) + '.h5')        
        my_logger('Total training this epoch took ' + str(round(time.time()-t1,2)) + ' seconds',logfile)

    if cfg.output_classes == 6:
    	plot_training(session,losses, metrics, val_performance, full_segm_DICE, smooth=20, loss_name = [cfg.loss_function], class_names = ['Air','GM','WM','CSF','Bone','Skin'])
    elif cfg.output_classes == 2:
    	plot_training(session,losses, metrics, val_performance, full_segm_DICE, smooth=20, loss_name = [cfg.loss_function], class_names = ['Background','Lesion'])
    plt.savefig(wd + '/' + session + '.png')
    plt.close()

############################## make prediction / inference ###########################################

#configFile = '/home/hirsch/Documents/projects/brainSegmentation/DeepPriors/configFiles/configFiles_stroke/segmentation/DeepPriors_segmentation.py'

def segment(configFile,workingDir):

    path = '/'.join(configFile.split('/')[:-1])
    configFileName = configFile.split('/')[-1][:-3]   
    sys.path.append(path)
    cfg = __import__(configFileName)
           
    start_epoch = int(cfg.path_to_model.split('.')[-2][cfg.path_to_model.split('.')[-2].find('epoch') + 5 : ]) + 1
        
    os.chdir(workingDir + '/training_sessions/')
    session = cfg.session
    wd = workingDir + '/training_sessions/' +session
    print('\n CURRENTLY IN SESSION {} \n'.format(session))
    if not os.path.exists(wd):    
        os.mkdir(session)
        os.mkdir(session + '/models')
        os.mkdir(session + '/predictions')
    os.chdir(wd)
    
    logfile = 'segmentations.log'

    #cfg.TPM_channel = workingDir + cfg.TPM_channel
    cfg.segmentChannels = [workingDir + x for x in cfg.segmentChannels]
    cfg.segmentLabels = workingDir + cfg.segmentLabels 

    from keras.models import load_model   
    if cfg.output_classes == 6:
	try:
	    from DM_MSKCC_model import dice_coef_multilabel6, dice_coef_multilabel0,dice_coef_multilabel1,dice_coef_multilabel2,dice_coef_multilabel3,dice_coef_multilabel4,dice_coef_multilabel5
	    my_custom_objects = {'dice_coef_multilabel6':dice_coef_multilabel6,
				     'dice_coef_multilabel0':dice_coef_multilabel0,
				     'dice_coef_multilabel1':dice_coef_multilabel1,
				     'dice_coef_multilabel2':dice_coef_multilabel2,
				     'dice_coef_multilabel3':dice_coef_multilabel3,
				     'dice_coef_multilabel4':dice_coef_multilabel4,
				     'dice_coef_multilabel5':dice_coef_multilabel5}
		#custom_metrics =[dice_coef_multilabel6,dice_coef_multilabel0,dice_coef_multilabel1,dice_coef_multilabel2,dice_coef_multilabel3,dice_coef_multilabel4,dice_coef_multilabel5]
		#my_custom_objects = dict(zip(np.sort(my_custom_objects.keys()), custom_metrics))
	    model = load_model(cfg.path_to_model, custom_objects = my_custom_objects )
	except:
	    from DM_MSKCC_model import w_dice_coef_multilabel6, dice_coef_multilabel0,dice_coef_multilabel1,dice_coef_multilabel2,dice_coef_multilabel3,dice_coef_multilabel4,dice_coef_multilabel5
	    my_custom_objects = {'w_dice_coef_multilabel6':w_dice_coef_multilabel6,
					     'dice_coef_multilabel0':dice_coef_multilabel0,
					     'dice_coef_multilabel1':dice_coef_multilabel1,
					     'dice_coef_multilabel2':dice_coef_multilabel2,
					     'dice_coef_multilabel3':dice_coef_multilabel3,
					     'dice_coef_multilabel4':dice_coef_multilabel4,
					     'dice_coef_multilabel5':dice_coef_multilabel5}
    elif cfg.output_classes == 2:
        try:
            from DM_MSKCC_model import Generalised_dice_coef_multilabel2, dice_coef_multilabel0,dice_coef_multilabel1
            my_custom_objects = {'Generalised_dice_coef_multilabel2':Generalised_dice_coef_multilabel2,
				     'dice_coef_multilabel0':dice_coef_multilabel0,
				     'dice_coef_multilabel1':dice_coef_multilabel1}
        except:
            from DM_MSKCC_model import w_dice_coef_multilabel2, dice_coef_multilabel0,dice_coef_multilabel1
            my_custom_objects = {'w_dice_coef_multilabel2':w_dice_coef_multilabel2,
				     'dice_coef_multilabel0':dice_coef_multilabel0,
				     'dice_coef_multilabel1':dice_coef_multilabel1}
        model = load_model(cfg.path_to_model, custom_objects = my_custom_objects )

    full_segm_DICE = []
    np.set_printoptions(precision=3)

    print("------------------------------------------------------")
    print("                 FULL HEAD SEGMENTATION")
    print("------------------------------------------------------")
    dice_compare = True
    dsc = []
    subjectIndex = 0
    epoch = 0
    with open(cfg.segmentLabels) as vl:
        n_segmentSubjects = len(vl.readlines())
    if cfg.test_subjects > n_segmentSubjects:
	print("Given number of subjects for test set (" + str(cfg.test_subjects) +") is larger than the amount of \
	subjects in test set (" +str(n_segmentSubjects)+ ")")
	cfg.test_subjects = n_segmentSubjects
	print('Using {} number of test subjects'.format(n_segmentSubjects))
    list_subjects_fullSegmentation = range(cfg.test_subjects)
    for subjectIndex in list_subjects_fullSegmentation: 
        
        if (cfg.model == 'DeepMedic') or (cfg.model == 'Atrous'):
            fullHeadSegmentation(wd, cfg.penalty_MATRIX, dice_compare, dsc, model, cfg.segmentChannels, cfg.segmentLabels, subjectIndex, \
            cfg.output_classes, cfg.dpatch, cfg.size_test_minibatches, logfile, epoch, cfg.saveSegmentation)
        else:
            continue

        my_logger('--------------- TEST EVALUATION ---------------', logfile)
        my_logger('          Full segmentation evaluation of subject' + str(subjectIndex), logfile)
        if dice_compare: my_logger('DCS ' + str(dsc[-1]),logfile)
            
    my_logger('         FULL SEGMENTATION SUMMARY STATISTICS ', logfile)
    full_segm_DICE.append(np.mean(dsc))
    my_logger('Overall DCS:   ' + str(full_segm_DICE[-1]),logfile)
    
    plt.hist(dsc, 80, edgecolor = 'black')
    #plt.axvline(np.mean(dsc), color = 'red', linewidth = 3)
    #plt.axvline(0.89, color = 'b', linestyle='dashed', linewidth = 3)
    plt.xlabel('Dice score')
    plt.ylabel('Frequency')
    plt.title('Dice score distribution')
    #create legend
    #handles = [Rectangle((0,0),1,1,color=c,ec="k") for c in ['r','b']]
    #labels= ["Achieved (" + str(np.round(np.mean(dsc),2)) + ")","DeepMedic (0.89)"]
    #plt.legend(handles, labels)
    #plt.savefig('/home/hirsch/Documents/projects/brainSegmentation/deepMedicKeras/Output/images/diceHist_epoch' + str(epoch) + '_' + logfile_model + '.png')
