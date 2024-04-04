#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The goal of this script is to produce a map of averaging times based on the 
decorrelated model.
"""



# i) imports

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.io import loadmat
import copy


# ii) Definitions

n_r_average = 10
n_az_average = 2
n_t = 24
c_phase_to_def = (-17.4 / (4* np.pi)) # result in mm


# iii) Load data
# data = loadmat('../data_stochastic_modelling/data_bafu_stochastic_model/submatrix_collection_training_20x20_2023_2days.mat')
data = loadmat('../data_stochastic_modelling/data_bafu_stochastic_model/Single_Full_Info_mat.mat')
data_SLC_1 = loadmat('../data_stochastic_modelling/data_bafu_stochastic_model/SLC_1.mat')['SLC_1']
data_SLC_2 = loadmat('../data_stochastic_modelling/data_bafu_stochastic_model/SLC_2.mat')['SLC_2']

info_mats = data['Info_mat']

# Data reduction

n_r, n_az, n_f = info_mats.shape

n_r_new = np.ceil(n_r / n_r_average).astype(int)
n_az_new = np.ceil(n_az/n_az_average).astype(int)

A_r = np.zeros([n_r_new, n_r])
A_az = np.zeros([n_az, n_az_new])

for k in range(n_r_new):
    for l in range(n_r):
        if l >= (k)*n_r_average and l< (k+1)*n_r_average:    
            A_r[k,l] = 1
        else:
            pass
        
for k in range(n_az):
    for l in range(n_az_new):
        if k >= (l)*n_az_average and k< (l+1)*n_az_average:    
            A_az[k,l] = 1
        else:
            pass
                

# Normalize rows or cols
A_r = A_r / (np.sum(A_r, axis = 1).reshape([-1,1]))
A_az = A_az / (np.sum(A_az, axis = 0).reshape([1,-1]))

info_mats_reduced = np.zeros([n_r_new, n_az_new, n_f])
for k in range(n_f):
    info_mats_reduced[:,:,k] = A_r @ info_mats[:,:,k] @ A_az


# Build the different images for display
# BH 3 indices [261, 3189] in slc, [131, 319] in mli
# dict_ind_base_data = {'range_mats' : 0,\
#                       'azimuth_mats' : 1,\
#                       'x_mats' : 2,\
#                       'y_mats' : 3,\
#                       'z_mats' : 4,\
#                       'coherence_mats' : 5,\
#                       'aoi_mats' : 6,\
#                       'meanphase_mats' : 7,\
#                       'time_mats' : 8,\
#                       }
    

# # Create subimages
# index_choice_az = np.round(np.linspace(250, 270, 21)).astype(int)
# index_choice_r = np.round(np.linspace(3150, 3250, 401)).astype(int)

# amplitude_img = np.abs(data_SLC_1[np.ix_(index_choice_r, index_choice_az)])
# phase_img = np.angle(data_SLC_1[np.ix_(index_choice_r, index_choice_az)])
# intf_img = np.angle((data_SLC_1*np.conj(data_SLC_2))[np.ix_(index_choice_r, index_choice_az)])
# coh_img = info_mats[:,:,5][np.ix_(index_choice_r, index_choice_az)]
# range_img = info_mats[:,:,0][np.ix_(index_choice_r, index_choice_az)]
# az_img = info_mats[:,:,1][np.ix_(index_choice_r, index_choice_az)]
# z_img = info_mats[:,:,4][np.ix_(index_choice_r, index_choice_az)]
# aoi_img = info_mats[:,:,6][np.ix_(index_choice_r, index_choice_az)]


# iii) Load decorrelated model


# compute
def compute_cov_mat_for_pixel(i,j):
    # Compute the temporal covariance matrices for a specific pixel
    # i = row nr, j = col nr
    
    pass
    
def get_averaging_length_for_pixel(i,j):
    # Compute the temporal averaging needed for sigma <= 1mm
    # i = row nr, j = col nr
    
    cov_mat = compute_cov_mat_for_pixel(i,j, n_t)
    
    variance = torch.zeros(n_t)
    for k in range(n_t):
        variance[k] = (1/k**2) * torch.sum(cov_mat[0:k, 0:k])
    std_def = np.sqrt(variance) * np.abs(c_phase_to_def)
    averaging_length = torch.where(std_def <= 1)[0][0].item()
    
    pass
    

# iv) Iterate through pixels

averaging_lengths = torch.zeros([n_r,n_az, 2])
for i in range(n_r):
    for j in range(n_az):
        averaging_lengths[i,j,:] = get_averaging_length_for_pixel(i,j)
        