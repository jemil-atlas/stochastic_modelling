#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The goal of this script is to produce an illustration of basic tri data.
"""



# i) imports

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import copy


# ii) Load data
# data = loadmat('../data_stochastic_modelling/data_bafu_stochastic_model/submatrix_collection_training_20x20_2023_2days.mat')
data = loadmat('../data_stochastic_modelling/data_bafu_stochastic_model/Single_Full_Info_mat.mat')
data_SLC_1 = loadmat('../data_stochastic_modelling/data_bafu_stochastic_model/SLC_1.mat')['SLC_1']
data_SLC_2 = loadmat('../data_stochastic_modelling/data_bafu_stochastic_model/SLC_2.mat')['SLC_2']

info_mats = data['Info_mat']

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
    

# Create subimages
index_choice_az = np.round(np.linspace(250, 270, 21)).astype(int)
index_choice_r = np.round(np.linspace(3150, 3250, 401)).astype(int)

amplitude_img = np.abs(data_SLC_1[np.ix_(index_choice_r, index_choice_az)])
phase_img = np.angle(data_SLC_1[np.ix_(index_choice_r, index_choice_az)])
intf_img = np.angle((data_SLC_1*np.conj(data_SLC_2))[np.ix_(index_choice_r, index_choice_az)])
coh_img = info_mats[:,:,5][np.ix_(index_choice_r, index_choice_az)]
range_img = info_mats[:,:,0][np.ix_(index_choice_r, index_choice_az)]
az_img = info_mats[:,:,1][np.ix_(index_choice_r, index_choice_az)]
z_img = info_mats[:,:,4][np.ix_(index_choice_r, index_choice_az)]
aoi_img = info_mats[:,:,6][np.ix_(index_choice_r, index_choice_az)]


# Plot the subplots

fig, axs = plt.subplots(2, 4, figsize=(15, 10))

ax = axs[0, 0]
ax.imshow(amplitude_img, aspect='auto')
ax.set_title('Amplitude')
ax.axis('off')

ax = axs[0, 1]
ax.imshow(phase_img, aspect='auto')
ax.set_title('Phase')
ax.axis('off')

ax = axs[0, 2]
ax.imshow(intf_img, aspect='auto')
ax.set_title('Phase difference')
ax.axis('off')

ax = axs[0, 3]
ax.imshow(coh_img, aspect='auto')
ax.set_title('Coherence')
ax.axis('off')


ax = axs[1, 0]
ax.imshow(range_img, aspect='auto')
ax.set_title('Range')
ax.axis('off')

ax = axs[1, 1]
ax.imshow(az_img, aspect='auto')
ax.set_title('Azimuth')
ax.axis('off')

ax = axs[1, 2]
ax.imshow(z_img, aspect='auto')
ax.set_title('Elevation')
ax.axis('off')

ax = axs[1, 3]
ax.imshow(aoi_img, aspect='auto')
ax.set_title('Angle of incidence')
ax.axis('off')





