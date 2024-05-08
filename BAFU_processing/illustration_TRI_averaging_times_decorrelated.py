#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The goal of this script is to produce a map of averaging times based on the 
decorrelated model.
"""



# i) imports

import numpy as np
import torch
import pyro
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.io import savemat
import copy


# ii) Definitions

n_r_average = 10
n_az_average = 2
n_t = 20
c_phase_to_def = (-17.4 / (4* np.pi)) # result in mm


# iii) Load data
# data = loadmat('../data_stochastic_modelling/data_bafu_stochastic_model/submatrix_collection_training_20x20_2023_2days.mat')
data = loadmat('../data_stochastic_modelling/data_bafu_stochastic_model/Single_Full_Info_mat.mat')
data_SLC_1 = loadmat('../data_stochastic_modelling/data_bafu_stochastic_model/SLC_1.mat')['SLC_1']
data_SLC_2 = loadmat('../data_stochastic_modelling/data_bafu_stochastic_model/SLC_2.mat')['SLC_2']

info_mats = data['Info_mat']

data_mean = info_mats.mean(axis=(0, 1), keepdims=True)
data_std = info_mats.std(axis=(0, 1), keepdims=True)
info_mats_standardized = (info_mats - data_mean) / (data_std + 1e-6)


# Data reduction

n_r, n_az, n_f = info_mats.shape

n_r_new = np.ceil(n_r / n_r_average).astype(int)
n_az_new = np.ceil(n_az/n_az_average).astype(int)
n_pixels = n_r_new * n_az_new

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
    info_mats_reduced[:,:,k] = A_r @ info_mats_standardized[:,:,k] @ A_az
info_mats_reduced = np.reshape(info_mats_reduced, [1, n_pixels, n_f])
info_mats_reduced = np.concatenate((info_mats_reduced, np.zeros([*info_mats_reduced.shape[0:2],1])), axis = 2)

#torch.unsqueeze(torch.tensor(info_mats_reduced),0).numpy()

# Build the different images for display
# BH 3 indices [261, 3189] in slc, [131, 319] in mli
dict_ind_base_data = {'range_mats' : 0,\
                      'azimuth_mats' : 1,\
                      'x_mats' : 2,\
                      'y_mats' : 3,\
                      'z_mats' : 4,\
                      'coherence_mats' : 5,\
                      'aoi_mats' : 6,\
                      'meanphase_mats' : 7,\
                      'time_mats' : 8,\
                      }
    

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
# model_save_path = '../results_stochastic_modelling/results_bafu_stochastic_model/ts_model.pth'
model_save_path = '../results_stochastic_modelling/results_bafu_stochastic_model/phase_mean_and_variance_models_2024-03-23.pth'
pyro.get_param_store().load(model_save_path)

simplicity = False


# Import dataset and model structures

# i) Dataset preparation

# basic inputs to stochastic model
class BaseData():
    def __init__(self, info_mats_data, dict_ind_base_data):
        self.n_data = info_mats_data.shape[0]
        for name, index in dict_ind_base_data.items():
            base_data = torch.tensor(np.transpose(info_mats_data[:,:,index].reshape([-1,n_r_new,n_az_new]), (0,2,1))).double()
            setattr(self, name, base_data)

        def get_basic_attr_list(self):
            attributes = [attr for attr in dir(self) if not (attr.startswith('__'))]
            return attributes
        self.basic_attribute_list = get_basic_attr_list(self)
        self.num_basic_attributes = len(self.basic_attribute_list)
        
    def extract_regression_tensor(self, list_regression_vars):
        
        regvar_list = []
        for regvar in list_regression_vars:
            regvar_list.append(getattr(self,regvar))
        regression_tensor = torch.stack(regvar_list, dim = 3)
        return regression_tensor
        
        # n_regvars = len(list_regression_vars)
        # regression_tensor = torch.zeros([])
        # for regvar in list_regression_vars:
        #     getattr(self, regvar)
            
        
base_data = BaseData(info_mats_reduced, dict_ind_base_data)


# # full outputs of stochastic model
# class FullData():
#     def __init__(self, base_data, phase_mats, **optional_input):
#         self.K_phase_mats = optional_input.get('K_phase_mats', None)
#         if self.K_phase_mats is not None:
#                 variances = np.diagonal(self.K_phase_mats.detach(), axis1=1, axis2=2)
#                 self.aps_variance_mats = variances.reshape((n_samples, n_x, n_y))
#         # List distributional_data
#         def get_distributional_attr_list(self):
#             attributes = [attr for attr in dir(self) if not (attr.startswith('__') )]
#             return attributes
#         self.distributional_attribute_list = get_distributional_attr_list(self)
#         self.num_distributional_attributes = len(self.distributional_attribute_list)
#         # copy from base_data
#         for name, attribute in base_data.__dict__.items():
#             setattr(self, name, attribute)
#         # Add simulations
#         self.phase_mats = phase_mats     
#         self.phase_mats_illu = phase_mats[illustration_choice,:]

# full_data = FullData(base_data, phase_mats)

list_regression_vars = ['x_mats', 'y_mats', 'z_mats', 'coherence_mats', 'aoi_mats']
# list_regression_vars = ['coherence_mats']
n_regression_vars = len(list_regression_vars)
regression_data = np.transpose(base_data.extract_regression_tensor(list_regression_vars), (0,2,1,3))

def mean_function(regression_data, alpha_0, alpha_1, alpha_2, simplicity = False):
    mu_1 = alpha_0
    mu_2 = torch.einsum('ijkl, l -> ijk', regression_data, alpha_1)
    regvar_products = torch.einsum('ijkl, ijkm -> ijklm', regression_data, regression_data)
    mu_3 = torch.einsum('ijklm, lm -> ijk', regvar_products, alpha_2)
    
    if simplicity == True:    
        mu = mu_1 + mu_2 
    else:
        mu = mu_1 + mu_2 + mu_3
    return mu

def var_function(regression_data, beta_0, beta_1, beta_2, simplicity = False):
    sigma_1 = beta_0
    sigma_2 = torch.einsum('ijkl, l -> ijk', regression_data, beta_1)
    regvar_products = torch.einsum('ijkl, ijkm -> ijklm', regression_data, regression_data)
    sigma_3 = torch.einsum('ijklm, lm -> ijk', regvar_products, beta_2)
    
    if simplicity == True:    
        sigma = torch.exp(sigma_1 + sigma_2) + 1e-5
    else:
        sigma = torch.exp(sigma_1 + sigma_2 + sigma_3) + 1e-5
    return sigma

def eval_model(regression_data):
    # Instantiate parameters
    alpha_0 = pyro.get_param_store()['alpha_0'].detach()
    alpha_1 = pyro.get_param_store()['alpha_1'].detach()
    alpha_2 = pyro.get_param_store()['alpha_2'].detach()
    
    beta_0 = pyro.get_param_store()['beta_0'].detach()
    beta_1 = pyro.get_param_store()['beta_1'].detach()
    beta_2 = pyro.get_param_store()['beta_2'].detach()
        
    # Determine distribution and sample from it
    with pyro.plate('batch_plate_eval', size = 1, dim = -3):
        mu = mean_function(regression_data, alpha_0, alpha_1, alpha_2, simplicity = simplicity)
        sigma = var_function(regression_data, beta_0, beta_1, beta_2, simplicity = simplicity)
        data_dist = pyro.distributions.Normal(mu, sigma)
        with pyro.plate('y_plate_eval', size = n_r_new, dim = -2):
            with pyro.plate('x_plate_eval', size = n_az_new, dim = -1):
                phase_sample = pyro.sample('phase_sample_eval', data_dist)
    
    return phase_sample, mu, sigma

simulation_posttrain, mu_posttrain, sigma_posttrain = copy.copy(eval_model(regression_data))
simulation_posttrain = simulation_posttrain.reshape([-1, n_r_new,n_az_new]) 
sigma_posttrain_cropped = torch.minimum(sigma_posttrain,torch.tensor(3))


# compute
def compute_cov_mat_for_pixel(i,j, n_t):
    # Compute the temporal covariance matrices for a specific pixel
    # i = row nr, j = col nr
    cov = (sigma_posttrain_cropped[0,i,j]**2)*torch.eye(n_t)
    return cov
    
def get_averaging_length_for_pixel(i,j):
    # Compute the temporal averaging needed for sigma <= 1mm
    # i = row nr, j = col nr
    # output is in nr of interferograms
    
    cov_mat = compute_cov_mat_for_pixel(i,j, n_t)
    
    variance = torch.zeros(n_t)
    for k in range(n_t):
        variance[k] = (1/(k+1)**2) * torch.sum(cov_mat[0:k+1, 0:k+1])
    std_def = np.sqrt(variance) * np.abs(c_phase_to_def)
    averaging_length = torch.where(std_def <= 1)[0][0].item()
    
    return averaging_length
    

# iv) Iterate through pixels

averaging_lengths = torch.zeros([n_r_new,n_az_new])
for i in range(n_r_new):
    print('row nr {}'.format(i))
    for j in range(n_az_new):
        averaging_lengths[i,j] = get_averaging_length_for_pixel(i,j)
averaging_lengths_in_h = averaging_lengths/30
averaging_lengths_in_min = averaging_lengths*2

save_path = '../results_stochastic_modelling/results_bafu_stochastic_model/averaging_times_in_min_decorrelated.mat'
savemat(save_path, {'averaging_lengths_in_m' : averaging_lengths_in_min.numpy()})


fig, axs = plt.subplots(1,2, figsize = (10,5), dpi = 300)        

mu_img = axs[0].imshow(mu_posttrain[0,:,:]) 
axs[0].set_title('Phase mean (rad)')
axs[0].axis('off')
fig.colorbar(mu_img, ax=axs[0])  # Add colorbar for the first axis

sigma_img = axs[1].imshow(sigma_posttrain[0,:,:], vmin = 0, vmax = 2)  # 2  = 2pi / sqrt(12) = std uniform
axs[1].set_title('Phase standard deviation (rad)')
axs[1].axis('off')
fig.colorbar(sigma_img, ax=axs[1])  # Add colorbar for the second axis

plt.figure(2, figsize=(10, 10), dpi=300)
img = plt.imshow(averaging_lengths_in_min, vmin = 0, vmax  = 15)  # Assuming averaging_lengths_in_h is defined elsewhere
cbar = plt.colorbar(img)
cbar.set_label('Time (in minutes)')  # Set colorbar label
plt.title('Averaging times to achieve 1mm standard deviation')
plt.axis('off')
plt.show()

