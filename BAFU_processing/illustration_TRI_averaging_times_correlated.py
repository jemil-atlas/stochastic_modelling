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
n_t = 100
n_t_h = n_t/30
time = torch.linspace(0,n_t_h, n_t)
c_phase_to_def = (-17.4 / (4* np.pi)) # result in mm


# iii) Load data
# data = loadmat('../data_stochastic_modelling/data_bafu_stochastic_model/submatrix_collection_training_20x20_2023_2days.mat')
data = loadmat('../data_stochastic_modelling/data_bafu_stochastic_model/Single_Full_Info_mat.mat')
data_SLC_1 = loadmat('../data_stochastic_modelling/data_bafu_stochastic_model/SLC_1.mat')['SLC_1']
data_SLC_2 = loadmat('../data_stochastic_modelling/data_bafu_stochastic_model/SLC_2.mat')['SLC_2']

info_mats = data['Info_mat']

# data_mean = info_mats.mean(axis=(0, 1), keepdims=True)
# data_std = info_mats.std(axis=(0, 1), keepdims=True)
# info_mats_standardized = (info_mats - data_mean) / (data_std + 1e-6)


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
    # info_mats_reduced[:,:,k] = A_r @ info_mats_standardized[:,:,k] @ A_az
    info_mats_reduced[:,:,k] = A_r @ info_mats[:,:,k] @ A_az
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


# iii) Load correlated model
model_save_path = '../results_stochastic_modelling/results_bafu_stochastic_model/ts_model.pth'
# model_save_path = '../results_stochastic_modelling/results_bafu_stochastic_model/phase_mean_and_variance_models_2024-03-23.pth'

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


# full outputs of stochastic model
class FullData():
    def __init__(self, base_data, phase_mats, **optional_input):
        # load nr of samples
        self.n_samples = optional_input.get('n_samples', None)
        # load the variance scaling matrices
        self.A_mats = optional_input.get('A_mats', None)
        self.B_mats = optional_input.get('B_mats', None)
        # load the covariance matrices
        self.K_noise_mats = optional_input.get('K_noise_mats', None)
        self.K_smooth_mats = optional_input.get('K_smooth_mats', None)
        self.cov_mats_noise = optional_input.get('cov_mats_noise', None)
        if self.cov_mats_noise is not None:
                noise_variances = np.diagonal(self.cov_mats_noise.detach(), axis1=1, axis2=2)
                self.noise_variance_mats = noise_variances.reshape((self.n_samples, n_t))
        self.cov_mats_smooth = optional_input.get('cov_mats_smooth', None)
        if self.cov_mats_smooth is not None:
                smooth_variances = np.diagonal(self.cov_mats_smooth.detach(), axis1=1, axis2=2)
                self.smooth_variance_mats = smooth_variances.reshape((self.n_samples, n_t))
        # List distributional_data
        def get_distributional_attr_list(self):
            attributes = [attr for attr in dir(self) if not (attr.startswith('__') )]
            return attributes
        self.distributional_attribute_list = get_distributional_attr_list(self)
        self.num_distributional_attributes = len(self.distributional_attribute_list)
        # copy from base_data
        for name, attribute in base_data.__dict__.items():
            setattr(self, name, attribute)
        # Add simulations
        self.phase_mats = phase_mats            

phase_mats = 0
full_data = FullData(base_data, phase_mats)

  
"""
    4. Build stochastic model -------------------------------------------------
"""

# i) Auxiliary function definitions

# Map into high dim space
class ANNMap(pyro.nn.PyroModule):
    # Initialize the module
    def __init__(self, dim_input, dim_hidden, dim_output):
        # Evoke by passing input, hidden, and output dimensions
        # Initialize instance using init method from base class
        super().__init__()
        self.dim_input = dim_input
        
        # linear transforms
        self.fc_1 = torch.nn.Linear(dim_input, dim_hidden)
        self.fc_2 = torch.nn.Linear(dim_hidden, dim_hidden)
        self.fc_3 = torch.nn.Linear(dim_hidden, dim_output)
        # nonlinear transforms
        self.nonlinear = torch.nn.Sigmoid()
        
    def forward(self, x, atypical_shape = False):
        # Define forward computation on the input data x
        if atypical_shape == False :
            # Shape the minibatch so that batch_dims are on left, argument_dims on right
            x = x.reshape([-1, x.shape[1], self.dim_input])
        else:
            # Shape the minibatch for processing of a vector of inputs
            x = x.reshape([-1, x.shape[1], self.dim_input])
        
        # Then compute hidden units and output of nonlinear pass
        hidden_units_1 = self.nonlinear(self.fc_1(x))
        hidden_units_2 = self.nonlinear(self.fc_2(hidden_units_1))
        ann_output = self.fc_3(hidden_units_2)
        return ann_output



# ii) Covariance function definition

def construct_cov_mats(ann_map, coh_mats, t_mats):
        
    # fuse to input and compute coefficients
    input_mats = torch.cat((coh_mats.unsqueeze(2), t_mats.unsqueeze(2)), dim=2)
    feature_mats = ann_map(input_mats.float())
    n_rows, n_cols, n_output = feature_mats.shape
    A_mats = torch.zeros([n_rows, n_cols, n_cols])
    B_mats = torch.zeros([n_rows, n_cols, n_cols])
    for k in range(n_rows):
        A_mats[k] = torch.diag(feature_mats[k,:,0])
        B_mats[k] = torch.diag(feature_mats[k,:,1])
    
    # compute basis matrices
    K_noise_mats = torch.eye(n_cols).repeat(n_rows,1,1)
    
    K_smooth_mats = torch.zeros([n_rows, n_cols, n_cols])
    corr_lengths = pyro.param('corr_lengths', init_tensor = 1*torch.ones([2]), constraint = pyro.distributions.constraints.positive)
    rel_scales = pyro.param('scales', init_tensor = 0.5*torch.ones([2]), constraint = pyro.distributions.constraints.simplex)
    def kernel_fun_sqexp(v_1,v_2):
        # Calculate distance matrix dist_mat
        v_1_reshaped = v_1.unsqueeze(1)  # shape becomes [n_total, 1]
        v_2_reshaped = v_2.unsqueeze(0)  # shape becomes [1, n_total]
        # corr_lengths_reshaped = corr_lengths.unsqueeze(0)
        dist_mat = torch.abs((v_1_reshaped - v_2_reshaped) / corr_lengths[0])
        # squared exponential kernel function
        cov_mat = torch.exp(-dist_mat**2)
        return cov_mat
    
    def kernel_fun_exp(v_1,v_2):
        # Calculate distance matrix dist_mat
        v_1_reshaped = v_1.unsqueeze(1)  # shape becomes [n_total, 1]
        v_2_reshaped = v_2.unsqueeze(0)  # shape becomes [1, n_total]
        # corr_lengths_reshaped = corr_lengths.unsqueeze(0)
        dist_mat = torch.abs((v_1_reshaped - v_2_reshaped) / corr_lengths[1])
        # exponential kernel function
        cov_mat = torch.exp(-dist_mat**1)
        return cov_mat
        
    for k in range(n_rows):
        K_smooth_mats[k,:,:] = rel_scales[0]*kernel_fun_sqexp(t_mats[k,:], t_mats[k,:]) \
                                + rel_scales[1]*kernel_fun_exp(t_mats[k,:], t_mats[k,:])
        
    return K_noise_mats, K_smooth_mats, A_mats, B_mats
        

# iii) Stochastics class
subsample_size = 64

class TRIStochastics(pyro.nn.PyroModule):
    # Initialize the module
    def __init__(self, list_inputs, dim_hidden, base_data):
        super().__init__()  # Initialize the PyroModule superclass (otherwise module registration fails)
        # list_inputs contains a list of names of attributes that are used as 
        # inputs to construct a covariance function. Names must be attributes of
        # base_data class.
        self.dim_input = 2
        self.dim_hidden = dim_hidden
        self.dim_output = 2
        
        # combine base_data inputs
        self.list_inputs = list_inputs
        self.integrate_base_data(self.list_inputs, base_data)                       
        self.base_data = base_data
        self.ann_map = ANNMap(self.dim_input, self.dim_hidden, self.dim_output)
        
    
    # Model
    def model(self, base_data, observations = None, subsample_size = subsample_size):
        # integrate different base_data
        self.integrate_base_data(self.list_inputs,base_data)       
        # print("Samples:{}".format(self.n_samples))      
        
        with pyro.plate('batch_plate',size = self.n_samples, dim = -1, subsample_size = subsample_size) as ind:
            K_noise_mats, K_smooth_mats, A_mats, B_mats = construct_cov_mats(self.ann_map, 
                                    self.base_data.coherence_mats[ind,:], self.base_data.time_mats[ind,:]) 
            cov_regularizer = 1e-6*(torch.eye(n_t).repeat(subsample_size, 1, 1))
            cov_mats_noise_temp = torch.bmm(A_mats, K_noise_mats)
            cov_mats_noise = torch.bmm(cov_mats_noise_temp, A_mats.transpose(1,2))
            cov_mats_smooth_temp = torch.bmm(B_mats, K_smooth_mats)
            cov_mats_smooth = torch.bmm(cov_mats_smooth_temp, B_mats.transpose(1,2))   
            subsampled_observations = observations[ind] if observations is not None else None
            obs_dist = pyro.distributions.MultivariateNormal(loc = torch.zeros([subsample_size, n_t]), covariance_matrix = cov_mats_smooth + cov_mats_noise + cov_regularizer)
            # print("batch_shape : {}, event_shape : {}".format(obs_dist.batch_shape, obs_dist.event_shape))
            obs = pyro.sample("obs", obs_dist, obs = subsampled_observations)
        # data = FullData(base_data, obs.reshape([self.n_samples, n_x,n_y]).detach(), K_phase_mats = cov_mats)
        data = FullData(base_data, obs.reshape([subsample_size, n_t]).detach(), 
                        **{'K_noise_mats' : K_noise_mats, 'K_smooth_mats' : K_smooth_mats, \
                           'cov_mats_noise' : cov_mats_noise, 'cov_mats_smooth' : cov_mats_smooth, \
                           'A_mats' : A_mats, 'B_mats' : B_mats, 'n_samples' : len(ind)})
        return obs, data
    
    # eval Model
    def eval_model(self, base_data, i,j):
        # integrate different base_data
        self.integrate_base_data(self.list_inputs,base_data)       
        # print("Samples:{}".format(self.n_samples))      
        
        with pyro.plate('eval_batch_plate',size = 1, dim = -1):
            K_noise_mats, K_smooth_mats, A_mats, B_mats = construct_cov_mats(self.ann_map, 
                                    torch.repeat_interleave(base_data.coherence_mats[0,i,j],n_t).unsqueeze(0), 
                                    time.unsqueeze(0)) 
            cov_regularizer = 1e-6*(torch.eye(n_t).repeat(1, 1, 1))
            cov_mats_noise_temp = torch.bmm(A_mats, K_noise_mats)
            cov_mats_noise = torch.bmm(cov_mats_noise_temp, A_mats.transpose(1,2))
            cov_mats_smooth_temp = torch.bmm(B_mats, K_smooth_mats)
            cov_mats_smooth = torch.bmm(cov_mats_smooth_temp, B_mats.transpose(1,2))   
            obs_dist = pyro.distributions.MultivariateNormal(loc = torch.zeros([1, n_t]), covariance_matrix = cov_mats_smooth + cov_mats_noise + cov_regularizer)
            obs = pyro.sample("obs", obs_dist)
        
        data = FullData(base_data, obs.reshape([1, n_t]).detach(), 
                        **{'K_noise_mats' : K_noise_mats, 'K_smooth_mats' : K_smooth_mats, \
                           'cov_mats_noise' : cov_mats_noise, 'cov_mats_smooth' : cov_mats_smooth, \
                           'A_mats' : A_mats, 'B_mats' : B_mats, 'n_samples' : 1})
        return obs, data
    
    def integrate_base_data(self, list_inputs, base_data):
        self.n_samples = base_data.range_mats.shape[0]
        self.list_base_data_mats = [getattr(base_data, input_attr).reshape([self.n_samples, n_pixels, -1]) for input_attr in list_inputs]
        self.base_data_mats = torch.cat(self.list_base_data_mats, dim = -1)
        self.dim_arg = self.base_data_mats.shape[-1]
        
    # Guide
    def guide(self, base_data, observations = None):
        pass

list_inputs = ['coherence_mats', 'time_mats' ]

tri_stochastics = TRIStochastics(list_inputs, 100, base_data)
tri_stochastics.load_state_dict(torch.load(model_save_path))

simulation_posttrain, full_data_posttrain = copy.copy(tri_stochastics.eval_model(base_data, 0,0))
simulation_posttrain = simulation_posttrain.reshape([-1, n_t]) 



# compute
def compute_cov_mat_for_pixel(i,j, n_t):
    # Compute the temporal covariance matrices for a specific pixel
    # i = row nr, j = col nr
    simulation_temp, full_data_temp = copy.copy(tri_stochastics.eval_model(base_data, i,j))
    cov = full_data_temp.cov_mats_noise + full_data_temp.cov_mats_smooth
    return cov
    
def get_averaging_length_for_pixel(i,j):
    # Compute the temporal averaging needed for sigma <= 1mm
    # i = row nr, j = col nr
    # output is in nr of interferograms
    
    cov_mat = compute_cov_mat_for_pixel(i,j, n_t).detach()
    cov_mat_cropped = torch.minimum(cov_mat, torch.tensor(3))
    
    variance = torch.zeros(n_t)
    for k in range(n_t):
        variance[k] = (1/(k+1)**2) * torch.sum(cov_mat_cropped[0,0:k+1, 0:k+1])
    std_def = np.sqrt(variance) * np.abs(c_phase_to_def)
    averaging_length = torch.where(std_def <= 1)[0][0].item()
    
    return averaging_length
    

# iv) Iterate through pixels

averaging_lengths = torch.zeros([n_r_new,n_az_new])
for i in range(n_r_new):
    print('row nr {}'.format(i))
    for j in range(n_az_new):
        averaging_lengths[i,j] = get_averaging_length_for_pixel(j,i)
averaging_lengths_in_h = averaging_lengths/30
averaging_lengths_in_min = averaging_lengths*2

save_path = '../results_stochastic_modelling/results_bafu_stochastic_model/averaging_times_in_min_correlated.mat'
savemat(save_path, {'averaging_lengths_in_m' : averaging_lengths_in_min.numpy()})

plt.figure(2, figsize=(10, 10), dpi=300)
img = plt.imshow(averaging_lengths_in_min, vmin = 0, vmax  = 15)  # Assuming averaging_lengths_in_h is defined elsewhere
cbar = plt.colorbar(img)
cbar.set_label('Time (in minutes)')  # Set colorbar label
plt.title('Averaging times to achieve 1mm standard deviation')
plt.axis('off')
plt.show()

