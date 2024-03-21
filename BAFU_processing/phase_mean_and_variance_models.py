#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The goal of this script is to train a simple stochastic model to learn the 
relationships between explanatory variables and mean and variance of the TRI 
data. Together with a penalty favouring sparse coefficients, this allows 
detecting the most important features affecting things phase mean and variance. 
These investigations require crafting a procedure that is flexible w.r.t its 
possible inputs and agnostic to the exact nature of the explanatory variables.
For this, do the following:
    1. Definitions and imports
    2. Build auxiliary datasets
    3. Build model and guide
    4. Training via svi
    5. Plots and illustrations
  
"""



"""
    1. Definitions and imports
"""

# i) imports

import numpy as np
import pyro
import matplotlib.pyplot as plt
import torch
import copy
from scipy.io import loadmat


# ii) Load data
torch.set_default_dtype(torch.float64)
data = loadmat('../data_stochastic_modelling/data_bafu_stochastic_model/submatrix_collection_training_20x20_2023_2days.mat')
data_struct = data['submatrix_collection']
phase_vals_data = data_struct[0][0][0]
info_mats_data = data_struct[0][0][1]

# Delete_data_for easier debugging
phase_vals_data = phase_vals_data[1000:1010,:]
info_mats_data = info_mats_data[1000:1010,:,:]


# iii) Definitions

n_x = 20
n_y = 20
n_total = n_x*n_y
shape_example_data = np.transpose(info_mats_data[:,:,0].reshape([-1,n_x,n_y]), (0,2,1)).shape
n_samples = shape_example_data[0]

x = np.linspace(0,1,n_x)
y = np.linspace(0,1,n_y)

xx, yy = np.meshgrid(y,x)
x_vec = np.vstack((xx.flatten(), yy.flatten())).T

pyro.clear_param_store()



"""
    2. Build auxiliary datasets
"""


# i) Compile datasets from loaded data

# Create dictionary of types of base data information and associated indices

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

phase_mats = torch.tensor(np.transpose(phase_vals_data[:,:].reshape([-1,n_x,n_y]), (0,2,1))).double()


# ii) Dataset preparation

# basic inputs to stochastic model
class BaseData():
    def __init__(self, info_mats_data, dict_ind_base_data):
        self.n_data = info_mats_data.shape[0]
        for name, index in dict_ind_base_data.items():
            base_data = torch.tensor(np.transpose(info_mats_data[:,:,index].reshape([-1,n_x,n_y]), (0,2,1))).double()
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
            
        
base_data = BaseData(info_mats_data, dict_ind_base_data)


# full outputs of stochastic model
class FullData():
    def __init__(self, base_data, phase_mats, **optional_input):
        self.K_phase_mats = optional_input.get('K_phase_mats', None)
        if self.K_phase_mats is not None:
                variances = np.diagonal(self.K_phase_mats.detach(), axis1=1, axis2=2)
                self.aps_variance_mats = variances.reshape((n_samples, n_x, n_y))
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

full_data = FullData(base_data, phase_mats)



"""
    3. Build model and guide
"""


# i) Auxiliary definitions

list_regression_vars = ['x_mats', 'y_mats', 'z_mats', 'coherence_mats', 'aoi_mats']
n_regression_vars = len(list_regression_vars)
regression_data = base_data.extract_regression_tensor(list_regression_vars)

def mean_function(regression_data, alpha_0, alpha_1, alpha_2):
    mu_1 = alpha_0
    mu_2 = torch.einsum('ijkl, l -> ijk', regression_data, alpha_1)
    regvar_products = torch.einsum('ijkl, ijkm -> ijklm', regression_data, regression_data)
    mu_3 = torch.einsum('ijklm, lm -> ijk', regvar_products, alpha_2)
    
    mu = mu_1 + mu_2 + mu_3
    return mu


def var_function(regression_data, beta_0, beta_1, beta_2):
    sigma_1 = beta_0
    sigma_2 = torch.einsum('ijkl, l -> ijk', regression_data, beta_1)
    regvar_products = torch.einsum('ijkl, ijkm -> ijklm', regression_data, regression_data)
    sigma_3 = torch.einsum('ijklm, lm -> ijk', regvar_products, beta_2)
    
    sigma = torch.sqrt(sigma_1**2 + sigma_2**2 + sigma_3**2)
    return sigma



# iii) Stochastic model

subsample_size = 8
def model(regression_data, observations = None):
    alpha_0 = pyro.param('alpha_0', torch.ones([1]))
    alpha_1 = pyro.param('alpha_1', torch.ones([n_regression_vars]))
    alpha_2 = pyro.param('alpha_2', torch.ones([n_regression_vars, n_regression_vars]))
    
    beta_0 = pyro.param('beta_0', torch.ones([1]))
    beta_1 = pyro.param('beta_1', torch.ones([n_regression_vars]))
    beta_2 = pyro.param('beta_2', torch.ones([n_regression_vars, n_regression_vars]))
        
    with pyro.plate('batch_plate', size = n_samples, dim = -3, subsample_size = subsample_size) as ind:
        mu = mean_function(regression_data[ind,...], alpha_0, alpha_1, alpha_2)
        sigma = var_function(regression_data[ind,...], beta_0, beta_1, beta_2)
        data_dist = pyro.distributions.Normal(mu, sigma)
        
        with pyro.plate('y_plate', size = n_y, dim = -2):
            with pyro.plate('x_plate', size = n_x, dim = -1):
                phase_sample = pyro.sample('phase_sample', data_dist, obs = observations)
                
    return phase_sample

simulation_pretrain = copy.copy(model(regression_data)).reshape([subsample_size, n_x,n_y])   


# iv) Guide

def guide(regression_data, observations = None):
    pass


"""
    4. Training via svi
"""

# i) Set up training

# specifying scalar options
learning_rate = 0.01
num_epochs = 200
adam_args = {"lr" : learning_rate}

# Setting up svi
optimizer = pyro.optim.AdamW(adam_args)
elbo_loss = pyro.infer.Trace_ELBO()
svi = pyro.infer.SVI(model = model, guide = guide, optim = optimizer, loss = elbo_loss)


# ii) Execute training

train_elbo = []
for epoch in range(num_epochs):
    epoch_loss = svi.step(regression_data, phase_mats)
    train_elbo.append(-epoch_loss)
    if epoch % 10 == 0:
        print("Epoch : {} train loss : {}".format(epoch, epoch_loss))


# iii) Simulation posttraining

simulation_posttrain = copy.copy(model(regression_data)).reshape([subsample_size, n_x,n_y])



"""
    5. Plots and illustrations
"""


















