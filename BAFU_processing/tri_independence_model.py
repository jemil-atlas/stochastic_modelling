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
from datetime import date


# ii) Load data
torch.set_default_dtype(torch.float64)
# data = loadmat('../data_stochastic_modelling/data_bafu_stochastic_model/submatrix_collection_training_20x20_2023_2days.mat')
data = loadmat('../data_stochastic_modelling/data_bafu_stochastic_model/submatrix_collection_training_20x20_2023_mli_2days.mat')
data_struct = data['submatrix_collection']
# phase_vals_data = data_struct[0][0][0]
phase_vals_data = np.angle(data_struct[0][0][0])
info_mats_data = data_struct[0][0][1]

# Delete_data_for easier debugging
nice_patch_unw = np.linspace(1000, 1010,11).astype(int)
nice_patch_mli = np.linspace(1810, 1820,11).astype(int)
many_patch_mli = np.linspace(1810, 2000,191).astype(int)
all_patch_mli = np.linspace(0,19999,20000).astype(int)

# iii) Data preselection

n_illu = 4
patch_choice = all_patch_mli
illustration_choice = np.round(np.linspace(0, len(patch_choice)-1, n_illu)).astype(int)
phase_vals_data = phase_vals_data[patch_choice,:]
info_mats_data = info_mats_data[patch_choice,:,:]

# Data normalization
data_mean = info_mats_data.mean(axis=(0, 1), keepdims=True)
data_std = info_mats_data.std(axis=(0, 1), keepdims=True)
info_mats_data = (info_mats_data - data_mean) / (data_std + 1e-6)


# iv) Definitions

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
today = date.today()



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
        self.phase_mats_illu = phase_mats[illustration_choice,:]

full_data = FullData(base_data, phase_mats)



"""
    3. Build model and guide
"""


# i) Auxiliary definitions

list_regression_vars = ['x_mats', 'y_mats', 'z_mats', 'coherence_mats', 'aoi_mats']
# list_regression_vars = ['coherence_mats']
n_regression_vars = len(list_regression_vars)
regression_data = base_data.extract_regression_tensor(list_regression_vars)

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


# iii) Stochastic model

subsample_size = 512
simplicity = False

def model(regression_data, observations = None):
    # Instantiate parameters
    alpha_0 = pyro.param('alpha_0', torch.zeros([1]))
    alpha_1 = pyro.param('alpha_1', torch.zeros([n_regression_vars]))
    alpha_2 = pyro.param('alpha_2', torch.zeros([n_regression_vars, n_regression_vars]))
    
    beta_0 = pyro.param('beta_0', 1*torch.ones([1]))
    beta_1 = pyro.param('beta_1', 0*torch.ones([n_regression_vars]))
    beta_2 = pyro.param('beta_2', 0*torch.ones([n_regression_vars, n_regression_vars]))
        
    # Determine distribution and sample from it
    with pyro.plate('batch_plate', size = n_samples, dim = -3, subsample_size = subsample_size) as ind:
        mu = mean_function(regression_data[ind,...], alpha_0, alpha_1, alpha_2, simplicity = simplicity)
        sigma = var_function(regression_data[ind,...], beta_0, beta_1, beta_2, simplicity = simplicity)
        data_dist = pyro.distributions.Normal(mu, sigma)
        subsampled_observations = observations[ind,...] if observations is not None else None
        
        with pyro.plate('y_plate', size = n_y, dim = -2):
            with pyro.plate('x_plate', size = n_x, dim = -1):
                phase_sample = pyro.sample('phase_sample', data_dist, obs = subsampled_observations)
                
    # Optional regularization
    reg_coeff = 0.1
    l1_norm_reg = torch.linalg.norm(alpha_1, ord = 1) + torch.linalg.norm(alpha_2, ord = 1) \
        + torch.linalg.norm(beta_1, ord = 1) + torch.linalg.norm(beta_2, ord = 1)
    pyro.factor('l1_reg', -reg_coeff * l1_norm_reg)
    
    return phase_sample, mu, sigma

pyro.render_model(model, model_args=(regression_data,), render_distributions=True, render_params=True)


# iv) Guide and eval model

def guide(regression_data, observations = None):
    pass

def eval_model(regression_data):
    # Instantiate parameters
    alpha_0 = pyro.get_param_store()['alpha_0'].detach()
    alpha_1 = pyro.get_param_store()['alpha_1'].detach()
    alpha_2 = pyro.get_param_store()['alpha_2'].detach()
    
    beta_0 = pyro.get_param_store()['beta_0'].detach()
    beta_1 = pyro.get_param_store()['beta_1'].detach()
    beta_2 = pyro.get_param_store()['beta_2'].detach()
        
    # Determine distribution and sample from it
    with pyro.plate('batch_plate_eval', size = n_illu, dim = -3):
        mu = mean_function(regression_data, alpha_0, alpha_1, alpha_2, simplicity = simplicity)
        sigma = var_function(regression_data, beta_0, beta_1, beta_2, simplicity = simplicity)
        data_dist = pyro.distributions.Normal(mu, sigma)
        with pyro.plate('y_plate_eval', size = n_y, dim = -2):
            with pyro.plate('x_plate_eval', size = n_x, dim = -1):
                phase_sample = pyro.sample('phase_sample_eval', data_dist)
    
    return phase_sample, mu, sigma

simulation_pretrain, mu_pretrain, sigma_pretrain = copy.copy(eval_model(regression_data[illustration_choice,...]))
simulation_pretrain = simulation_pretrain.reshape([-1, n_x,n_y]) 

pyro.render_model(model, model_args=(regression_data,), render_distributions=True, render_params=True)



"""
    4. Training via svi
"""

# i) Set up training

# specifying scalar options
learning_rate = 3*1e-5
num_epochs = 15000
adam_args = {"lr" : learning_rate}

# Setting up svi
optimizer = pyro.optim.NAdam(adam_args)
elbo_loss = pyro.infer.Trace_ELBO()
svi = pyro.infer.SVI(model = model, guide = guide, optim = optimizer, loss = elbo_loss)


# ii) Execute training

model_save_path = '../results_stochastic_modelling/results_bafu_stochastic_model/phase_mean_and_variance_models_{}.pth'.format(today)
best_loss = float('inf')
train_elbo = []
for epoch in range(num_epochs):
    epoch_loss = svi.step(regression_data, phase_mats)
    train_elbo.append(epoch_loss)
    
    # Check if the current model is better than what we've seen before
    if epoch_loss < 0.99*best_loss:
        best_loss = epoch_loss
        # Save the model
        pyro.get_param_store().save(model_save_path)
        print(f"Saved the model at epoch {epoch} with loss {best_loss}")
    
    if epoch % 100 == 0:
        print("Epoch : {} train loss : {}".format(epoch, epoch_loss))

# Load best parameters
pyro.get_param_store().load(model_save_path)


# iii) Simulation posttraining

simulation_posttrain, mu_posttrain, sigma_posttrain = copy.copy(eval_model(regression_data[illustration_choice,...]))
simulation_posttrain = simulation_posttrain.reshape([-1, n_x,n_y])



"""
    5. Plots and illustrations
"""


# i) Illustrate training and data

fig = plt.figure(1, dpi = 300)
plt.plot(train_elbo)
plt.title('Training loss')
plt.xlabel('Step nr')
plt.ylabel('Loss')


fig, axs = plt.subplots(n_illu, 4, figsize=(15, 15))

attributes = [full_data.phase_mats, full_data.coherence_mats, full_data.z_mats, full_data.aoi_mats]
titles = ["Phase Mats", "Coherence Mats", "Elevation Mats", "AOI Mats"]

for i in range(n_illu):  # loop over samples
    for j in range(4):  # loop over attributes
        ax = axs[i, j]
        attribute = attributes[j]
        ax.imshow(attribute[illustration_choice[i], :, :])
        if i == 0:
            ax.set_title(titles[j])

        ax.axis('off')

plt.tight_layout()
plt.show()


# ii) Illustrate data, initial simulations, trained simulations

fig, axs = plt.subplots(n_illu, 3, figsize=(9, 15))

attributes = [full_data.phase_mats_illu, simulation_pretrain.detach(), simulation_posttrain.detach()]
titles = ["Phase: data", "pretrain", " posttrain"]

for i in range(n_illu):  # loop over samples
    for j in range(3):  # loop over attributes
        ax = axs[i, j]
        attribute = attributes[j]
        ax.imshow(attribute[i, :, :], vmin = -3, vmax = 3)
        if i == 0:
            ax.set_title(titles[j])

        ax.axis('off')

plt.tight_layout()
plt.show()


# iii) Illustrate mean and variance

def illustrate_mean_and_var(image_nr):
    
    fig, axs = plt.subplots(2, 5, figsize=(10, 5))
    
    axs[0,0].imshow(full_data.coherence_mats[illustration_choice[image_nr],:,:])
    axs[0,1].imshow(full_data.x_mats[illustration_choice[image_nr],:,:])
    axs[0,2].imshow(full_data.y_mats[illustration_choice[image_nr],:,:])
    axs[0,3].imshow(full_data.z_mats[illustration_choice[image_nr],:,:])
    axs[0,4].imshow(full_data.aoi_mats[illustration_choice[image_nr],:,:])
    
    axs[1,1].imshow(mu_pretrain[image_nr,:,:].detach())
    axs[1,2].imshow(sigma_pretrain[image_nr,:,:].detach())
    axs[1,3].imshow(mu_posttrain[image_nr,:,:].detach())
    axs[1,4].imshow(sigma_posttrain[image_nr,:,:].detach())
    
    titles = ["Coherence", "x values", "y values", "z values", "AOI", "", \
              "mu pretrain", "sigma pretrain" , "mu posttrain" , "sigma posttrain"]
    
    for i in range(2):  # loop over samples
        for j in range(5):  # loop over attributes
            ax = axs[i,j]
            ax.set_title(titles[5*i+j])
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()

illustrate_mean_and_var(0)
illustrate_mean_and_var(1)
illustrate_mean_and_var(2)
illustrate_mean_and_var(3)


# iv) Illustrate the parameters

param_names_1 = ['offset', 'x', 'y', 'z', 'coh', 'aoi']
param_names_2 = ['x', 'y', 'z', 'coh', 'aoi']

param_dict = {}
for param, value in pyro.get_param_store().items():
    param_dict[param] = value

fig, axs = plt.subplots(2, 2, figsize=(13, 10))
# alpha values
x_values_1 = torch.linspace(0,n_regression_vars,n_regression_vars+1)
x_values_2 = x_values_1[0:-1]
axs[0,0].bar(x_values_1,
             torch.cat((param_dict['alpha_0'].detach().reshape([1]),
                        param_dict['alpha_1'].detach())))
axs[0,0].set_xlabel('alpha')
axs[0,0].set_xticks(x_values_1)
axs[0,0].set_xticklabels(param_names_1)
axs[0,0].set_ylabel('param value')
axs[0,0].set_title('Parameter values alpha_0, alpha_1')

im_alpha_2 = axs[0,1].imshow(param_dict['alpha_2'].detach())
axs[0,1].set_xlabel('alpha_2_ij')
axs[0,1].set_xticks(x_values_2)
axs[0,1].set_xticklabels(param_names_2)
axs[0,1].set_ylabel('alpha_2_ij')
axs[0,1].set_yticks(x_values_2)
axs[0,1].set_yticklabels(param_names_2)
axs[0,1].set_title('Parameter values alpha_2')
plt.colorbar(im_alpha_2, ax=axs[0,1])

# beta_values
axs[1,0].bar(torch.linspace(0,n_regression_vars,n_regression_vars+1),
             torch.cat((param_dict['beta_0'].detach().reshape([1]),
                        param_dict['beta_1'].detach())))
axs[1,0].set_xlabel('beta')
axs[1,0].set_xticks(x_values_1)
axs[1,0].set_xticklabels(param_names_1)
axs[1,0].set_ylabel('param value')
axs[1,0].set_title('Parameter values beta_0, beta_1')

im_beta_2 = axs[1,1].imshow(param_dict['beta_2'].detach())
axs[1,1].set_xlabel('beta_2_ij')
axs[1,1].set_xticks(x_values_2)
axs[1,1].set_xticklabels(param_names_2)
axs[1,1].set_ylabel('beta_2_ij')
axs[1,1].set_yticks(x_values_2)
axs[1,1].set_yticklabels(param_names_2)
axs[1,1].set_title('Parameter values beta_2')
plt.colorbar(im_beta_2, ax=axs[1,1])

plt.tight_layout()
plt.show()


# v) Illustrate mean and variance





