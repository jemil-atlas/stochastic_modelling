#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Produces images and models for the "Spatial model" section in the BAFU Report.
The goal of this script is to train a stochastic model of TRI data based on pyro's
svi formalism and real radar data from breithorn. The model consists in a multivariate
gaussian distribution whose mean and kernel parameters depend on auxiliary data.
It predicts the spatial covariance matrices for image patches based on a kernel
learned from 20x20 image patches and associated x, y, z, coherence, aoi explanatory data.
We want to train the model using pyro-ppl and identify potential questions and how
to answer them using more flexible stochastic models and inference on them.
For this, do the following:
    1. Imports and definitions
    2. Build auxiliary datasets
    3. Simulate data
    4. Build stochastic model
    5. Inference
    6. Plots and illustrations
    
"""


"""
    1. Imports and definitions
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
patch_choice = nice_patch_mli
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
        
    def __extract_regression_tensor(self, list_regression_vars):
        
        regvar_list = []
        for regvar in list_regression_vars:
            regvar_list.append(getattr(self,regvar))
        regression_tensor = torch.stack(regvar_list, dim = 3)
        return regression_tensor
        
base_data = BaseData(info_mats_data, dict_ind_base_data)


# full outputs of stochastic model
class FullData():
    def __init__(self, base_data, phase_mats, **optional_input):
        self.K_phase_mats = optional_input.get('K_phase_mats', None)
        if self.K_phase_mats is not None:
                n_samples = self.K_phase_mats.shape[0]
                variances = np.diagonal(self.K_phase_mats.detach(), axis1=1, axis2=2)
                covariances =  np.hstack((np.diagonal(self.K_phase_mats.detach(), 
                        offset = 1, axis1=1, axis2=2), np.zeros([n_samples,1])))
                covariances = variances.reshape((n_samples, n_x, n_y))
                covariances = covariances[:,:,:-1]
                self.aps_variance_mats = variances.reshape((n_samples, n_x, n_y))
                self.covariances = covariances
                
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
        # self.phase_mats_illu = phase_mats[illustration_choice,:]

full_data = FullData(base_data, phase_mats)

    
    
"""
    4. Build stochastic model
"""

# i) Auxiliary function definitions

# Map into high dim space
class MapToL2(pyro.nn.PyroModule):
    # Initialize the module
    def __init__(self, dim_arg, dim_hidden, dim_l2):
        # Evoke by passing argument, hidden, and l2 dimensions
        # Initialize instance using init method from base class
        super().__init__()
        self.dim_arg = dim_arg
        
        # linear transforms
        self.fc_1 = torch.nn.Linear(dim_arg, dim_hidden)
        self.fc_2 = torch.nn.Linear(dim_hidden, dim_hidden)
        self.fc_3 = torch.nn.Linear(dim_hidden, dim_l2)
        # nonlinear transforms
        self.nonlinear = torch.nn.Sigmoid()
        
    def forward(self, x):
        # Define forward computation on the input data x
        # Shape the minibatch so that batch_dims are on left, argument_dims on right
        x = x.reshape([-1, n_total, self.dim_arg])
        
        # Then compute hidden units and output of nonlinear pass
        hidden_units_1 = self.nonlinear(self.fc_1(x))
        hidden_units_2 = self.nonlinear(self.fc_2(hidden_units_1))
        feature_mat = self.fc_3(hidden_units_2)
        return feature_mat



# ii) Covariance function definition

def covariance_function(map_to_l2, base_data_mats):
        
    # compute covariance
    feature_mats = map_to_l2(base_data_mats)
    feature_mats_T = feature_mats.permute(0,2,1)
    cov_mats = torch.bmm(feature_mats, feature_mats_T)
    # eigenvalues = torch.linalg.eigvalsh(cov_mats[0,:,:])
    # print("Minimum eigenvalue:", torch.min(eigenvalues))
    return cov_mats
        

# iii) Stochastics class
subsample_size = 8

class TRIStochastics(pyro.nn.PyroModule):
    # Initialize the module
    def __init__(self, list_inputs, dim_hidden, dim_l2, base_data):
        super().__init__()  # Initialize the PyroModule superclass (otherwise module registration fails)
        # list_inputs contains a list of names of attributes that are used as 
        # inputs to construct a covariance function. Names must be attributes of
        # base_data class.
        self.dim_hidden = dim_hidden
        self.dim_l2 = dim_l2
        
        # combine base_data inputs
        self.list_inputs = list_inputs
        self.integrate_base_data(self.list_inputs,base_data)                       
        self.base_data = base_data
        self.map_to_l2 = MapToL2(self.dim_arg, self.dim_hidden, self.dim_l2)
    
    # Model
    def model(self, base_data, observations = None):
        # integrate different base_data
        self.integrate_base_data(self.list_inputs,base_data)       
        # register the map_to_l2 parameters with pyro
        pyro.module("map_to_l2", self.map_to_l2)        
        
        with pyro.plate('batch_plate',size = self.n_samples, dim = -1, subsample_size = subsample_size) as ind:
            cov_mats = covariance_function(self.map_to_l2, self.base_data_mats[ind,:,:,:]) 
            cov_regularizer = 1e-3*(torch.eye(n_total).repeat(subsample_size, 1, 1))
            subsampled_observations = observations[ind] if observations is not None else None
            obs_dist = pyro.distributions.MultivariateNormal(loc = torch.zeros([subsample_size, n_total]), covariance_matrix = cov_mats + cov_regularizer)
            obs = pyro.sample("obs", obs_dist, obs = subsampled_observations)
            
        data = FullData(base_data, obs.reshape([subsample_size, n_x,n_y]).detach())
        return obs, data
    
    # Evaluation Model
    def eval_model(self, base_data, illustration_choice):
        # integrate different base_data
        self.integrate_base_data(self.list_inputs, base_data)   
        self.illustration_choice = illustration_choice
        self.n_illu = len(self.illustration_choice)
        self.base_data_mats_illustration = self.base_data_mats[illustration_choice,...]
        # register the map_to_l2 parameters with pyro
        pyro.module("map_to_l2", self.map_to_l2)        
        
        with pyro.plate('batch_plate',size = self.n_illu, dim = -1):
            cov_mats = covariance_function(self.map_to_l2, self.base_data_mats_illustration) 
            cov_regularizer = 1e-3*(torch.eye(n_total).repeat(self.n_illu, 1, 1))
            obs_dist = pyro.distributions.MultivariateNormal(loc = torch.zeros([self.n_illu, n_total]), covariance_matrix = cov_mats + cov_regularizer)
            obs = pyro.sample("obs", obs_dist)
            
        data = FullData(base_data, obs.reshape([self.n_illu, n_x,n_y]).detach(), **{'K_phase_mats' : cov_mats})
        return obs, data
    
    def integrate_base_data(self, list_inputs, base_data):
        self.n_samples = base_data.range_mats.shape[0]
        self.list_base_data_mats = [getattr(base_data, input_attr).reshape([self.n_samples, n_x,n_y, -1]) for input_attr in list_inputs]
        self.base_data_mats = torch.cat(self.list_base_data_mats, dim = -1)
        self.dim_arg = self.base_data_mats.shape[-1]
        
    # Guide
    def guide(self, base_data, observations = None):
        pass

list_inputs = ['x_mats', 'y_mats', 'z_mats', 'aoi_mats', 'coherence_mats' ]
# list_inputs = ['x_mats', 'y_mats' ]
# list_inputs = ['coherence_mats' ]
tri_stochastics = TRIStochastics(list_inputs, 50, 20, base_data)
observations = full_data.phase_mats.reshape([n_samples, -1])

pyro.render_model(tri_stochastics.model, model_args=(base_data,), render_distributions=True, render_params=True)


# iv) Simulation pretraining

simulation_pretrain, full_data_pretrain = copy.copy(tri_stochastics.eval_model(base_data, illustration_choice))
simulation_pretrain = copy.copy(simulation_pretrain.reshape([n_illu, n_x,n_y]))



"""
    5. Inference
"""


# i) Set up training

# specifying scalar options
learning_rate = 0.1
num_epochs = 100
adam_args = {"lr" : learning_rate}

# Setting up svi
optimizer = pyro.optim.AdamW(adam_args)
elbo_loss = pyro.infer.Trace_ELBO()
svi = pyro.infer.SVI(model = tri_stochastics.model, guide = tri_stochastics.guide, optim = optimizer, loss= elbo_loss)


# ii) Execute training

model_save_path = '../results_stochastic_modelling/results_bafu_stochastic_model/spatial_model_{}.pth'.format(today)
best_loss = float('inf')
train_elbo = []
for epoch in range(num_epochs):
    epoch_loss = svi.step(base_data, phase_mats.reshape([-1, n_total]))
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

simulation_posttrain, full_data_posttrain = copy.copy(tri_stochastics.eval_model(base_data, illustration_choice))
simulation_posttrain = copy.copy(simulation_posttrain.reshape([n_illu, n_x,n_y]))



"""
    6. Plots and illustrations
"""


# i) Illustrate training and data

fig = plt.figure(1, dpi = 300)
plt.plot(train_elbo)
plt.title('Training loss')
plt.xlabel('Step nr')
plt.ylabel('Loss')


fig, axs = plt.subplots(n_illu, 4, figsize=(15, 3*n_illu))

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

fig, axs = plt.subplots(n_illu, 3, figsize=(9, 3*n_illu))

attributes = [full_data.phase_mats[illustration_choice,...], simulation_pretrain.detach(), simulation_posttrain.detach()]
titles = ["Phase: data", "pretrain", " posttrain"]

for i in range(n_illu):  # loop over samples
    for j in range(3):  # loop over attributes
        ax = axs[i, j]
        attribute = attributes[j]
        ax.imshow(attribute[i, :, :])
        if i == 0:
            ax.set_title(titles[j])
        ax.axis('off')
plt.tight_layout()
plt.show()


# iii) Plot variance maps, correlation maps, covariance matrices

def plot_covariance_info(base_data, full_data):
    pass


# Invoke the plots for the illustration_choice

# Invoke the plots for some fake data to gauge impact of explanatory variables
# We 


# # iii) Multiple realizations for same base_data

# def plot_realizations(base_data, full_data, **plotting_opts):
#     # This function plots the base data and a bunch of realizations. This is used
#     # to compare the realizations of the trained model and the true underlying model.
#     # plotting opts include the 'n_realizations', the number of realizations to
#     # be shown during plotting and 'scenario_index' the index of the scenario
#     # that is shown (i.e. determines the randomly chosen base_data like elevation etc)
    
#     # Fetch basic & distributional data
#     base_attributes = [getattr(base_data,attr) for attr in base_data.basic_attribute_list]
#     base_attribute_names = [attr for attr in base_data.basic_attribute_list]
#     n_base_data = base_data.num_basic_attributes
        
#     n_realizations = plotting_opts.get('n_illu', 4)
#     scenario_index = plotting_opts.get('scenario_index', 0)


#     # Support function    
#     def eval_attribute_dims(attribute, scenario_index):
#         attribute_dim = (attribute[0,:,:]).shape
#         n_attribute_dim = len(attribute_dim)
#         if n_attribute_dim ==3:
#             image = attribute[scenario_index,:,:,0]
#             message = " 1 of {} matrices".format(attribute_dim[2])
#         else:
#             image = attribute[scenario_index,:,:]
#             message = ""
#         return image, message
    
    
#     # Plot data    
#     n_rows_plot = 3
#     n_cols_plot = np.max([n_base_data, n_realizations])
    
#     tri_images_posttrain, _ = tri_stochastics.eval_model(base_data, n_illu)
    
#     fig, axs = plt.subplots(n_rows_plot, n_cols_plot, figsize=(n_cols_plot*3, n_rows_plot*3))
#     for i in range(n_rows_plot):  # loop over rows of plot
#         if i == 0:
#             for j in range(n_base_data):
#                 ax = axs[i, j]
#                 ax.axis('off')
#                 attribute = base_attributes[j]
#                 image, message = eval_attribute_dims(attribute, scenario_index)
#                 ax.imshow(image)
#                 ax.set_title(base_attribute_names[j] + message)
#         if i == 1:
#             for j in range(n_realizations):
#                 ax = axs[i, j]
#                 ax.axis('off')
#                 image = phase_mats[j,:,:]
#                 ax.set_title("Realization nr {}".format(j))
#                 image_example = ax.imshow(image)
#             colormap = image_example.get_cmap()
#         if i == 2:
#             for j in range(n_realizations):
#                 ax = axs[i, j]
#                 ax.axis('off')
#                 image = tri_images_posttrain[j,:,:]
#                 ax.imshow(image.detach(), cmap = colormap)
#                 ax.set_title("Realization nr {}".format(j))
#         # if i == 3:
#         #     for j in range(n_distributional_data_posttrain):
#         #         ax = axs[i, j]
#         #         ax.axis('off')
#         #         attribute = distributional_attributes_posttrain[j]
#         #         image, message = eval_attribute_dims(attribute, scenario_index)
#         #         ax.imshow(image)
#         #         ax.set_title(distributional_attribute_names_posttrain[j] + message)         
        
#     plt.axis('off')
#     plt.tight_layout()
#     plt.show()


# plot_realizations(base_data, full_data, n_realizations = 8, scenario_index = 1)















































