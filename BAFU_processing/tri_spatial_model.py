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
        
    def __extract_regression_tensor(self, list_regression_vars):
        
        regvar_list = []
        for regvar in list_regression_vars:
            regvar_list.append(getattr(self,regvar))
        regression_tensor = torch.stack(regvar_list, dim = 3)
        return regression_tensor
        
base_data = BaseData(info_mats_data, dict_ind_base_data)


# full outputs of stochastic model

def convert_covmat_to_correlation(cov_mat):
    # Assuming std is your matrix of shape [10, 400] with standard deviations
    inv_std = (1 / (np.sqrt(np.diagonal(cov_mat, axis1=1, axis2=2)) + 1e-3))
    
    inv_std_diag_matrices = np.zeros(cov_mat.shape)
    rows, cols = np.diag_indices_from(inv_std_diag_matrices[0])
    inv_std_diag_matrices[:, rows, cols] = inv_std
    
    corrmats = np.matmul(inv_std_diag_matrices, np.matmul(cov_mat, inv_std_diag_matrices))
    return corrmats


class FullData():
    def __init__(self, base_data, phase_mats, **optional_input):
        self.K_phase_mats = optional_input.get('K_phase_mats', None)
        if self.K_phase_mats is not None:
                n_samples = self.K_phase_mats.shape[0]
                variances = np.diagonal(self.K_phase_mats.detach(), axis1=1, axis2=2)
                std = np.sqrt(np.diagonal(self.K_phase_mats.detach(), axis1=1, axis2=2))
                self.correlation_matrices = convert_covmat_to_correlation(self.K_phase_mats.detach())
                
                covariances =  np.hstack((np.diagonal(self.K_phase_mats.detach(), 
                        offset = 1, axis1=1, axis2=2), np.zeros([n_samples,1])))
                covariances = variances.reshape((n_samples, n_x, n_y))
                covariances = covariances[:,:,:-1]
                correlations =  np.hstack((np.diagonal(self.correlation_matrices, 
                        offset = 1, axis1=1, axis2=2), np.zeros([n_samples,1])))
                correlations = correlations.reshape((n_samples, n_x, n_y))
                correlations = correlations[:,:,:-1]
                
                self.variance_mats = variances.reshape((n_samples, n_x, n_y))
                self.covariances = covariances
                self.correlations = correlations
                
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
subsample_size = 16

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

list_inputs = ['range_mats', 'azimuth_mats', 'z_mats', 'aoi_mats', 'coherence_mats' ]
# list_inputs = ['x_mats', 'y_mats', 'z_mats', 'aoi_mats', 'coherence_mats' ]
# list_inputs = ['x_mats', 'y_mats' ]
# list_inputs = ['coherence_mats' ]
tri_stochastics = TRIStochastics(list_inputs, 100, 50, base_data)
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
learning_rate = 1e-10
num_epochs = 100
adam_args = {"lr" : learning_rate}

# Setting up svi
optimizer = pyro.optim.NAdam(adam_args)
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

def plot_covariance_info_real(base_data, illustration_index):
    simulations, full_data_simulations = copy.copy(tri_stochastics.eval_model(base_data, illustration_choice))
    basic_attr_name_list = list(dict_ind_base_data.keys())
    basic_attr_list_temp = [getattr(full_data_simulations, name) for name in basic_attr_name_list]
    basic_attr_list = [basic_attr[illustration_choice,...] for basic_attr in basic_attr_list_temp]
    fig, axs = plt.subplots(3, 3, figsize=(9, 9))  
    for i in range(3):
        for j in range(3):
            axs[i, j].imshow(basic_attr_list[i * 3 + j][illustration_index,...])
            axs[i, j].set_title(basic_attr_name_list[i*3+j])  # Hide axis
            axs[i, j].axis('off')  # Hide axis
    
    plt.tight_layout()
    plt.show()
    
    fig, axs = plt.subplots(1, 3, figsize=(10, 5))  
    axs[0].imshow(full_data_simulations.K_phase_mats[illustration_index,...].detach())
    axs[0].set_title('Full Covariance matrix')
    axs[0].axis('off')
    
    axs[1].imshow(full_data_simulations.variance_mats[illustration_index,...])
    axs[1].set_title('Variances')
    axs[1].axis('off')
    
    axs[2].imshow(full_data_simulations.correlations[illustration_index,...])
    axs[2].set_title('Correlations between neighbors')
    axs[2].axis('off')
    fig.suptitle('Predicted distributional parameters')
    plt.tight_layout()
    plt.show()


plot_covariance_info_real(base_data, 3)


# Build synthetic data to illustrate the impact of certain explanatory variables

synth_data_dict_init = {'aoi_mats' : 0 ,\
                'azimuth_mats' : 0 ,\
                'coherence_mats' : 0.5 ,\
                'meanphase_mats' : 0 ,\
                'range_mats' : 0 ,\
                'time_mats' : 0 ,\
                'x_mats' : 0 ,\
                'y_mats' : 0 ,\
                'z_mats' : 0 }
synth_data_dict_z = copy.copy(synth_data_dict_init)
synth_data_dict_z['z_mats'] =  True
synth_data_dict_range = copy.copy(synth_data_dict_init)
synth_data_dict_range['range_mats'] =  True
synth_data_dict_coh = copy.copy(synth_data_dict_init)
synth_data_dict_coh['coherence_mats'] =  True
    
def build_synthetic_base_data(base_data, synth_data_dict):
    synth_base_data = copy.copy(base_data)
    basic_attr_name_list = list(dict_ind_base_data.keys())
    for name in basic_attr_name_list:
        if synth_data_dict[name]  == True:
            pass
        else:
            original_attr = getattr(synth_base_data, name)
            original_dims = original_attr.shape
            setattr(synth_base_data, name, synth_data_dict[name]*torch.ones(original_dims))
    
    return synth_base_data
    
    
    
def plot_covariance_info_synthetic(base_data, synth_data_dict, illustration_index):
    synth_base_data = build_synthetic_base_data(base_data, synth_data_dict)    
    
    simulations, full_data_simulations = copy.copy(tri_stochastics.eval_model(synth_base_data, illustration_choice))
    basic_attr_name_list = list(dict_ind_base_data.keys())
    basic_attr_list_temp = [getattr(full_data_simulations, name) for name in basic_attr_name_list]
    basic_attr_list = [basic_attr[illustration_choice,...] for basic_attr in basic_attr_list_temp]

    fig, axs = plt.subplots(1, 4, figsize=(10, 5))  
    exp_var_name = [key for key, value in synth_data_dict.items() if value is True][0]
    explanatory_variable = getattr(full_data_simulations, exp_var_name)[illustration_index,...].detach()
    axs[0].imshow(explanatory_variable)
    axs[0].set_title('Variable {}'.format(exp_var_name))
    axs[0].axis('off')
    
    axs[1].imshow(full_data_simulations.K_phase_mats[illustration_index,...].detach())
    axs[1].set_title('Full Covariance matrix')
    axs[1].axis('off')
    
    axs[2].imshow(full_data_simulations.variance_mats[illustration_index,...])
    axs[2].set_title('Variances')
    axs[2].axis('off')
    
    axs[3].imshow(full_data_simulations.correlations[illustration_index,...])
    axs[3].set_title('Correlations between neighbors')
    axs[3].axis('off')
    fig.suptitle('Predicted distributional parameters')
    plt.tight_layout()
    plt.show()

plot_covariance_info_synthetic(base_data, synth_data_dict_z, 3)
plot_covariance_info_synthetic(base_data, synth_data_dict_range, 3)
plot_covariance_info_synthetic(base_data, synth_data_dict_coh, 3)


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















































