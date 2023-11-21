#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The goal of this script is to train a stochastic model of TRI data based on pyro's
svi formalism and real radar data from breithorn. The model consists in a multivariate
gaussian distribution whose mean and kernel parameters depend on auxiliary data.
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


# ii) Load data
torch.set_default_dtype(torch.float64)
data = loadmat('./Data/submatrix_collection_training_20x20_2023_2days.mat')
# data = loadmat('./Data/submatrix_collection_training_100x100_2023_2days.mat')
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

# Transposition necessary since row wise/column wise unravelling difference 
# between matlab and numpy
# The sequence is range, az, x, y, z, coh, aoi, mean, time
range_mats = torch.tensor(np.transpose(info_mats_data[:,:,0].reshape([-1,n_x,n_y]), (0,2,1))).double()
azimuth_mats = torch.tensor(np.transpose(info_mats_data[:,:,1].reshape([-1,n_x,n_y]), (0,2,1))).double()
x_mats = torch.tensor(np.transpose(info_mats_data[:,:,2].reshape([-1,n_x,n_y]), (0,2,1))).double()
y_mats = torch.tensor(np.transpose(info_mats_data[:,:,3].reshape([-1,n_x,n_y]), (0,2,1))).double()
z_mats = torch.tensor(np.transpose(info_mats_data[:,:,4].reshape([-1,n_x,n_y]), (0,2,1))).double()
coherence_mats = torch.tensor(np.transpose(info_mats_data[:,:,5].reshape([-1,n_x,n_y]), (0,2,1))).double()
aoi_mats = torch.tensor(np.transpose(info_mats_data[:,:,6].reshape([-1,n_x,n_y]), (0,2,1))).double()
meanphase_mats = torch.tensor(np.transpose(info_mats_data[:,:,7].reshape([-1,n_x,n_y]), (0,2,1))).double()
time_mats = torch.tensor(np.transpose(info_mats_data[:,:,8].reshape([-1,n_x,n_y]), (0,2,1))).double()

phase_mats = torch.tensor(np.transpose(phase_vals_data[:,:].reshape([-1,n_x,n_y]), (0,2,1))).double()



# # vi) Create new dataset
# # Create Dataset subclass
# class TRIDataset(torch.utils.data.Dataset):
#     def __init__(self, data):
#         self.data = data

#     def __getitem__(self, index):
#         return self.data[index]

#     def __len__(self):
#         return len(self.data)

# # Invoke class instance
# tri_dataset = TRIDataset()



"""
    3. Simulate data
"""
    

# ii) Dataset preparation

# basic inputs to stochastic model
class BaseData():
    def __init__(self, range_mats, azimuth_mats, x_mats, y_mats, z_mats, coherence_mats, aoi_mats, meanphase_mats, time_mats):
        self.range_mats = range_mats
        self.azimuth_mats = azimuth_mats
        self.x_mats = x_mats
        self.y_mats = y_mats
        self.z_mats = z_mats
        self.coherence_mats = coherence_mats
        self.aoi_mats = aoi_mats
        self.meanphase_mats = meanphase_mats
        self.time_mats = time_mats
                
        def get_basic_attr_list(self):
            attributes = [attr for attr in dir(self) if not (attr.startswith('__'))]
            return attributes
        self.basic_attribute_list = get_basic_attr_list(self)
        self.num_basic_attributes = len(self.basic_attribute_list)
        
base_data = BaseData(range_mats, azimuth_mats,  x_mats, y_mats, z_mats, coherence_mats, aoi_mats, meanphase_mats, time_mats)


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
    
    def integrate_base_data(self, list_inputs, base_data):
        self.n_samples = base_data.range_mats.shape[0]
        self.list_base_data_mats = [getattr(base_data, input_attr).reshape([self.n_samples, n_x,n_y, -1]) for input_attr in list_inputs]
        self.base_data_mats = torch.cat(self.list_base_data_mats, dim = -1)
        self.dim_arg = self.base_data_mats.shape[-1]
        
    # Guide
    def guide(self, base_data, observations = None):
        pass

# list_inputs = ['x_mats', 'y_mats', 'z_mats', 'aoi_mats', 'coherence_mats' ]
list_inputs = ['x_mats', 'y_mats' ]
# list_inputs = ['coherence_mats' ]
tri_stochastics = TRIStochastics(list_inputs, 50, 20, base_data)
observations = full_data.phase_mats.reshape([n_samples, -1])


# iv) Simulation pretraining

simulation_pretrain, full_data_pretrain = copy.copy(tri_stochastics.model(base_data))
simulation_pretrain = copy.copy(simulation_pretrain.reshape([subsample_size, n_x,n_y]))



"""
    5. Inference
"""


# i) Set up training

# specifying scalar options
learning_rate = 0.01
num_epochs = 200
adam_args = {"lr" : learning_rate}

# Setting up svi
optimizer = pyro.optim.AdamW(adam_args)
elbo_loss = pyro.infer.Trace_ELBO()
svi = pyro.infer.SVI(model = tri_stochastics.model, guide = tri_stochastics.guide, optim = optimizer, loss= elbo_loss)


# ii) Execute training

train_elbo = []
for epoch in range(num_epochs):
    epoch_loss = svi.step(base_data, observations)
    train_elbo.append(-epoch_loss)
    if epoch % 10 == 0:
        print("Epoch : {} train loss : {}".format(epoch, epoch_loss))


# iii) Simulation posttraining

simulation_posttrain, full_data_posttrain = copy.copy(tri_stochastics.model(base_data))
simulation_posttrain = copy.copy(simulation_posttrain.reshape([subsample_size, n_x,n_y]))



"""
    6. Plots and illustrations
"""


# i) Illustrate data

fig, axs = plt.subplots(3, 4, figsize=(15, 9))

attributes = [full_data.phase_mats, full_data.coherence_mats, full_data.z_mats, full_data.aoi_mats]
titles = ["Phase Mats", "Coherence Mats", "Elevation Mats", "AOI Mats"]

for i in range(3):  # loop over samples
    for j in range(4):  # loop over attributes
        ax = axs[i, j]
        attribute = attributes[j]
        ax.imshow(attribute[i, :, :])
        if i == 0:
            ax.set_title(titles[j])

        ax.axis('off')

plt.tight_layout()
plt.show()


# ii) Illustrate data, initial simulations, trained simulations

fig, axs = plt.subplots(5, 3, figsize=(9, 15))

attributes = [full_data.phase_mats, simulation_pretrain.detach(), simulation_posttrain.detach()]
titles = ["Phase: data", "pretrain", " posttrain"]

for i in range(5):  # loop over samples
    for j in range(3):  # loop over attributes
        ax = axs[i, j]
        attribute = attributes[j]
        ax.imshow(attribute[i, :, :])
        if i == 0:
            ax.set_title(titles[j])

        ax.axis('off')

plt.tight_layout()
plt.show()


# iii) Multiple realizations for same base_data

def plot_realizations(base_data, full_data, **plotting_opts):
    # This function plots the base data and a bunch of realizations. This is used
    # to compare the realizations of the trained model and the true underlying model.
    # plotting opts include the 'n_realizations', the number of realizations to
    # be shown during plotting and 'scenario_index' the index of the scenario
    # that is shown (i.e. determines the randomly chosen base_data like elevation etc)
    
    # Fetch basic & distributional data
    base_attributes = [getattr(base_data,attr) for attr in base_data.basic_attribute_list]
    base_attribute_names = [attr for attr in base_data.basic_attribute_list]
    n_base_data = base_data.num_basic_attributes
    
    # distributional_attributes = [getattr(full_data, attr) for attr in full_data.distributional_attribute_list]
    # distributional_attribute_names = [attr for attr in full_data.distributional_attribute_list]
    # n_distributional_data = full_data.num_distributional_attributes
    
    distributional_attributes_posttrain = [getattr(full_data_posttrain, attr) for attr in full_data_posttrain.distributional_attribute_list]
    distributional_attribute_names_posttrain = [attr for attr in full_data_posttrain.distributional_attribute_list]
    n_distributional_data_posttrain = full_data_posttrain.num_distributional_attributes
    
    n_realizations = plotting_opts.get('n_realizations', 4)
    scenario_index = plotting_opts.get('scenario_index', 0)

    # Simulate new data


    # Support function    
    def eval_attribute_dims(attribute, scenario_index):
        attribute_dim = (attribute[0,:,:]).shape
        n_attribute_dim = len(attribute_dim)
        if n_attribute_dim ==3:
            image = attribute[scenario_index,:,:,0]
            message = " 1 of {} matrices".format(attribute_dim[2])
        else:
            image = attribute[scenario_index,:,:]
            message = ""
        return image, message
    
    
    # Plot data    
    n_rows_plot = 3
    n_cols_plot = np.max([n_base_data, n_realizations])
    base_data_slice = BaseData(torch.repeat_interleave((range_mats[scenario_index,:,:]).unsqueeze(0), subsample_size, dim = 0),
                               torch.repeat_interleave((azimuth_mats[scenario_index,:,:]).unsqueeze(0), subsample_size, dim = 0),
                               torch.repeat_interleave((x_mats[scenario_index,:,:]).unsqueeze(0), subsample_size, dim = 0),
                               torch.repeat_interleave((y_mats[scenario_index,:,:]).unsqueeze(0), subsample_size, dim = 0),
                               torch.repeat_interleave((z_mats[scenario_index,:,:]).unsqueeze(0), subsample_size, dim = 0),
                               torch.repeat_interleave((coherence_mats[scenario_index,:,:]).unsqueeze(0), subsample_size, dim = 0),
                               torch.repeat_interleave((aoi_mats[scenario_index,:,:]).unsqueeze(0), subsample_size, dim = 0),
                               torch.repeat_interleave((meanphase_mats[scenario_index,:,:]).unsqueeze(0), subsample_size, dim = 0),
                               torch.repeat_interleave((time_mats[scenario_index,:,:]).unsqueeze(0), subsample_size, dim = 0)
                               )
    
    tri_images_posttrain = (tri_stochastics.model(base_data_slice)[0]).reshape([subsample_size, n_x,n_y])
    
    fig, axs = plt.subplots(n_rows_plot, n_cols_plot, figsize=(n_cols_plot*3, n_rows_plot*3))
    for i in range(n_rows_plot):  # loop over rows of plot
        if i == 0:
            for j in range(n_base_data):
                ax = axs[i, j]
                ax.axis('off')
                attribute = base_attributes[j]
                image, message = eval_attribute_dims(attribute, scenario_index)
                ax.imshow(image)
                ax.set_title(base_attribute_names[j] + message)
        if i == 1:
            for j in range(n_realizations):
                ax = axs[i, j]
                ax.axis('off')
                image = phase_mats[j,:,:]
                ax.set_title("Realization nr {}".format(j))
                image_example = ax.imshow(image)
            colormap = image_example.get_cmap()
        if i == 2:
            for j in range(n_realizations):
                ax = axs[i, j]
                ax.axis('off')
                image = tri_images_posttrain[j,:,:]
                ax.imshow(image.detach(), cmap = colormap)
                ax.set_title("Realization nr {}".format(j))
        # if i == 3:
        #     for j in range(n_distributional_data_posttrain):
        #         ax = axs[i, j]
        #         ax.axis('off')
        #         attribute = distributional_attributes_posttrain[j]
        #         image, message = eval_attribute_dims(attribute, scenario_index)
        #         ax.imshow(image)
        #         ax.set_title(distributional_attribute_names_posttrain[j] + message)
                
        
    plt.axis('off')
    plt.tight_layout()
    plt.show()


plot_realizations(base_data, full_data, n_realizations = 8, scenario_index = 1)















































