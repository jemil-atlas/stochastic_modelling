#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The goal of this script is to learn a simple stochastic model of data that
represents terrestrial radar interferometry. The model consists in a multivariate
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


# ii) Definitions

n_x = 10
n_y = 10
n_total = n_x*n_y
n_samples = 10

x = np.linspace(0,1,n_x)
y = np.linspace(0,1,n_y)

xx, yy = np.meshgrid(y,x)
x_vec = np.vstack((xx.flatten(), yy.flatten())).T

pyro.clear_param_store()
torch.set_default_dtype(torch.float64)



"""
    2. Build auxiliary datasets
"""


# i) Range

range_mats = np.zeros([n_samples, n_x, n_y])
r_start = np.zeros([n_samples])
r_end = np.zeros([n_samples])
for k in range(n_samples):
    r_start[k] = np.random.uniform(0,500)
    r_end[k] = r_start[k] + 500 
    range_mats[k,:,:] = np.flipud(np.repeat(np.linspace(r_start[k], r_end[k], n_x).reshape([n_x,1]),n_y,axis = 1))
range_mats = torch.tensor(range_mats).double()


# ii) Azimuth

azimuth_mats = np.zeros([n_samples, n_x, n_y])
az_start = np.zeros([n_samples])
az_end = np.zeros([n_samples])
for k in range(n_samples):
    az_start[k] = np.random.uniform(0,np.pi/2)
    az_end[k] = az_start[k] + np.pi/2
    azimuth_mats[k,:,:] = np.fliplr(np.repeat(np.linspace(az_start[k], az_end[k], n_y).reshape([1, n_y]),n_x,axis = 0))
azimuth_mats = torch.tensor(azimuth_mats).double()


# iii) Location

location_mats = np.zeros([n_samples, n_x, n_y,2])
for k in range(n_samples):
    location_mats[:,:,:,0] = np.cos(azimuth_mats)*range_mats
    location_mats[:,:,:,1] = np.sin(azimuth_mats)*range_mats
location_mats = torch.tensor(location_mats).double()


# iv) Coherence

d_coherence = 0.2
cov_fun_coherence = lambda x_vec1, x_vec2 : np.exp(-((np.linalg.norm(x_vec1-x_vec2,2)/d_coherence)**2))
K_coherence = np.zeros([n_total,n_total])
for k in range(n_total):
    for l in range(n_total):
        K_coherence[k,l] = cov_fun_coherence(x_vec[k,:], x_vec[l,:])
coherence_mats = np.zeros([n_samples, n_x, n_y ])
for k in range(n_samples):
    coherence_mats[k,:,:] = 1 + 1*np.clip(np.random.multivariate_normal(np.zeros(n_total), K_coherence).reshape([n_x,n_y]), -1,0)
coherence_mats = torch.tensor(coherence_mats).double()


# v) Elevation

d_elevation = 0.2
cov_fun_elevation = lambda x_vec1, x_vec2 : 100*np.exp(-((np.linalg.norm(x_vec1-x_vec2,2)/d_elevation)**2))
K_elevation = np.zeros([n_total,n_total])
for k in range(n_total):
    for l in range(n_total):
        K_elevation[k,l] = cov_fun_elevation(x_vec[k,:], x_vec[l,:])
elevation_mats = np.zeros([n_samples, n_x, n_y ])
for k in range(n_samples):
    elevation_mats[k,:,:] = np.random.multivariate_normal(np.zeros(n_total), K_elevation).reshape([n_x,n_y])
elevation_mats = torch.tensor(elevation_mats).double()


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


# i) Initialize stochastic parameters

covariance_scenario_table = {'range' : 0,\
                             'location' : 1,\
                             'elevation' : 0,\
                             'coherence' : 0
                             }

def build_noise_variance_mats(coherence_mats, covariance_scenario_table):
    
    coherence_dependence = covariance_scenario_table['coherence']
    logical_multiplier = 1 if coherence_dependence else 0
    noise_variance_mats = logical_multiplier * 0.1*(1-coherence_mats**2)
    return noise_variance_mats

def phase_cov_fun(range_1, range_2, elevation_1, elevation_2, location_1, location_2, covariance_scenario_table):
    d_location = 500
    d_elevation = 50
    
    elevation_dependence = covariance_scenario_table['elevation']
    location_dependence = covariance_scenario_table['location']
    range_dependence = covariance_scenario_table['range']
    
    term_1 = (1/1000)*(np.minimum(range_1, range_2)) if range_dependence else 1
    term_2 = np.exp(-((np.linalg.norm(location_1-location_2,2)/d_location)**2)) if location_dependence else 1
    term_3 = (1/100)*np.sqrt(np.abs(elevation_1*elevation_2)) if elevation_dependence else 0
    term_4 = np.exp(-((np.abs(elevation_1 - elevation_2)/d_elevation)**2))
    
    phase_cov_val = term_1*term_2 + term_3*term_4
    return phase_cov_val
    

# ii) Dataset preparation

# basic inputs to stochastic model
class BaseData():
    def __init__(self, range_mats, azimuth_mats, location_mats, elevation_mats, coherence_mats, covariance_scenario_table):
        self.range_mats = range_mats
        self.azimuth_mats = azimuth_mats
        self.location_mats = location_mats
        self.elevation_mats = elevation_mats
        self.coherence_mats = coherence_mats
        self.covariance_scenario_table = covariance_scenario_table
        def get_basic_attr_list(self):
            attributes = [attr for attr in dir(self) if not (attr.startswith('__') or attr.startswith('covariance_scenario'))]
            return attributes
        self.basic_attribute_list = get_basic_attr_list(self)
        self.num_basic_attributes = len(self.basic_attribute_list)
base_data = BaseData(range_mats, azimuth_mats, location_mats, elevation_mats, coherence_mats, covariance_scenario_table)
        
# full outputs of stochastic model
class FullData():
    def __init__(self, base_data, phase_mats, noise_variance_mats, K_phase_mats):
        n_samples = base_data.range_mats.shape[0]
        
        # Integrate distributional_data
        self.noise_variance_mats = noise_variance_mats
        self.K_phase_mats = K_phase_mats
        variances = np.diagonal(K_phase_mats, axis1=1, axis2=2)
        self.aps_variance_mats = variances.reshape((n_samples, n_x, n_y))
        
        # List distributional_data
        def get_distributional_attr_list(self):
            attributes = [attr for attr in dir(self) if not (attr.startswith('__') or attr.startswith('covariance_scenario'))]
            return attributes
        self.distributional_attribute_list = get_distributional_attr_list(self)
        self.num_distributional_attributes = len(self.distributional_attribute_list)

        # copy from base_data
        for name, attribute in base_data.__dict__.items():
            setattr(self, name, attribute)
        # Add simulations
        self.phase_mats = phase_mats            

# iii) Define stochastic model for data generation
        
def simple_model(base_data):
    n_samples = base_data.range_mats.shape[0]
    
    range_mats = base_data.range_mats
    location_mats = base_data.location_mats
    elevation_mats = base_data.elevation_mats
    coherence_mats = base_data.coherence_mats
    covariance_scenario_table = base_data.covariance_scenario_table
    
    noise_variance_mats = build_noise_variance_mats(coherence_mats, covariance_scenario_table)
    phase_mats = np.zeros([n_samples, n_x, n_y ])
    K_phase_mats = np.zeros([n_samples, n_total, n_total])
    for m in range(n_samples):
        for k in range(n_total):
            for l in range(n_total):
                K_phase_mats[m,k,l] = phase_cov_fun(range_mats.reshape([n_samples, -1])[m,k], range_mats.reshape([n_samples, -1])[m,l],
                                                  elevation_mats.reshape([n_samples, -1])[m,k], elevation_mats.reshape([n_samples, -1])[m,l],
                                                  location_mats.reshape([n_samples, -1, 2])[m,k], location_mats.reshape([n_samples, -1, 2])[m,l],
                                                  covariance_scenario_table)
        K_phase_mats[m,:,:] = K_phase_mats[m,:,:] + np.diag(noise_variance_mats[m,:,:].flatten())
    for k in range(n_samples):
        phase_mats[k,:,:] = (np.random.multivariate_normal(np.zeros(n_total), K_phase_mats[k,:,:])).reshape([n_x,n_y])
        
    data = FullData(base_data, phase_mats, noise_variance_mats, K_phase_mats)
    
    return data


# iii) Apply stochastic model

full_data = simple_model(base_data)
observations = torch.tensor(full_data.phase_mats.reshape([n_samples, -1])).float()


    
    
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
# map_to_l2 = MapToL2(1,10,5)


# ii) Covariance function definition

def covariance_function(map_to_l2, base_data_mats):
        
    # compute covariance
    feature_mats = map_to_l2(base_data_mats)
    feature_mats_T = feature_mats.permute(0,2,1)
    cov_mats = torch.bmm(feature_mats, feature_mats_T)
    return cov_mats
        

# iii) Stochastics class

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
        # print("Samples:{}".format(self.n_samples))
        # register the map_to_l2 parameters with pyro
        pyro.module("map_to_l2", self.map_to_l2)        
        # TODO This is where the error comes from cov mat has wront dims, when called leading to wrong obs dist shapes
        cov_mats = covariance_function(self.map_to_l2, self.base_data_mats) 
        cov_regularizer = 1e-3*(torch.eye(n_total).repeat(self.n_samples, 1, 1))
        
        with pyro.plate('batch_plate',size = self.n_samples, dim = -1):
            obs_dist = pyro.distributions.MultivariateNormal(loc = torch.zeros([self.n_samples, n_total]), covariance_matrix = cov_mats + cov_regularizer)
            # print("batch_shape : {}, event_shape : {}".format(obs_dist.batch_shape, obs_dist.event_shape))
            obs = pyro.sample("obs", obs_dist, obs = observations)
        data = FullData(base_data, obs.reshape([self.n_samples, n_x,n_y]).detach(), cov_regularizer.detach(), cov_mats.detach())
        return obs, data
    
    def integrate_base_data(self, list_inputs, base_data):
        self.n_samples = base_data.range_mats.shape[0]
        self.list_base_data_mats = [getattr(base_data, input_attr).reshape([self.n_samples, n_x,n_y, -1]) for input_attr in list_inputs]
        self.base_data_mats = torch.cat(self.list_base_data_mats, dim = -1)
        self.dim_arg = self.base_data_mats.shape[-1]
        
    # Guide
    def guide(self, base_data, observations = None):
        pass

list_inputs = ['location_mats']
tri_stochastics = TRIStochastics(list_inputs, 50, 50, base_data)


# iv) Simulation pretraining

simulation_pretrain, full_data_pretrain = copy.copy(tri_stochastics.model(base_data))
simulation_pretrain = copy.copy(simulation_pretrain.reshape([n_samples, n_x,n_y]))



"""
    5. Inference
"""


# i) Set up training

# specifying scalar options
learning_rate = 0.003
num_epochs = 6000
adam_args = {"lr" : learning_rate}

# Setting up svi
optimizer = pyro.optim.Adam(adam_args)
elbo_loss = pyro.infer.Trace_ELBO()
svi = pyro.infer.SVI(model = tri_stochastics.model, guide = tri_stochastics.guide, optim = optimizer, loss= elbo_loss)


# ii) Execute training

train_elbo = []
for epoch in range(num_epochs):
    epoch_loss = svi.step(base_data, observations)
    train_elbo.append(-epoch_loss)
    if epoch % 100 == 0:
        print("Epoch : {} train loss : {}".format(epoch, epoch_loss))


# iii) Simulation posttraining

simulation_posttrain, full_data_posttrain = copy.copy(tri_stochastics.model(base_data))
simulation_posttrain = copy.copy(simulation_posttrain.reshape([n_samples, n_x,n_y]))



"""
    6. Plots and illustrations
"""


# i) Illustrate data

fig, axs = plt.subplots(3, 5, figsize=(15, 9))

attributes = [full_data.phase_mats, full_data.aps_variance_mats, full_data.coherence_mats, full_data.elevation_mats, full_data.K_phase_mats]
titles = ["Phase Mats", "APS Variance Mats", "Coherence Mats", "Elevation Mats", "Covariance Mats"]

for i in range(3):  # loop over samples
    for j in range(5):  # loop over attributes
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
    
    distributional_attributes = [getattr(full_data, attr) for attr in full_data.distributional_attribute_list]
    distributional_attribute_names = [attr for attr in full_data.distributional_attribute_list]
    n_distributional_data = full_data.num_distributional_attributes
    
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
    n_rows_plot = 5
    n_cols_plot = np.max([n_base_data, n_realizations])
    base_data_slice = BaseData(range_mats[scenario_index,:,:].unsqueeze(0),
                               azimuth_mats[scenario_index,:,:].unsqueeze(0),
                               location_mats[scenario_index,:,:].unsqueeze(0),
                               elevation_mats[scenario_index,:,:].unsqueeze(0),
                               coherence_mats[scenario_index,:,:].unsqueeze(0),
                               covariance_scenario_table)
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
            for j in range(n_distributional_data):
                ax = axs[i, j]
                ax.axis('off')
                attribute = distributional_attributes[j]
                image, message = eval_attribute_dims(attribute, scenario_index)
                ax.imshow(image)
                ax.set_title(distributional_attribute_names[j] + message)
        if i == 2:
            for j in range(n_realizations):
                ax = axs[i, j]
                image = simple_model(base_data_slice).phase_mats.squeeze()
                ax.imshow(image)
                ax.set_title("Realization nr {}".format(j))
        if i == 3:
            for j in range(n_distributional_data):
                ax = axs[i, j]
                ax.axis('off')
                attribute = distributional_attributes_posttrain[j]
                image, message = eval_attribute_dims(attribute, scenario_index)
                ax.imshow(image)
                ax.set_title(distributional_attribute_names[j] + message)
                
        if i == 4:
            for j in range(n_realizations):
                ax = axs[i, j]
                ax.axis('off')
                image = (tri_stochastics.model(base_data_slice)[0]).reshape([n_x,n_y])
                ax.imshow(image.detach())
                ax.set_title("Realization nr {}".format(j))
        
    plt.axis('off')
    plt.tight_layout()
    plt.show()


plot_realizations(base_data, full_data, n_realizations = 4)















































