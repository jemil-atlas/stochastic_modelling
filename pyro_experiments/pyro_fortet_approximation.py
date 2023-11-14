#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The goal of this script is to showcase that the neural network training allows
arbitrarily complicated covariance functions to be learned.
For this, do the following:
    1. Imports and definitions
    2. Generate data via Fortets theorem
    3. Set up a learnable model
    4. Perform optimization
    5. Plots and illustrations
"""


"""
    1. Imports and definitions -----------------------------------------------
"""


# i) Imports

import pyro
# from scipy.signal import sawtooth
import torch
import copy
import numpy as np
import matplotlib.pyplot as plt


# ii) Definitions

n_t = 100 
n_simu = 20
dim_cov_features = 5
t = np.linspace(0,1,n_t)
tt = np.repeat(t.reshape([1,-1]), repeats = n_simu, axis = 0)
tt_tensor = torch.tensor(tt)

torch.set_default_dtype(torch.float64)
pyro.clear_param_store()



"""
    2. Generate data via Fortets theorem -------------------------------------
"""


# i) Define base functions
def square_wave(t, period=2*np.pi):
    return np.sign(np.sin(t * (2 * np.pi / period)))

def sawtooth(t, period=2*np.pi):
    return 2 * (t / period - np.floor(0.5 + t / period))

# ii) Set up family of functions

coeffs = np.linspace(1,0.1,dim_cov_features)
alphas = np.linspace(0,10, dim_cov_features)
list_funs = []
# base_fun = lambda t, alpha: np.cos(alpha*t)
base_fun = lambda t, alpha : square_wave(alpha*t)
for k in range(dim_cov_features):
    list_funs.append(lambda t, coeff = coeffs[k], alpha = alphas[k]: coeff * base_fun(t,alpha))

# iii) Set up covariance function via Fortet

def cov_fun(t_1, t_2):
    phi_1 = np.zeros([dim_cov_features])
    phi_2 = np.zeros([dim_cov_features])
    for k in range(dim_cov_features):
        phi_1[k] = list_funs[k](t_1)
        phi_2[k] = list_funs[k](t_2)
    cov_val = phi_1.T@phi_2
    return cov_val


# iv) Construct covariance matrix

K = np.zeros([n_t,n_t])
for k in range(n_t):
    for l in range(n_t):
        K[k,l] = cov_fun(t[k],t[l])


# v) Generate data

y_data = np.zeros([n_simu,n_t])
for k in range(n_simu):
    y_data[k,:] = np.random.multivariate_normal(np.zeros([n_t]), K)



"""
    3. Set up a learnable model -----------------------------------------------
"""


# i) Mapping class

class MapToL2(pyro.nn.PyroModule):
    def __init__(self, dim_hidden, dim_l2):
        # Initialize using method from base class
        super().__init__()
        self.dim_input = 1
        self.dim_hidden = dim_hidden
        self.dim_l2 = dim_l2
        
        # linear transforms
        self.fc_1 = torch.nn.Linear(self.dim_input, self.dim_hidden)
        self.fc_2 = torch.nn.Linear(self.dim_hidden, self.dim_hidden)
        self.fc_3 = torch.nn.Linear(self.dim_hidden, self.dim_l2)
        # nonlinear transforms
        self.nonlinear = torch.nn.Sigmoid()
        
    def forward(self, x):
        # Define forward computation on the input data x
        # Shape the minibatch so that batch_dims are on left, argument_dims on right
        x = x.reshape([-1, n_t, 1])
        
        # Then compute hidden units and output of nonlinear pass
        hidden_units_1 = self.nonlinear(self.fc_1(x))
        hidden_units_2 = self.nonlinear(self.fc_2(hidden_units_1))
        feature_mat = self.fc_3(hidden_units_2)
        return feature_mat
        
    
# ii) Covariance function definition

def covariance_function(map_to_l2, tt):
        
    # compute covariance
    feature_mats = map_to_l2(tt)
    feature_mats_T = feature_mats.permute(0,2,1)
    cov_mats = torch.bmm(feature_mats, feature_mats_T)
    # eigenvalues = torch.linalg.eigvalsh(cov_mats[0,:,:])
    # print("Minimum eigenvalue:", torch.min(eigenvalues))
    return cov_mats
    

# iii) Stochastics object class

class TimeseriesStochastics(pyro.nn.PyroModule):
    # Initialization of class and l2 mapping
    def __init__(self, dim_hidden, dim_l2):
        super().__init__()
        self.dim_input = 1
        self.dim_hidden = dim_hidden
        self.dim_l2 = dim_l2
        self.map_to_l2 = MapToL2(self.dim_hidden, self.dim_l2)
        self.tt_tensor = tt_tensor
        
    # Stochastic model for forward simulation
    def model(self, obs = None):
        pyro.module('map_to_l2', self.map_to_l2)
        # Set up distribution
        self.mean_vecs = torch.zeros([n_simu, n_t])
        self.cov_mats = covariance_function(self.map_to_l2, self.tt_tensor) 
        cov_regularizer = 1e-3*(torch.eye(n_t).repeat(n_simu, 1, 1))
        obs_dist = pyro.distributions.MultivariateNormal(loc = self.mean_vecs, covariance_matrix = self.cov_mats + cov_regularizer)
    
        with pyro.plate('batch_plate', size = n_simu, dim = -1):
            observations = pyro.sample('observations', obs_dist, obs = obs )
        return observations

    def guide(self, obs = None):
        pass
    

# iv) Initial results of untrained model

timeseries_stochastics = TimeseriesStochastics(dim_hidden = 100, dim_l2 = 50)

simulation_pretrain = copy.copy(timeseries_stochastics.model())
K_pretrain = timeseries_stochastics.cov_mats[0,:,:].detach()



"""
    4. Perform optimization --------------------------------------------------
"""


# i) Set up training

# specifying scalar options
learning_rate = 0.01
num_epochs = 1000
adam_args = {"lr" : learning_rate}

# Setting up svi
optimizer = pyro.optim.Adam(adam_args)
elbo_loss = pyro.infer.Trace_ELBO()
svi = pyro.infer.SVI(model = timeseries_stochastics.model, guide = timeseries_stochastics.guide, optim = optimizer, loss= elbo_loss)


# ii) Execute training

observations = torch.tensor(y_data)
train_elbo = []
for epoch in range(num_epochs):
    epoch_loss = svi.step(observations)
    train_elbo.append(-epoch_loss)
    if epoch % 100 == 0:
        print("Epoch : {} train loss : {}".format(epoch, epoch_loss))


# iii) Simulation posttraining

simulation_posttrain = copy.copy(timeseries_stochastics.model())
K_posttrain = timeseries_stochastics.cov_mats[0,:,:].detach()



"""
    5. Plots and illustrations -----------------------------------------------
"""


# i) Original data, pre- post-train

fig, axs = plt.subplots(3, 2, figsize=(12, 12))
axs[0,0].plot(t, y_data.T)
axs[0,0].set_title('Original data')
axs[0,1].imshow(K)
axs[0,1].set_title('Covariance matrix')

axs[1,0].plot(t, simulation_pretrain.T.detach())
axs[1,0].set_title('Pre train data')
axs[1,1].imshow(K_pretrain)
axs[1,1].set_title('Covariance matrix pretrain')

axs[2,0].plot(t, simulation_posttrain.T.detach())
axs[2,0].set_title('Post train data')
axs[2,1].imshow(K_posttrain)
axs[2,1].set_title('Covariance matrix posttrain')













