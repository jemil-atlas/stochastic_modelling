#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The goal of this script is to learn a basic timeseries model from timeseries data.
We will fit a gaussian process featuring nontrivial mean and covariance and the 
use it to simulate and interpolate.
For this, do the following:
    1. Imports and definitions
    2. Generate data
    3. Define model and guide
    4. SVI and predictions
    5. Plots and illustrations

We will upgrade the timeseries tests in other installments of this series:
    pyro_timeseries_tests_1 : parametric model
    pyro_timeseries_tests_2 : autoregressive process
    pyro_timeseries_tests_3 : gaussian process
    pyro_timeseries_tests_4 : gaussian process with covariates
    pyro_timeseries_tests_5 : hierarchical gaussian process
    pyro_timeseries_tests_6 : deep gaussian process
"""


"""
    1. Imports and definitions
"""


# i) Imports

import torch
import pyro
import numpy as np
import matplotlib.pyplot as plt


# ii) Definitions

pyro.clear_param_store()
n_series = 200
n_time = 100
x = torch.linspace(0,1,n_time).reshape([-1, n_time])
x_mat = torch.repeat_interleave(x, n_series, dim = 0)



"""
    2. Generate data
"""


# i) Generate mean function

def mu_fun(x, h):
    mu_vec = torch.zeros([1, n_time]) + x + 0.5*h
    return mu_vec


# ii) Generate noise variance function

def sigma_fun(x, h):
    sigma_mat = np.abs(h)
    return sigma_mat



# iii) Generate functions for h

mu_h = torch.sin(2*np.pi*x).reshape([-1,n_time])
cov_fun_true = lambda x_1,x_2: 0.2*torch.minimum(x_1,x_2)
cov_mat_h = torch.zeros([n_time,n_time])
for k in range(n_time):
    for l in range(n_time):
        cov_mat_h[k,l] = cov_fun_true(x[0,k],x[0,l])


# iv) Generate timeseries based on covariate model

h = torch.zeros([n_series, n_time])
for k in range(n_series):
    h[k,:] = torch.abs(torch.tensor(np.random.multivariate_normal(mu_h.flatten(), cov_mat_h)))
    
mu_mat = mu_fun(x, h)
sigma_mat = sigma_fun(x, h)
y = torch.zeros([n_series,n_time])
for k in range(n_series):
    y[k,:] = torch.tensor(np.random.normal(mu_mat[k,:].flatten(), sigma_mat[k,:]))



"""
    3. Define model and guide
"""


# i) Setup neural nets

class GPNN(pyro.nn.PyroModule):
    # Initialize the module
    def __init__(self, dim_input, dim_hidden, nn_batch_shape):
        # Evoke by passing bottleneck and hidden dimension
        # Initialize instance using init method from base class
        super().__init__()
        self.nn_batch_shape = nn_batch_shape
        
        # linear transforms
        self.fc_1 = torch.nn.Linear(dim_input, dim_hidden)
        self.fc_2 = torch.nn.Linear(dim_hidden, dim_hidden)
        self.fc_31 = torch.nn.Linear(dim_hidden, 1)
        self.fc_32 = torch.nn.Linear(dim_hidden, 1)
        # nonlinear transforms
        self.nonlinear = torch.nn.Tanh()
        
    def forward(self, x, h):
        # Define forward computation on the input data x and h
        x_reshaped = x.reshape([-1,1])
        h_reshaped = h.reshape([-1,1])
        xh = torch.hstack((x_reshaped,h_reshaped))
        
        # Then compute hidden units and output of nonlinear pass
        hidden_units = self.nonlinear(self.fc_1(xh))
        hidden_units = self.nonlinear(self.fc_2(hidden_units))
        mu_reshaped = self.fc_31(hidden_units)
        sigma_reshaped = torch.exp(self.fc_32(hidden_units))
        
        mu = mu_reshaped.reshape(self.nn_batch_shape)
        sigma = sigma_reshaped.reshape(self.nn_batch_shape)
        
        return mu, sigma
 
nn_batch_shape = [n_series, n_time]       
gp_nn = GPNN(2,5, nn_batch_shape)


# i) Gaussian process model

def model(y_obs = None):
    # Setup parameters
    mu = pyro.param("mu", init_tensor = torch.ones(n_time).reshape([-1,n_time]))
    sigma = pyro.param("sigma", init_tensor = torch.eye(n_time).reshape([-1,n_time,n_time]), constraint = pyro.distributions.constraints.positive_definite)
    reg_tensor = 1e-4*torch.eye(n_time).reshape([-1,n_time,n_time])
    
    # Setup distribution and sample
    y_dist = pyro.distributions.MultivariateNormal(mu, sigma + reg_tensor).expand([n_series])
    with pyro.plate("batch_plate", size = n_series, dim = -1):
        y = pyro.sample("y", y_dist, obs = y_obs)
    return y


# ii) Guide for gp model

def guide(y_obs = None):
    pass



"""
    4. SVI and predictions
"""


# i) Setup optimization

optimizer = pyro.optim.Adam({"lr" : 1e-3})
loss = pyro.infer.Trace_ELBO()
svi = pyro.infer.SVI(model, guide, optim = optimizer, loss = loss)


# ii) Perform training loop

n_epochs = 10000
for k in range(n_epochs):
    svi.step(y)
    if k % 100 == 0:
        print("Elbo loss step {} : {}".format(k, svi.evaluate_loss(y)))


"""
    5. Plots and illustrations
"""


# i) Print out parameters

for name, param in pyro.get_param_store().items():
    print("{} : {}".format(name, param))
    

# ii) Plot the data and model outputs

y_model = model().detach()
mu = pyro.get_param_store()['mu'].detach()
sigma = pyro.get_param_store()['sigma'].detach()


fig, ax = plt.subplots(2, figsize = (10,5), dpi=300)
ax[0].plot(x.T, y.T)
ax[0].set_title('Original data')
ax[1].plot(x.T, y_model.T)
ax[1].set_title('Model data')
plt.tight_layout()
plt.plot

fig, ax = plt.subplots(1,2, figsize = (10,5), dpi=300)
ax[0].plot(x.T, mu.T)
ax[0].set_title('mean')
ax[1].imshow(sigma.squeeze())
ax[1].set_title('covariance')
plt.tight_layout()
plt.plot


















