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
x = torch.linspace(-1,1,n_time).reshape([-1, n_time])



"""
    2. Generate data
"""


# i) Generate mean function

list_funs = [lambda x: torch.ones(x.shape), lambda x: x, lambda x: x**2, lambda x : torch.sin(4*np.pi*x)]
n_funs = len(list_funs)
true_params = torch.tensor([1, -1, 1, 1])

def mu_fun(x, params):
    mu_vec = torch.zeros([1, n_time])
    for k in range(n_funs):
        mu_vec += params[k]*(list_funs[k](x))
    return mu_vec


# ii) Generate covariance function

d=0.3
cov_fun_true = lambda x_1,x_2: 0.3*torch.exp(-(x_1 - x_2)**2/d)
cov_mat = torch.zeros([n_time,n_time])
for k in range(n_time):
    for l in range(n_time):
        cov_mat[k,l] = cov_fun_true(x[0,k],x[0,l])


# ii) Generate timeseries based on parametric model

mu_vec = mu_fun(x, true_params)
y = torch.zeros([n_series,n_time])
for k in range(n_series):
    y[k,:] = torch.tensor(np.random.multivariate_normal(mu_vec.flatten(), cov_mat))



"""
    3. Define model and guide
"""


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


















