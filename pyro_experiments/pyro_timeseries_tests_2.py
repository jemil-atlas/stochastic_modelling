#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The goal of this script is to learn a basic timeseries model from timeseries data.
We will start basic with just a bunch of parameters fitted to form a parametric
model dependent on time.
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

n_series = 20
n_time = 100
x = torch.linspace(-1,1,n_time).reshape([-1, n_time])



"""
    2. Generate data
"""


# i) Generate parameters

list_funs = [lambda x: torch.ones(x.shape), lambda x: x, lambda x: x**2, lambda x : torch.sin(4*np.pi*x)]
n_funs = len(list_funs)
true_params = torch.tensor([1, -1, 1, 1])

sigma_noise = 0.3
def mu_fun(x, params):
    mu_mat = torch.zeros([n_series, n_time])
    for k in range(n_funs):
        mu_mat += params[k]*torch.repeat_interleave((list_funs[k](x)), n_series, axis = 0)
    return mu_mat


# ii) Generate timeseries based on parametric model

mu_mat = mu_fun(x, true_params)
y = torch.tensor(np.random.normal(mu_mat, sigma_noise))



"""
    3. Define model and guide
"""


# i) ARMA model: construction involves autoregressive and moving average terms
n_phi = 3
n_theta = 3
n_max = np.max([n_phi, n_theta])
def model(y_obs = None):
    # Parameter setup
    coeffs_phi = pyro.param("coeffs_phi", init_tensor = torch.zeros(n_phi))
    coeffs_theta = pyro.param("coeffs_theta", init_tensor = torch.zeros(n_theta))
    sigma = pyro.param("sigma", init_tensor = torch.ones(1))
    
    # Noise setup
    extension_tensor = torch.ones([n_series, n_time])
    epsilon_dist = pyro.distributions.Normal(0*extension_tensor,sigma).to_event(1)
    epsilon = pyro.sample("epsilon", epsilon_dist)
    
    # Constructive process
    y = torch.zeros([n_series, n_time])
    for k in pyro.plate("batch_plate", size = n_series, dim = -1):
        for l in range(n_time):
            y_kl_mean = torch.conv1d(epsilon, )
            y_kl_dist = pyro.distributions.Normal(y_kl_mean, sigma)
            y_obs_kl_or_none = y_obs[k,l] if y_obs is not None else None
            y_kl = pyro.sample("y_{}{}".format(k,l), y_kl_dist, obs = y_obs_kl_or_none)
    
    return y


# ii) Guide for ARMA model

def guide(y_obs = None):
    pass



"""
    4. SVI and predictions
"""


# i) Setup optimization

optimizer = pyro.optim.Adam({"lr" : 1e-2})
loss = pyro.infer.Trace_ELBO()
svi = pyro.infer.SVI(model, guide, optim = optimizer, loss = loss)


# ii) Perform training loop

n_epochs = 500
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

fig, ax = plt.subplots(2, figsize = (10,5), dpi=300)
ax[0].plot(x.T, y.T)
ax[0].set_title('Original data')
ax[1].plot(x.T, y_model.T)
ax[1].set_title('Model data')
plt.tight_layout()
plt.plot



















