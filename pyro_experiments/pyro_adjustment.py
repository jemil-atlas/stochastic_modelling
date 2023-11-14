#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adjustment task showcasing also uncertainty of estimated parameters. Fit a line 
via svi and interpret the guide parameters as uncerteinties in the parameters 
posteriors. Compare with results of classical adjustment.

For this, do the following:
    1. Imports and definitions
    2. Generate Data
    3. Classical Adjustment
    4. Guide, Model, SVI
    5. Plots and illustrations
"""


"""
    1. Imports and definitions
"""

# i) Imports

import numpy as np
import torch
import pyro
import matplotlib.pyplot as plt


# ii) Definitons

a_true = 1.0
b_true = 1.0
theta_true = torch.tensor(np.vstack((a_true,b_true))).flatten()
n_simu = 10
t_data = torch.linspace(0,1,n_simu)

pyro.clear_param_store()


"""
    2. Generate Data
"""

# Noise
sigma_noise = 0.1
sigma_mat = sigma_noise**2*np.eye(n_simu)
noise = torch.tensor(np.random.normal(0,sigma_noise,[n_simu]))

# Data
A_mat = torch.tensor(np.vstack((np.ones(n_simu), t_data)).T)
x_data = A_mat@theta_true + noise



"""
    3. Classical Adjustment
"""

# Estimate the BLUE for theta
pinv_A = torch.linalg.pinv(A_mat)
theta_adjustment = pinv_A@x_data

# Infer its covariance matrix
sigma_adjustment = pinv_A@sigma_mat@pinv_A.T



"""
    4. Guide, Model, SVI
"""

# Build the model
def model(t_obs, x_obs = None):
    # static, non-adjustable quantities
    n_obs = t_obs.shape[0]
    A_model = torch.tensor(np.vstack((np.ones(n_obs), t_obs)).T)
    
    # parameters and distributions
    # theta = pyro.param('theta', 1.0*torch.zeros([2]))
    z = pyro.sample('z', pyro.distributions.Normal(torch.zeros([2]), 10).to_event(1))
    x_dist = pyro.distributions.Normal(A_model@z, torch.tensor(sigma_noise))
    
    # declare observations independent
    with pyro.plate('t_plate', size = n_obs,dim = -1):
        x = pyro.sample('x_obs', x_dist, obs = x_obs)
    return x


model_trace = pyro.poutine.trace(model).get_trace(t_data)
model_trace.nodes
print(model_trace.format_shapes())


# Build the guide
# guide = pyro.infer.autoguide.AutoNormal(model)
def guide(t_obs, x_obs = None):
    z_mean = pyro.param('z_mean', torch.zeros([2]))
    z_cov = pyro.param('z_cov', torch.eye(2),constraint = pyro.distributions.constraints.positive_definite)
    z = pyro.sample('z', pyro.distributions.MultivariateNormal(loc = z_mean, covariance_matrix = z_cov))
    return z

guide_trace = pyro.poutine.trace(guide).get_trace(t_data)
guide_trace.nodes
print(guide_trace.format_shapes())


# Perform inferenc
adam = pyro.optim.Adam({"lr": 0.01})
elbo = pyro.infer.Trace_ELBO()
svi = pyro.infer.SVI(model = model, guide = guide, optim = adam, loss = elbo)

for k in range(10000):
    current_loss = svi.step(t_data, x_data)
    if k % 200 == 0:
        print('Step: {} , Loss : {}'.format(k,current_loss))

for param_name, param_value in pyro.get_param_store().items():
    print('{} : {}'.format(param_name, param_value))

theta_svi = pyro.get_param_store()['z_mean']
sigma_svi = pyro.get_param_store()['z_cov']



"""
    5. Plots and illustrations
"""

# Generate the samples
sample_adjustment = np.random.multivariate_normal(theta_adjustment, sigma_adjustment, 1000)
sample_svi = np.random.multivariate_normal(theta_svi.detach(), sigma_svi.detach(), 1000)

# Create a figure with two subplots
fig, axs = plt.subplots(2, figsize=(10, 10))

# Plot histograms of the two samples
n_bins = 20

# We can set the number of bins with the `bins` kwarg
axs[0].hist(sample_adjustment, bins=n_bins)
axs[0].set_title('Histogram of samples from adjustment solution')
axs[1].hist(sample_svi, bins=n_bins)
axs[1].set_title('Histogram of samples from svi solution')

plt.tight_layout()
plt.show()

print('Adjustment solution: theta : {} , sigma = {}'.format(theta_adjustment, sigma_adjustment))
print('SVI solution: theta : {} , sigma = {}'.format(theta_svi, sigma_svi))











