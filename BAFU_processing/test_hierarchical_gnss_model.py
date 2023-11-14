#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The goal of this script is to train and test a simple gnss timeseries model that
is supposed to provide a stochastic model explaining the genesis of coordinates for 
several different stations at several different timesteps.
For this, do the following:
    1. Imports and definitions
    2. Simulate synthetic data
    3. Stochastic model
    4. Training via SVI
    5. Plots and illustrations
"""



"""
1) Simulating Data:
Given the requirements, we're simulating data for 10 stations, over 100 time steps, and we want 30 different realizations of this data. That means our data will be shaped as (30, 10, 100).

For simplicity, I'll generate the data such that each station's data is a sinusoid with a different frequency:
"""
    
import copy
import numpy as np
import matplotlib.pyplot as plt

# Set seed for reproducibility
np.random.seed(42)

# Constants
num_realizations = 100
num_stations = 1
num_days = 3
num_timesteps = 12
num_basis_funs = 5
num_hours = num_days*num_timesteps

timesteps = np.linspace(0, num_timesteps -1, num_timesteps)  # over 24 hours
all_timesteps = np.linspace(0, num_hours-1, num_hours)  # over 30 days


# basis functions
list_basis_funs = []
def basis_fun(timesteps, frequency):
    return np.sin(frequency * timesteps) + np.random.normal(0, 0.1, size=len(timesteps))

frequencies = []
for k in range(num_basis_funs):
    frequency_temp = (4*k)/(num_timesteps*num_basis_funs)*np.pi
    frequencies.append(frequency_temp)
    list_basis_funs.append(basis_fun(timesteps, frequency_temp))

# Simulating data
d_station = np.random.lognormal(3,1,num_stations)
cov_fun_true = lambda t1,t2, station: np.exp(-((t1-t2)/d_station[station])**2)
K_data = np.zeros([num_hours, num_hours, num_stations])

for m in range(num_stations):    
    for k in range(num_hours):
        for l in range(num_hours):
            K_data[k,l,m] = cov_fun_true(all_timesteps[k], all_timesteps[l], m)
            


def simulate_station_data():
    data = np.zeros([num_realizations, num_stations, num_hours])
    
    for n in range(num_realizations):
        for m in range(num_stations):
            data[n,m,:] = np.random.multivariate_normal(np.zeros([num_hours]), K_data[:,:,m])
        
    return data

data = simulate_station_data()


# Plotting the first realization for visual check
plt.figure(figsize=(10, 10))
for station in range(num_stations):
    plt.subplot(num_stations, 1, station + 1)
    plt.plot(all_timesteps, data[:, station, :].T)
    plt.title(f"Station {station + 1}")
plt.tight_layout()
plt.show()



"""
Simple hierarchical model. Each day is the realization of a multivariate Gaussian
with mean params that should not be too far away from the previous ones.
"""


import pyro
import torch
from pyro.distributions import Normal, MultivariateNormal
pyro.clear_param_store()

# Convert data to PyTorch tensor
observations = torch.tensor(data, dtype=torch.float32).reshape(-1, num_stations, num_days, num_timesteps)

# # RBF kernel function
# def rbf_kernel(x1, x2, lengthscales, variance):
#     # Euclidean distance matrix
#     dist_matrix = torch.cdist(x1 / lengthscales, x2 / lengthscales, p=2)
#     return variance * torch.exp(-0.5 * dist_matrix**2)

# Autoregressive model
def ar_model(observations = None):
    # AR hyperparameters
    transition_mat = pyro.param("transition_mat", torch.eye(num_basis_funs))
    mu_alpha_initial = pyro.param("mu_alpha_initial", torch.ones(num_basis_funs))
    sigma_initial = pyro.param("sigma_initial", torch.eye(num_basis_funs), constraint = pyro.distributions.constraints.positive_definite)
    sigma_transition = pyro.param("sigma_transition", 0.01*torch.eye(num_basis_funs), constraint = pyro.distributions.constraints.positive_definite)
    sigma_noise = pyro.param("sigma_noise", 1e-3*torch.ones(1), constraint = pyro.distributions.constraints.positive )
    
    # Constants & containers
    obs_simu = torch.zeros([num_realizations,num_stations, num_days, num_timesteps])
    basis_fun_vals = torch.zeros([num_timesteps, num_basis_funs])
    for k in range(num_basis_funs):
        basis_fun_vals[:,k] = torch.tensor(list_basis_funs[k])
    alpha_mat = torch.zeros([num_realizations, num_basis_funs, num_days+1])

    with pyro.plate("realization_plate", size = num_realizations, dim = -1):
        dist_initial_alpha = pyro.distributions.MultivariateNormal(loc = mu_alpha_initial, covariance_matrix = sigma_initial)
        initial_alpha = pyro.sample("initial_alpha", dist_initial_alpha)
        list_alpha_mat = [initial_alpha.squeeze()]
        for k in range(num_days):
            local_alpha_dist = pyro.distributions.MultivariateNormal(loc = (list_alpha_mat[k]@(transition_mat.T)).reshape([num_realizations,num_stations,1,num_basis_funs]), covariance_matrix = sigma_transition).to_event(2)
            local_alpha = pyro.sample("local_alpha_{}".format(k), local_alpha_dist)
            list_alpha_mat.append(local_alpha.squeeze())
            local_mu = (local_alpha.squeeze()@basis_fun_vals.T).reshape([num_realizations,num_stations,1,num_timesteps])
            observations_or_none = (observations[:,0,k,:]).reshape([num_realizations,num_stations, 1, num_timesteps]) if observations is not None else None
            local_obs_dist = pyro.distributions.MultivariateNormal(loc = local_mu, covariance_matrix = sigma_noise*torch.eye(local_mu.shape[-1])).to_event(2)
            local_obs = pyro.sample("local_obs_{}".format(k), local_obs_dist, obs = observations_or_none )
            obs_simu[:, 0,k, :] = local_obs.squeeze()
    return obs_simu, alpha_mat


# Plotting untrained model outputs
pretrain_model_data, pretrain_alpha_mat = ar_model()
pretrain_model_data = pretrain_model_data.reshape(-1, num_stations, num_hours)
plt.figure(figsize=(10, 10))
for station in range(num_stations):
    plt.subplot(num_stations, 1, station + 1)
    plt.plot(all_timesteps, pretrain_model_data.detach()[:, station, :].T)
    plt.title(f"Station {station + 1}")
plt.tight_layout()
plt.show()



"""
Component 3: Inference Procedure

1. **Guide Definition**:
   We use the `AutoDiagonalNormal` guide from Pyro's library, which fits a diagonal Gaussian distribution to the posterior. This is also known as the Mean-Field Variational Family.

2. **Optimizer and Loss**:
   - We're employing the Adam optimizer, a popular choice for stochastic optimization. The learning rate (`lr`) is set to 0.03, but this might require tuning based on the convergence rate.
   - The loss function used is `Trace_ELBO`, which calculates the Evidence Lower BOund (ELBO). Maximizing ELBO is equivalent to minimizing the KL divergence between the approximate posterior (from our guide) and the true posterior.

3. **Stochastic Variational Inference (SVI)**:
   SVI is used to iteratively refine our guide's parameters to better approximate the true posterior. We perform SVI for a pre-defined number of iterations (`num_iterations`), monitoring the ELBO loss to check for convergence.

4. **Post-Inference Analysis**:
   After inference, we draw samples from the approximate posterior to study the inferred distributions of our model's hyperparameters (`lengthscales` and `variance`).
"""


import pyro.optim as optim
from pyro.infer import Trace_ELBO, SVI
from pyro.contrib.autoguide import AutoDiagonalNormal
torch.autograd.set_detect_anomaly(True)

# Setting up the guide
ar_guide = pyro.infer.autoguide.AutoNormal(ar_model)

# Optimizer and Loss
adam = optim.Adam({"lr": 0.02})
elbo = Trace_ELBO()
svi = SVI(ar_model, ar_guide, adam, loss=elbo)

# Training loop
num_iterations = 1000
losses = []

for j in range(num_iterations):
    # calculate the loss and take a gradient step
    loss = svi.step(observations)
    losses.append(loss)
    if j % 100 == 0:
        print(f"[iteration {j+1}] loss: {loss}")

# Plotting the loss
plt.figure(figsize=(10, 4))
plt.plot(losses)
plt.title("ELBO Loss over Iterations")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.show()

# Fetching inference results
for name, value in pyro.get_param_store().items():
    print("Name of parameter: {}, shape : {}".format(name, pyro.param(name).data.cpu().numpy().shape))

# Plotting trained model outputs
posttrain_model_data, posttrain_alpha_mat = ar_model()
posttrain_model_data = posttrain_model_data.reshape(-1, num_stations, num_hours)
plt.figure(figsize=(10, 10))
for station in range(num_stations):
    plt.subplot(num_stations, 1, station + 1)
    plt.plot(all_timesteps, posttrain_model_data.detach()[:, station, :].T)
    plt.title(f"Station {station + 1}")
plt.tight_layout()
plt.show()





















