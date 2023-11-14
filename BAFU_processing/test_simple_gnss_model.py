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
    
    
import numpy as np
import matplotlib.pyplot as plt

# Set seed for reproducibility
np.random.seed(42)

# Constants
num_realizations = 30
num_stations = 3
num_timesteps = 24
timesteps = np.linspace(0, num_timesteps -1, num_timesteps)  # over 24 hours


# # Simulating simple data = trend + noise
# def simulate_station_data(timesteps, frequency):
#     return np.sin(frequency * timesteps) + np.random.normal(0, 0.1, size=len(timesteps))

# timesteps = np.linspace(0, 4 * np.pi, num_timesteps)  # over 4 cycles of sine wave
# data = np.empty((num_realizations, num_stations, num_timesteps))

# for realization in range(num_realizations):
#     for station in range(num_stations):
#         frequency = 0.1 + 0.1 * station  # different frequency for each station
#         data[realization, station, :] = simulate_station_data(timesteps, frequency)


# Simulating stochastic data = GP
d_station = np.random.lognormal(3,0.1,num_stations)
cov_fun_true = lambda t1,t2, station: np.exp(-((t1-t2)/d_station[station])**2)
K_data = np.zeros([num_timesteps, num_timesteps, num_stations])

for m in range(num_stations):    
    for k in range(num_timesteps):
        for l in range(num_timesteps):
            K_data[k,l,m] = cov_fun_true(timesteps[k], timesteps[l], m)
            


def simulate_station_data():
    data = np.zeros([num_realizations, num_stations, num_timesteps])
    
    for n in range(num_realizations):
        for m in range(num_stations):
            data[n,m,:] = np.random.multivariate_normal(np.zeros([num_timesteps]), K_data[:,:,m])
        
    return data

data = simulate_station_data()

# Plotting the first realization for visual check
plt.figure(figsize=(10, 10))
for station in range(num_stations):
    plt.subplot(num_stations, 1, station + 1)
    plt.plot(timesteps, data[:, station, :].T)
    plt.title(f"Station {station + 1}")
plt.tight_layout()
plt.show()



"""
Now, for the model, we're going to create a Gaussian process (GP) model that will work with your data dimensions. The GP model will have a kernel function that accepts 4D input: station ID, x, y, z (all station-based) and t (time).

For the sake of this example, I'm simplifying it by considering station ID as a unique value for each station (1 to 10) and assuming x, y, and z as fixed values per station (this is just for demonstration purposes; in a real-world scenario, each station would have unique x, y, z coordinates). The primary variable of interest is t which spans across the 100 time steps.
"""


import pyro
import torch
from pyro.distributions import Normal, MultivariateNormal
pyro.clear_param_store()

# Convert data to PyTorch tensor
observations = torch.tensor(data, dtype=torch.float32).reshape(-1, num_stations*num_timesteps)

# Constants  dummy x, y, z values for stations
x, y, z = torch.linspace(1, num_stations, num_stations), torch.linspace(1,num_stations, num_stations), torch.linspace(1,num_stations, num_stations) 
timesteps = torch.linspace(0, 4 * np.pi, num_timesteps)

# Create the inputs for our GP
inputs = torch.zeros(num_stations, num_timesteps, 4)
for station in range(num_stations):
    inputs[station, :, 0] = x[station]
    inputs[station, :, 1] = y[station]
    inputs[station, :, 2] = z[station]
inputs[:, :, 3] = timesteps

# RBF kernel function
def rbf_kernel(x1, x2, lengthscales, variance):
    # Euclidean distance matrix
    dist_matrix = torch.cdist(x1 / lengthscales, x2 / lengthscales, p=2)
    return variance * torch.exp(-0.5 * dist_matrix**2)

# Gaussian process model
def gp_model(inputs, observations = None):
    # GP hyperparameters
    mu = pyro.param("mu", torch.zeros(num_timesteps*num_stations))
    lengthscales = pyro.param("lengthscales", torch.ones(4), constraint = pyro.distributions.constraints.positive) 
    variance = pyro.param("variance", torch.ones(1), constraint = pyro.distributions.constraints.positive )
    noise_variance = pyro.param("noise_variance", 1e-3*torch.ones(1), constraint = pyro.distributions.constraints.positive )
    
    # Kernel computation
    K = rbf_kernel(inputs.reshape(-1, 4), inputs.reshape(-1, 4), lengthscales, variance)
    K += (1e-3+noise_variance)*torch.eye(K.size(0)) # Add a small jitter for numerical stability
    
    # Multivariate normal likelihood
    obs_or_none = observations.reshape(-1, num_stations*num_timesteps) if observations is not None else None
    obs_dist = MultivariateNormal(mu, K).expand([num_realizations]).to_event(0)
    with pyro.plate("sample_plate", size = num_realizations, dim = -1):
        obs_sample = pyro.sample("obs", obs_dist, obs=obs_or_none)
        
    return obs_sample#.reshape(-1, num_stations, num_timesteps)

# Plotting untrained model outputs
pretrain_model_data = gp_model(inputs).reshape(-1, num_stations, num_timesteps)
plt.figure(figsize=(10, 10))
for station in range(num_stations):
    plt.subplot(num_stations, 1, station + 1)
    plt.plot(timesteps, pretrain_model_data.detach()[:, station, :].T)
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

# Setting up the guide
def gp_guide(inputs, observations):
    pass

# Optimizer and Loss
adam = optim.Adam({"lr": 0.003})
elbo = Trace_ELBO()
svi = SVI(gp_model, gp_guide, adam, loss=elbo)

# Training loop
num_iterations = 1500
losses = []

for j in range(num_iterations):
    # calculate the loss and take a gradient step
    loss = svi.step(inputs, observations)
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
    print(name, pyro.param(name).data.cpu().numpy())

# Plotting trained model outputs
posttrain_model_data = gp_model(inputs).reshape(-1, num_stations, num_timesteps)
plt.figure(figsize=(10, 10))
for station in range(num_stations):
    plt.subplot(num_stations, 1, station + 1)
    plt.plot(timesteps, posttrain_model_data.detach()[:, station, :].T)
    plt.title(f"Station {station + 1}")
plt.tight_layout()
plt.show()





















