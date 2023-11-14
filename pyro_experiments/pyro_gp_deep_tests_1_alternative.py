#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The goal of this script is to show how a deep gp can be trained to learn 
nonnormally distributed data. The data does not yet contain samples from a 
stochastic process, they are independent realizations of a distribution whose
parameters depend on an index set.
For this, do the following:
    1. Imports and definitions
    2. Generate Data
    3. Set up deep GP
    4. Training
    5. Plots and illustrations

"""



"""
    1. Imports and definitions
"""


# i) Imports

import pyro
import torch
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam
import matplotlib.pyplot as plt


# ii) Definitions

n_samples = 300

# Clear param store
pyro.clear_param_store()


"""
    2. Generate Data
"""


# i) Generate synthetic data

data = torch.rand(n_samples).reshape([-1,1])
t = torch.linspace(0,0,n_samples).reshape([-1,1])

# Define a prior layer
def prior_gp_layer(t, layer_name):
    # Define mean and covariance
    mean = torch.zeros_like(t)
    scale = pyro.param(f"{layer_name}_scale", torch.ones_like(t), constraint=dist.constraints.positive)
    
    obs_dist = dist.Normal(mean, scale).to_event(1)
    
    # Sample from Gaussian Process
    with pyro.plate(f"{layer_name}_plate", t.shape[0], dim=-1):
        layer_output = pyro.sample(layer_name, obs_dist)
    return layer_output

# Define a Gaussian Process layer
def gp_layer(z, layer_name, transform = None):
    # Define mean and covariance
    mean = pyro.param(f"{layer_name}_mean", torch.zeros_like(z))
    scale = pyro.param(f"{layer_name}_scale", torch.ones_like(z), constraint=dist.constraints.positive)
    obs_dist = dist.Normal(mean, scale).to_event(1)
    
    # Sample from Gaussian Process
    with pyro.plate(f"{layer_name}_plate", z.shape[0], dim=-1):
        layer_output = pyro.sample(layer_name, obs_dist)
    return layer_output

# Define the deep GP model
def deep_gp_model(t, data = None):
    transform = torch.nn.ReLU
    z0 = prior_gp_layer(t, "layer_0")
    z1 = gp_layer(z0, "layer_1", transform)
    z2 = gp_layer(z1, "layer_2", transform)

    return z2

# Define the guide (variational distribution)
guide = pyro.infer.autoguide.AutoDiagonalNormal(deep_gp_model)

# Pre-train data generation
pre_train_data = deep_gp_model(t)


# Initialize the inference algorithm
pyro.clear_param_store()
optimizer = ClippedAdam({"lr": 0.001})
svi = SVI(deep_gp_model, guide, optimizer, loss=Trace_ELBO())

# Training loop
losses = []
num_epochs = 2000
for epoch in range(num_epochs):
    loss = svi.step(t, data)
    losses.append(loss)
    if epoch % 500 == 0:
        print(f"Epoch {epoch} - ELBO loss: {loss}")

# Extract the trained parameters
params = pyro.get_param_store()
print(params)

# Prost-train data generation
post_train_data = deep_gp_model(t)



"""
    5. Plots and illustrations
"""




# Plotting the loss
plt.figure(figsize=(10, 6))
plt.plot(losses)
plt.title("ELBO Loss During Training")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()


# # Plotting
# plt.figure(figsize=(10, 5))
# plt.title("Deep Gaussian Process pre-train")
# plt.hist(data.detach().numpy())
# plt.legend()
# plt.show()





# Create a figure with 5 vertically aligned subplots
fig, axes = plt.subplots(3, 1, figsize=(10, 15))

# Plot the line plots
axes[0].hist(data.detach().numpy())
axes[0].set_title("Original_data")

axes[1].hist(pre_train_data.detach().numpy())
axes[1].set_title("GP_pretraining")

axes[2].hist(post_train_data.detach().numpy())
axes[2].set_title("GP_posttraining")

# Make layout tight
plt.tight_layout()
plt.show()



# # ii) Plot distributional_parameters

# mu_gp = pyro.get_param_store()['mu_gp']
# sigma_gp = pyro.get_param_store()['sigma_gp']

# fig, axes = plt.subplots(5, 1, figsize=(10, 15))

# # Plot the line plots
# axes[0].plot(t, mu_gp.detach())
# axes[0].set_title("GP_mean")

# axes[1].imshow(sigma_gp.detach())
# axes[1].set_title("GP_covariance")


























