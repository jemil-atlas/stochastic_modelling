#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The goal of this script is to showcase the effects of chaining together Gaussian
distributions (as is done in deep Gaussian processes) and how the sequential
application of Gaussian distributions can lead to non-Gaussian results.
For this, do the following:
    1. Imports and definitions
    2. Simple Chain: Linear mean
    3. Chain: Nonlinear mean
    4. Simple Chain: Linear variance
    5. Chain: Nonlinear variance
    6. Chain: ANN to produce mean and variance
    7. Plots and illustrations
"""



"""
    1. Imports and definitions
"""


# i) Imports

import numpy as np
import pyro
import torch
import matplotlib.pyplot as plt


# ii) Definitions

n_samples = 10000
n_depth = 5
n_depth_dgp_ann = 2 # (when an ann is called, how often is it called in the dgp )



"""
    2. Simple Chain: Linear mean
"""


# i) Model linear mean

def dgp_linear_mean(n_samples):
    # Iteratively call the Gaussian distribution with previous_output = current_mean
    latent_variable = pyro.sample('latent_variable_0',pyro.distributions.Normal(loc = torch.zeros(n_samples), scale = torch.ones(1)))
    for k in range(1,n_depth):
        latent_variable = pyro.sample('latent_variable_{}'.format(k), pyro.distributions.Normal(loc = latent_variable, scale = torch.ones(1)))
    return latent_variable
result_dgp_lm = dgp_linear_mean(n_samples)



"""
    3. Chain: Nonlinear mean
"""


# i) Model nonlinear mean

def dgp_nonlinear_mean(n_samples):
    # Iteratively call the Gaussian distribution with relu(previous_output) = current_mean
    latent_variable = pyro.sample('latent_variable_0',pyro.distributions.Normal(loc = torch.zeros(n_samples), scale = torch.ones(1)))
    for k in range(1,n_depth):
        latent_variable = pyro.sample('latent_variable_{}'.format(k), pyro.distributions.Normal(loc = torch.relu(latent_variable), scale = torch.ones(1)))
    return latent_variable
result_dgp_nlm = dgp_nonlinear_mean(n_samples)



"""
    4. Simple Chain: Linear variance
"""


# i) Model linear variance

def dgp_linear_var(n_samples):
    # Iteratively call the Gaussian distribution with |previous_output| = current_variance
    latent_variable = pyro.sample('latent_variable_0',pyro.distributions.Normal(loc = torch.zeros(n_samples), scale = torch.ones(1)))
    for k in range(1,n_depth):
        latent_variable = pyro.sample('latent_variable_{}'.format(k), pyro.distributions.Normal(loc = torch.zeros(n_samples), scale = torch.abs(latent_variable)))
    return latent_variable
result_dgp_lv = dgp_linear_var(n_samples)



"""
    5. Chain: Nonlinear variance
"""

# i) Model nonlinear variance

def dgp_nonlinear_var(n_samples):
    # Iteratively call the Gaussian distribution with relu(previous_output) = current_variance
    latent_variable = pyro.sample('latent_variable_0',pyro.distributions.Normal(loc = torch.zeros(n_samples), scale = torch.ones(1)))
    for k in range(1,n_depth):
        latent_variable = pyro.sample('latent_variable_{}'.format(k), pyro.distributions.Normal(loc = torch.zeros(n_samples), scale = torch.relu(latent_variable) + 0.01))
    return latent_variable
result_dgp_nlv = dgp_nonlinear_var(n_samples)



"""
    6. Chain: ANN to produce mean and variance
"""


# i) Construct ANN class

class ANN(pyro.nn.PyroModule):
    def __init__(self, input_dim, hidden_dim, output_dim, ann_depth):
        super(ANN, self).__init__()
        
        self.layers = torch.nn.ModuleList()
        
        # Input layer
        self.layers.append(torch.nn.Linear(input_dim, hidden_dim))
        self.layers.append(torch.nn.ReLU())
        
        # Hidden layers
        for _ in range(ann_depth):
            self.layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(torch.nn.ReLU())
        
        # Output layer
        self.layers.append(torch.nn.Linear(hidden_dim, output_dim))
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


#  ii) Instantiate two ann's

ann_1 = ANN(1,1,2,1)
ann_2 = ANN(1,1,2,5)


# iii) Build models where distrbutional params come from ANN

# ANN 1
def dgp_ann_1(n_samples):
    # Iteratively call the Gaussian distribution with mean/variance = ann(previous output)
    latent_variable = pyro.sample('latent_variable_0',pyro.distributions.Normal(loc = torch.zeros(n_samples), scale = torch.ones(1)))
    for k in range(1,n_depth_dgp_ann):
        output_ann = ann_1(latent_variable.reshape([-1, 1]))
        output_ann_mean = output_ann[:,0]
        output_ann_var = torch.sigmoid(output_ann[:,1])
        latent_variable = pyro.sample('latent_variable_{}'.format(k), pyro.distributions.Normal(loc = output_ann_mean, scale = output_ann_var + 0.01))
    return latent_variable
result_dgp_ann_1 = dgp_ann_1(n_samples)

# ANN 2
def dgp_ann_2(n_samples):
    # Iteratively call the Gaussian distribution with mean/variance = ann(previous output)
    latent_variable = pyro.sample('latent_variable_0',pyro.distributions.Normal(loc = torch.zeros(n_samples), scale = torch.ones(1)))
    for k in range(1,n_depth_dgp_ann):
        output_ann = ann_2(latent_variable.reshape([-1, 1]))
        output_ann_mean = output_ann[:,0]
        output_ann_var = torch.sigmoid(output_ann[:,1])
        latent_variable = pyro.sample('latent_variable_{}'.format(k), pyro.distributions.Normal(loc = output_ann_mean, scale = output_ann_var + 0.01))
    return latent_variable
result_dgp_ann_2 = dgp_ann_2(n_samples)




"""
    7. Plots and illustrations
"""




# Create a figure with 5 vertically aligned subplots
fig, axes = plt.subplots(6, 1, figsize=(10, 15))

# Plot the hist plots
axes[0].hist(result_dgp_lm.detach().numpy(), bins = 50)
axes[0].set_title("Linear mean DGP chain")

axes[1].hist(result_dgp_nlm.detach().numpy(), bins = 50)
axes[1].set_title("Nonlinear mean DGP chain")


axes[2].hist(result_dgp_lv.detach().numpy(), bins = 50)
axes[2].set_title("Linear variance DGP chain")

axes[3].hist(result_dgp_nlv.detach().numpy(), bins = 50)
axes[3].set_title("Nonlinear variance DGP chain")

# The ANN plots dont look interesting for Relu because you just see the pre-initialized
# biases as often the output of a whole Relu layer might be 0
# They also seem uninteresting for higher depths as things look more Gaussian then.
# This might be due to central limit theorem or something like that; it is not
# yet clear if this means, this constellation is uninteresting in all cases.
# Applying the ann to e.g. t = torch.linspace(0,1,10).reshape([-1,1]) via
# plt.plot(ann_1(t)[:,0].detach()) reveals the transformation to be mostly linear.
axes[4].hist(result_dgp_ann_1.detach().numpy(), bins = 50)
axes[4].set_title("ANN 1 DGP chain")

axes[5].hist(result_dgp_ann_2.detach().numpy(), bins = 50)
axes[5].set_title("ANN 2 DGP chain")




























