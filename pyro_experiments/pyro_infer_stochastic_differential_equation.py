#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The goal of this script is to use pyro to learn a stochastic differential equation.
This is done by demanding that Ax-b is distributed according to the standard
normal and then declaring A, b as parameters to be inferred.
For this, do the following:
    1. Imports and definitions
    2. Simulate data
    3. Set up model
    4. Train and evaluate
    5. Illustrations and plots
    
The script is meant solely for educational and illustrative purposes. Written by
Jemil Avers Butt, Atlas optimization GmbH, www.atlasoptimization.com.
"""


"""
    1. Imports and definitions
"""


# i) Imports

import torch
import pyro
import matplotlib.pyplot as plt


# ii) Definitions

n_simu = 100
n_time = 10

time = torch.linspace(0,1,n_time)
dt = time[1] - time[0]



"""
    2. Simulate data
"""


# i) Wave equation (d/dt)^2 x + lambda * x = 0)

# Differential equation parameters
omega = 2 * torch.pi
sine = torch.sin(omega * time)

# Discretized second derivative matrix
Delta = torch.zeros([n_time, n_time])
Delta.diagonal().fill_(-2)
Delta.diagonal(offset=1).fill_(1)
Delta.diagonal(offset=-1).fill_(1)
Delta = Delta / (dt ** 2)  # Scale by (dt^2)

# Constructing A_true
A_true = (1/100)*(Delta + omega**2 * torch.eye(n_time))
# A_true = A_true[1:-1,:]


# ii) Noise and data generation

sigma = 1
A_pinv = torch.linalg.pinv(A_true)

system_noise = torch.zeros([n_simu, n_time])
signal = torch.zeros([n_simu, n_time])
for k in range(n_simu):
    system_noise[k,:] = torch.normal(torch.zeros([n_time]), sigma)
    signal[k,:] = A_pinv @ system_noise[k,:]



"""
    3. Set up model
"""


# i) Pyro forward model

def model(signal):
    A_param = pyro.param("A_param", init_tensor = A_true.detach() )
    # A_param = pyro.param("A_param", init_tensor = torch.eye(n_time) )
    # A_param = pyro.param("A_param", init_tensor = 0.01* torch.randn([n_time,n_time]) )
    
    # Standard normal
    z_loc = torch.zeros([n_simu,n_time])
    z_scale = torch.ones([n_simu, n_time])
    z_dist = pyro.distributions.Normal(loc = z_loc, scale = z_scale).to_event(1)
    
    z_pred = signal @ A_param
    
    with pyro.plate('batch_plate', size = n_simu, dim = -1):
        z = pyro.sample('z', z_dist, obs = z_pred)
    return z
    

# iii) Define guide - since there are no stochastic latents, it is empty. 
# We let it pass the signal though, so it integrates with the rest of the script

def guide(signal):
    return None


"""
    4. Train and evaluate
"""


# i) Set up training

# specifying scalar options
learning_rate = 1*1e-3
num_epochs = 5000
adam_args = {"lr" : learning_rate}

# Setting up svi
optimizer = pyro.optim.NAdam(adam_args)
elbo_loss = pyro.infer.Trace_ELBO()
svi = pyro.infer.SVI(model = model, guide = guide, optim = optimizer, loss= elbo_loss)


# ii) Execute training

train_elbo = []
for epoch in range(num_epochs):
    epoch_loss = svi.step(signal)
    train_elbo.append(-epoch_loss)
    if epoch % 100 == 0:
        # log the data on tensorboard
        print("Epoch : {} train loss : {}".format(epoch, epoch_loss))

A_inferred = pyro.param("A_param")



"""
    5. Illustrations and plots
"""


# i) Plot A_true and A_inferred

fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns

# Display A_true in the first subplot
axs[0].imshow(A_true, cmap='viridis')
axs[0].set_title('A_true')
axs[0].set_xlabel('Column')
axs[0].set_ylabel('Row')

# Display A_inferred in the second subplot
axs[1].imshow(A_inferred.detach(), cmap='viridis')
axs[1].set_title('A_inferred')
axs[1].set_xlabel('Column')
axs[1].set_ylabel('Row')

plt.tight_layout()  # Adjust the layout
plt.show()


# ii) Plot latent z

n_illu = 20
n_illu = torch.min(torch.tensor([n_illu,n_simu]))
z_inferred = model(signal).detach()

fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns

# Display true system noise
axs[0].plot(time, system_noise[0:n_illu,:].T)
axs[0].set_title('True latents')
axs[0].set_xlabel('time')

# Display inferred system noise
axs[1].plot(time, z_inferred[0:n_illu,:].T)
axs[1].set_title('Inferred latents')
axs[1].set_xlabel('time')

plt.tight_layout()  # Adjust the layout
plt.show()
