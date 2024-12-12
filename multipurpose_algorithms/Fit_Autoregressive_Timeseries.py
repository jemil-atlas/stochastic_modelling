#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fit an autoregressive model of type X_k+1 = mu + eps + A@X_k to a sequence of
vector-valued observations. Use pyro to estimate distributional parameters and
the value of the autoregressive coefficients in A.
For this, do the following:
    1. Imports and definitions
    2. Simulate some data (or import)
    3. Build model and guide
    4. Inference 
    5. Plots and illustrations
"""

"""
    1. Imports and definitions
"""


# i) Imports

import torch
import pyro
import matplotlib.pyplot as plt
import matplotlib.cm as cm

seed = 0
torch.manual_seed(seed)
pyro.set_rng_seed(seed)


# ii) Definitions

n_event = 2
n_batch = 200

A_true = torch.normal(0, 0.4, [n_event, n_event])
B_true = torch.normal(0, 1, [n_event, n_event])

mu_true = torch.ones([1, n_event])
sigma_true = (B_true) @ (B_true.T)
time = torch.arange(0, n_batch)



"""
    2. Simulate some data (or import)
"""


# i) Data distribution setup

x_0 = torch.zeros([1,n_event])
x_k_list = [x_0]


# Draw iteratively
for k in range(n_batch):
    x_prev = x_k_list[k]
    x_mean = (A_true @ x_prev.T).T + mu_true
    data_dist = pyro.distributions.MultivariateNormal(x_mean, sigma_true)
    x_succ = data_dist.sample()
    x_k_list.append(x_succ)
x_k_list = x_k_list[1:]

data = torch.vstack(x_k_list)



"""
    3. Build model and guide
"""


# i) Pyro model

def model(observations = None):
    #  Set up parameters
    mu = pyro.param('mu', init_tensor = torch.zeros([1,n_event]))
    sigma = pyro.param('sigma', init_tensor = torch.eye(n_event), 
                       constraint = pyro.distributions.constraints.positive_definite)
    A = pyro.param('A', init_tensor = torch.zeros([n_event, n_event]))
    x_0 = pyro.param('x_0', init_tensor = torch.zeros([1,n_event]))
    
    # Sampling sequence
    x_k_list = [x_0] 
    for k in pyro.plate('batch_plate', size = n_batch):
        x_prev = x_k_list[k]
        x_mean = (A @ x_prev.T).T + mu
        model_dist = pyro.distributions.MultivariateNormal(loc = x_mean, covariance_matrix = sigma)
        obs_or_None = observations[k,:].reshape([-1,n_event]) if observations is not None else None
        x_succ = pyro.sample('model_sample_{}'.format(k), model_dist, obs = obs_or_None)
        x_k_list.append(x_succ)
    
    x_k_list = x_k_list[1:]
    sample = torch.vstack(x_k_list)
    return sample


# ii) Pyro guide

def guide(observations = None):
    pass


# iii) Untrained samples

n_sample_runs = 2
sample_runs = []
for k in range(n_sample_runs):
    sample_runs.append(model().detach().numpy())



"""
    4. Inference 
"""

# i) Run the optimization

adam = pyro.optim.Adam({"lr": 0.03})
elbo = pyro.infer.Trace_ELBO()
svi = pyro.infer.SVI(model, guide, adam, elbo)

losses = []
for step in range(200):  
    loss = svi.step(data)
    losses.append(loss)
    if step % 50 == 0:
        print('epoch: {}, ELBO loss: {}'.format(step, loss))


# ii) Sample from trained process

sample_runs_trained = []
for k in range(n_sample_runs):
    sample_runs_trained.append(model().detach().numpy())


"""
    5. Plots and illustrations
"""


# i) Plot data

plt.figure(1, dpi = 300)
plt.scatter(data[:,0], data[:,1])
plt.plot(data[:,0], data[:,1],  alpha=0.3, linestyle='-', marker='o')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('2D Data with Lines Connecting Successive Points')


# Normalize indices to use with colormap
norm = plt.Normalize(time.min(), time.max())
colors = cm.viridis(norm(time))  # Use a colormap (e.g., 'viridis')

# Plot with colored lines connecting successive data points
plt.figure(2, dpi = 300)
plt.scatter(data[:,0], data[:,1], c=time, cmap='viridis', edgecolors='black', label='Data Points')
for i in range(len(time) - 1):
    plt.plot(data[i:i+2,0], data[i:i+2,1], color=colors[i], alpha=0.4,)

# Add labels, colorbar, and legend
plt.colorbar(cm.ScalarMappable(norm=norm, cmap='viridis'), label='time')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('2D Data with Temporal Color Gradient')
plt.grid(True)
plt.show()


# ii) Plot model prior to training

fig, axes = plt.subplots(1, n_sample_runs, figsize=(10, 5), dpi = 300)
for k in range(n_sample_runs):
    axes[k].scatter(sample_runs[k][:,0], sample_runs[k][:,1], c=time, cmap='viridis', edgecolors='black', label='Data Points')
    for i in range(len(time) - 1):
        axes[k].plot(sample_runs[k][i:i+2,0], sample_runs[k][i:i+2,1], color=colors[i], alpha=0.3,)
    
    # Add labels, colorbar, and legend
    axes[k].set_xlabel('X-axis')
    axes[k].set_ylabel('Y-axis')
    axes[k].set_title('2D Data with Temporal Color Gradient')
    axes[k].grid(True)
    if k == n_sample_runs-1:
        fig.colorbar(cm.ScalarMappable(norm=norm, cmap='viridis'), label='time')
plt.tight_layout()
plt.show()


# iii) Plot model post training

fig, axes = plt.subplots(1, n_sample_runs, figsize=(10, 5), dpi = 300)
for k in range(n_sample_runs):
    axes[k].scatter(sample_runs_trained[k][:,0], sample_runs_trained[k][:,1], c=time, cmap='viridis', edgecolors='black', label='Data Points')
    for i in range(len(time) - 1):
        axes[k].plot(sample_runs_trained[k][i:i+2,0], sample_runs_trained[k][i:i+2,1], color=colors[i], alpha=0.3,)
    
    # Add labels, colorbar, and legend
    axes[k].set_xlabel('X-axis')
    axes[k].set_ylabel('Y-axis')
    axes[k].set_title('2D Data with Temporal Color Gradient')
    axes[k].grid(True)
    if k == n_sample_runs-1:
        fig.colorbar(cm.ScalarMappable(norm=norm, cmap='viridis'), label='time')
plt.tight_layout()
plt.show()


# iv) Compare inferred and ground truth

mu_est = pyro.get_param_store()['mu']
sigma_est = pyro.get_param_store()['sigma'] 
A_est = pyro.get_param_store()['A']

print(' A_true = {} \n A_est = {} \n'
      ' mu_true = {} \n mu_est = {} \n'
      ' sigma_true = {}\n sigma_est = {}'.format(A_true, A_est, mu_true, mu_est, sigma_true, sigma_est))
