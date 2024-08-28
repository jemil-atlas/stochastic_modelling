#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script isto test, if pyro can handle partial observations, in which the
observations are tuples of the type (obs_1, obs_2) or (obs_1, None). This situation
can arise during subbatching of tuples containing unequal amounts of observations.

For this, do the following:
    1. Imports and definitions
    2. Generate data
    3. Define stochastic model
    4. Perform inference
    5. Plots and illustrations
    
The script is meant solely for educational and illustrative purposes. Written by
Jemil Avers Butt, Atlas optimization GmbH, www.atlasoptimization.com.
"""



"""
    1. Imports and definitions
"""


# i) Imports

import pyro
import torch

import matplotlib.pyplot as plt


# ii) Definitions

n_data_1 = 200
n_data_2 = 50
torch.manual_seed(0)
pyro.set_rng_seed(0)



"""
    2. Generate data
"""


# i) Set up data distribution (=standard normal))

mu_1_true = torch.zeros([1])
mu_2_true = torch.zeros([1])
sigma_true = torch.ones([1])

extension_tensor_1 = torch.ones([n_data_1])
extension_tensor_2 = torch.ones([n_data_2])
data_dist_1 = pyro.distributions.Normal(loc = mu_1_true * extension_tensor_1, scale = sigma_true)
data_dist_2 = pyro.distributions.Normal(loc = mu_2_true * extension_tensor_2, scale = sigma_true)

# ii) Sample from dist to generate data

data_1 = data_dist_1.sample()
data_2 = data_dist_2.sample()
data = (data_1, data_2)



"""
    3. Define stochastic model
"""


# i) Define model as normal with mean and var parameters

def model(observations = None):
    mu_1 = pyro.param(name = 'mu_1', init_tensor = torch.tensor([5.0])) 
    mu_2 = pyro.param(name = 'mu_2', init_tensor = torch.tensor([5.0])) 
    sigma = pyro.param( name = 'sigma', init_tensor = torch.tensor([5.0]))
    
    obs_dist_1 = pyro.distributions.Normal(loc = mu_1 * extension_tensor_1, scale = sigma)
    with pyro.plate(name = 'data_plate_1', size = n_data_1, dim = -1):
        model_sample_1 = pyro.sample('observation_1', obs_dist_1, obs = observations[0])
        
    obs_dist_2 = pyro.distributions.Normal(loc = mu_2 * extension_tensor_2, scale = sigma)
    with pyro.plate(name = 'data_plate_2', size = n_data_2, dim = -1):
        model_sample_2 = pyro.sample('observation_2', obs_dist_2, obs = observations[1])

    return (model_sample_1, model_sample_2)



"""
    4. Perform inference
"""


# i) Set up guide

def guide(observations = None):
    pass


# ii) Set up inference


adam = pyro.optim.Adam({"lr": 0.01})
elbo = pyro.infer.Trace_ELBO()
svi = pyro.infer.SVI(model, guide, adam, elbo)


# iii) Perform svi

# The following would lead to correct solutions: 
# data_svi = data

# but what happens if i replace data_2 by a bunch of nones and eleiminate these 
# these observations? can svi handle data = (data_1, None)?
# data_svi = (data_1, None)
data_svi = (None, data_2)
for step in range(1000):
    loss = svi.step(data_svi)
    if step % 100 == 0:
        print('Loss = ' , loss)



"""
    5. Plots and illustrations
"""


# i) Print results

print('True mu_1 = {}, True_mu_2 = {},  True sigma = {} \n Inferred mu_1 = {:.3f}, Inferred mu_2 = {:.3f}, Inferred sigma = {:.3f} \n mean_1 = {:.3f}, mean_2 = {:.3f}'
      .format(mu_1_true, mu_2_true, sigma_true, 
              pyro.get_param_store()['mu_1'].item(),
              pyro.get_param_store()['mu_2'].item(),
              pyro.get_param_store()['sigma'].item(),
              torch.mean(data_1), torch.mean(data_2)))


# ii) Plot data

fig1 = plt.figure(num = 1, dpi = 300)
plt.hist(data_1.detach().numpy())
plt.hist(data_2.detach().numpy())
plt.title('Histogram of data')


# # iii) Plot distributions

# t = torch.linspace(-3,3,100)
# inferred_dist = pyro.distributions.Normal( loc = pyro.get_param_store()['mu'], 
#                                           scale = pyro.get_param_store()['sigma'])

# fig2 = plt.figure(num = 2, dpi = 300)
# plt.plot(t, torch.exp(data_dist.log_prob(t)), color = 'k', label = 'true', linestyle = '-')
# plt.plot(t, torch.exp(inferred_dist.log_prob(t)).detach(), color = 'k', label = 'inferred', linestyle = '--')
# plt.title('True and inferred distributions')




















