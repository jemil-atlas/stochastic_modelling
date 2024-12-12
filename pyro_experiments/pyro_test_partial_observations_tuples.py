#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script is to test, if pyro can handle partial observations, in which the
observations are more complex than tuples of the type (obs_1, obs_2) or (obs_1, None).
Instead, the data passed to the inference algorithm will be a tuple of tuples
data = [(data_1, data_2, ... , data_n), ... (data_1, ... ,data_n)] where each element
of the list is a 
This situation can arise during subbatching of tuples containing unequal amounts
of observations. This works now but is slow due to sequential plate.

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
import random

import numpy as np
import matplotlib.pyplot as plt


# ii) Definitions

n_data_1 = 10
n_data_2 = 5
n_data_total = n_data_1 + n_data_2
torch.manual_seed(1)
pyro.set_rng_seed(1)



"""
    2. Generate data
"""


# i) Set up data distribution (=standard normal))

mu_1_true = torch.zeros([1])
mu_2_true = torch.zeros([1])
sigma_true = torch.ones([1])

extension_tensor_1 = torch.ones([n_data_1])
extension_tensor_2 = torch.ones([n_data_2])
extension_tensor_total = torch.ones([n_data_total])
data_dist_1 = pyro.distributions.Normal(loc = mu_1_true * extension_tensor_1, scale = sigma_true)
data_dist_2 = pyro.distributions.Normal(loc = mu_2_true * extension_tensor_2, scale = sigma_true)

# ii) Sample from dist to generate data

# Generate data
data_1 = data_dist_1.sample()
data_2 = data_dist_2.sample()
# data = (data_1, data_2)

# Generate None masks
# n_data_comb = n_data_1 + n_data_2
# data_1_mask = torch.ones([n_data_comb])
# data_2_mask = torch.ones([n_data_comb])
data_1_Nones = [None] * n_data_2
data_2_Nones = [None] * n_data_1

# Shuffle data together into randomly data_1, data_2 (or None)
data_1_aug = data_1.tolist() + data_1_Nones
data_1_aug = [torch.tensor(d1) if d1 is not None else None for d1 in data_1_aug]
random.shuffle(data_1_aug)
data_2_aug = data_2.tolist() + data_2_Nones
data_2_aug = [torch.tensor(d2) if d2 is not None else None for d2 in data_2_aug]
random.shuffle(data_2_aug)

data_list = [ (d1, d2) for d1, d2 in zip(data_1_aug, data_2_aug)]
data_tuples = (data_1_aug, data_2_aug)

data_1_nones = [1 if data is None else 0 for data in data_1_aug]
data_2_nones = [1 if data is None else 0 for data in data_2_aug]



"""
    3. Define stochastic model
"""


# i) Define model as normal with mean and var parameters

# def model(observations = (None, None)):
#     mu_1 = pyro.param(name = 'mu_1', init_tensor = torch.tensor([5.0])) 
#     mu_2 = pyro.param(name = 'mu_2', init_tensor = torch.tensor([5.0])) 
#     sigma = pyro.param( name = 'sigma', init_tensor = torch.tensor([5.0]))
    
#     obs_dist_1 = pyro.distributions.Normal(loc = mu_1 * extension_tensor_1, scale = sigma)
#     for k in pyro.plate(name = 'data_plate_1', size = n_data_1, dim = -1):
#         # model_sample_1 = pyro.sample('observation_1_{}'.format(k), obs_dist_1, obs = observations[k][0])
#         model_sample_1 = pyro.sample('observation_1_{}'.format(k), obs_dist_1)
        
#     obs_dist_2 = pyro.distributions.Normal(loc = mu_2 * extension_tensor_2, scale = sigma)
#     for l in pyro.plate(name = 'data_plate_2', size = n_data_2, dim = -1):
#         # model_sample_2 = pyro.sample('observation_2_{}'.format(l), obs_dist_2, obs = observations[l][1])
#         model_sample_2 = pyro.sample('observation_2_{}'.format(l), obs_dist_2)

#     return (model_sample_1, model_sample_2)

# def model(observations = (None, None)):
#     mu_1 = pyro.param(name = 'mu_1', init_tensor = torch.tensor([5.0])) 
#     mu_2 = pyro.param(name = 'mu_2', init_tensor = torch.tensor([5.0])) 
#     sigma = pyro.param( name = 'sigma', init_tensor = torch.tensor([5.0]))
    
#     obs_dist_1 = pyro.distributions.Normal(loc = mu_1 * extension_tensor_1, scale = sigma)
#     for k in pyro.plate(name = 'data_plate_1', size = n_data_1):
#         obs1_or_none = observations[k][0] if observations[0] is not None else None
#         model_sample_1 = pyro.sample('observation_1_{}'.format(k), obs_dist_1, obs = obs1_or_none)
        
#     obs_dist_2 = pyro.distributions.Normal(loc = mu_2 * extension_tensor_2, scale = sigma)
#     for l in pyro.plate(name = 'data_plate_2', size = n_data_2):
#         obs2_or_none = observations[l][1] if observations[1] is not None else None
#         model_sample_2 = pyro.sample('observation_2_{}'.format(l), obs_dist_2, obs = obs2_or_none)

#     return (model_sample_1, model_sample_2)


def model(observations = None):
    # observations is data_list
    mu_1 = pyro.param(name = 'mu_1', init_tensor = torch.tensor([5.0])) 
    mu_2 = pyro.param(name = 'mu_2', init_tensor = torch.tensor([5.0])) 
    sigma = pyro.param( name = 'sigma', init_tensor = torch.tensor([5.0]))
    
    # Set up sampling
    obs_dist_1 = pyro.distributions.Normal(loc = mu_1, scale = sigma)
    obs_dist_2 = pyro.distributions.Normal(loc = mu_2, scale = sigma)
    
    # model_sample_1 = torch.zeros([n_data_total])
    # model_sample_2 = torch.zeros([n_data_total])
    
    # sample
    sample_data_list = []
    for k in pyro.plate(name = 'data_plate', size = n_data_total):
        obs1_or_none = observations[k][0] if observations is not None else None
        obs2_or_none = observations[k][1] if observations is not None else None
        
        model_sample_1 = pyro.sample('observation_1_{}'.format(k), obs_dist_1, obs = obs1_or_none)      
        model_sample_2 = pyro.sample('observation_2_{}'.format(k), obs_dist_2, obs = obs2_or_none)

        # compile sampling data_list
        sample_data_list.append((model_sample_1, model_sample_2)) 
    return sample_data_list

model_trace = pyro.poutine.trace(model)
spec_trace = model_trace.get_trace(data_list)
spec_trace.nodes
print(spec_trace.format_shapes())



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

# This is pretty nice: we now can pass in a list of tuples, sometimes there are
# observations in there and sometimes there arent
data_svi = data_list
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




















