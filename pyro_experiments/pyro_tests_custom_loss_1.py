#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The goal of this script is to implement a custom loss in pyro and use it to optimize
a model. We compare the results to standard ELBO and illustrate the difference.
For this, do the following:
    1. Imports and definitions
    2. Generate data
    3. Model and guide
    4. Modified loss and training
    5. Plots and illustrations
The example we are going to implement here is to estimate the mean of normal 
distribution based on data including outliers. Trace_ELBO should be non-robust
and we want to change Trace_ELBO to a simple L1 loss leading to robust fits.
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

n_data = 100
n_outliers = 10
pyro.clear_param_store()
np.random.seed(0)
torch.random.manual_seed(0)


"""
    2. Generate data
"""


# i) Generate data (normally distributed))

mu = 1
sigma = 0.1
data_no_outliers = np.random.normal(mu, sigma, size = [n_data,1])


# ii) Construct outliers

outliers_index = np.random.randint(0,n_data-1, size = [n_outliers,])
outliers = np.zeros([n_data,1])
outliers[outliers_index] = np.random.normal(0, 5, size = [n_outliers,1])


# iii) Data assembly

data = data_no_outliers + outliers
data_tens = torch.tensor(data)



"""
    3. Model and guide
"""


# i) Model for observations

def model(y_obs = None):
    mu = pyro.param('mu', init_tensor = torch.zeros([1,1]))
    sigma = pyro.param('sigma', init_tensor = torch.ones([1,1]), constraint = pyro.distributions.constraints.positive)
    y_dist = pyro.distributions.Normal(mu, sigma).expand([n_data,1]).to_event(1)
    # Sample with 1st dim = batch_dim of size n_data, second dim event_dim of size 1
    with pyro.plate('batch_plate', size = n_data, dim = -1):
        y_sample = pyro.sample('y_sample', y_dist, obs = y_obs)
    return y_sample


# ii) Guide for latents

def guide(y_obs = None):
    pass



"""
    4. Modified loss and training
"""


# i) Standard loss and training

# optimizer = pyro.optim.Adam({"lr" : 1e-2})
# Trace_ELBO_loss = pyro.infer.Trace_ELBO()
# svi = pyro.infer.SVI(model, guide, optim = optimizer, loss = Trace_ELBO_loss)

# n_epoch = 1000
# for k in range(n_epoch):
#     svi.step(data_tens)
#     if k % 100 == 0:
#         loss = svi.evaluate_loss(data_tens)
#         print('Loss Trace ELBO : {} '.format(loss))


# # ii) Equivalent low level loss construction and training    

# loss_fn = lambda model, guide: pyro.infer.Trace_ELBO().differentiable_loss(model, guide, data_tens)
# # Record params by going through the execution trace
# loss_trace = pyro.poutine.trace(loss_fn, param_only = True).get_trace(model, guide)
# params = [site['value'].unconstrained() for site in loss_trace.nodes.values()]

# # Optimization loop
# optimizer = torch.optim.Adam(params, lr=0.01)
# n_epoch = 1000
# for k in range(n_epoch):
#     # compute loss
#     loss = loss_fn(model, guide)
#     loss.backward()
#     # take a step and zero the parameter gradients
#     optimizer.step()
#     optimizer.zero_grad()
#     if k % 100 == 0:
#         loss_val = loss.item()
#         print('Loss Trace ELBO : {} '.format(loss_val))



# # iii) Modified low-level loss and training : L1 regularizer on params

# def L1_regularizer(params):
#     reg_loss = 0.0
#     for param in params:
#         reg_loss = reg_loss + torch.norm(param, p = 1)
#     return reg_loss

# loss_fn = lambda model, guide: pyro.infer.Trace_ELBO().differentiable_loss(model, guide, data_tens)
# # Record params by going through the execution trace
# loss_trace = pyro.poutine.trace(loss_fn, param_only = True).get_trace(model, guide)
# # Optimization only possible over leaf-tensors -> unconstrained version, i.e.
# # mu, log(sigma)
# params = [site['value'].unconstrained() for site in loss_trace.nodes.values()]

# # Optimization loop
# optimizer = torch.optim.Adam(params, lr=0.01)
# n_epoch = 1000
# for k in range(n_epoch):
#     # compute loss
#     loss = 0*loss_fn(model, guide) + 10* n_data* L1_regularizer(params)
#     loss.backward()
#     # take a step and zero the parameter gradients
#     optimizer.step()
#     optimizer.zero_grad()
#     if k % 100 == 0:
#         loss_val = loss.item()
#         print('Loss Trace ELBO with L1 reg : {} '.format(loss_val))

        
# iv) SVI-compatible compact elbo reproduction

# # simple_elbo takes a model, a guide, and their respective arguments as inputs
# def simple_elbo(model, guide, *args, **kwargs):
#     # run the guide and trace its execution
#     guide_trace = pyro.poutine.trace(guide).get_trace(*args, **kwargs)
#     # run the model and replay it against the samples from the guide
#     model_trace = pyro.poutine.trace(
#         pyro.poutine.replay(model, trace=guide_trace)).get_trace(*args, **kwargs)
#     # construct the elbo loss function
#     return -1*(model_trace.log_prob_sum() - guide_trace.log_prob_sum())

# optimizer = pyro.optim.Adam({"lr": 0.01})
# svi = pyro.infer.SVI(model, guide, optimizer, loss=simple_elbo)

# n_epoch = 1000
# for k in range(n_epoch):
#     svi.step(data_tens)
#     if k % 100 == 0:
#         loss = svi.evaluate_loss(data_tens)
#         print('Loss Trace ELBO : {} '.format(loss))
        
        
# v) SVI-compatible modified loss

# We want to take the L1 loss between mu and y_sample as an additional penalty
# i.e. ELBO_with_L1 = ELBO + \|mu - y\|_1
def ELBO_with_L1(model, guide, *args, **kwargs):
    # run the guide and trace its execution
    guide_trace = pyro.poutine.trace(guide).get_trace(*args, **kwargs)
    # run the model and replay it against the samples from the guide
    model_trace = pyro.poutine.trace(
        pyro.poutine.replay(model, trace=guide_trace)).get_trace(*args, **kwargs)
    # construct the elbo loss function
    elbo = -1*(model_trace.log_prob_sum() - guide_trace.log_prob_sum())
    # construct \|mu - y\|_1
    l1_penalty = torch.norm(model_trace.nodes['mu']['value'] - model_trace.nodes['y_sample']['value'],p=1)
    return 1*elbo + 1* l1_penalty

optimizer = pyro.optim.Adam({"lr": 0.01})
svi = pyro.infer.SVI(model, guide, optimizer, loss=ELBO_with_L1)

n_epoch = 1000
for k in range(n_epoch):
    svi.step(data_tens)
    if k % 100 == 0:
        loss = svi.evaluate_loss(data_tens)
        print('Loss Trace ELBO with L1 norm : {} '.format(loss))
        
    
        
"""
    5. Plots and illustrations
"""


# i) Print out params

for name, param in pyro.get_param_store().items():
    print('Param {} : {}'.format(name, param))

# Observable impacts:
    
# L1 norm regularization of params pushes mu to 0 and sigma to 1 since 
# \|log sigma\| is penalized due to the parameters being unconstrained.
#
# L1 norm addition to the loss function makes the estimator for the mean 
# mu more robust. When the coefficient of the elbo is chosen to be 0, we 
# recreate simple \|mu - y\|_1 l1 minimization. The sigma parameter is 
# not affected.



plt.figure(1, figsize = (10,10), dpi = 300)
plt.hist(data_tens.numpy())






































































