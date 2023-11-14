#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script aims to illustrate how parameters determining the observations can be
optimized so as to fulfill some adherence to a distribution.
For this, do the following:
    1. Imports and definitions
    2. Define model and guide
    3. Inference
    4. Plots and illustrations
"""


"""
    1. Imports and definitions
"""


import pyro
import pyro.distributions as dist
import pyro.optim as optim
from pyro.infer import SVI, Trace_ELBO
import torch

n_samples = 100
pyro.clear_param_store()



"""
    2. Define model and guide
"""


def model():
    theta = pyro.param('theta', torch.tensor(1,dtype=torch.float32))
    obs_dist = dist.Normal(2., 0.1)
    with pyro.plate('batch_plate', size = n_samples, dim = -1):
        obs = pyro.sample("obs", obs_dist, obs=f_of_theta(theta))
    return obs


def guide():
    # You can put some priors on theta here
    pass

def f_of_theta(theta):
    fun_val = torch.sqrt(theta) 
    return fun_val



"""
    3. inference
"""

# Initialize optimizer and SVI
optimizer = optim.Adam({"lr": 0.01})
svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

# Optimization loop to optimize theta
theta = 0.5  # Initialize theta
num_steps = 1000
for step in range(num_steps):
    loss = svi.step()
    # Update theta based on your optimization algorithm, or it might be updated in your guide function
    if step % 100 == 0:
        print('Step : {}  Loss : {}'.format(step, loss))
    
    

"""
    4. Plots and illustrations
"""
    
for name, param in pyro.get_param_store().items():
    print( '{} : {}'.format(name, param))