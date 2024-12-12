#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script is to test, if pyro can handle multiple plates in one stack when
performing sequential (non vectorized) evaluation and inference. This situation
is relevant in order to allow for Calipy to handle both sequential and vectorized
evaluation and inference with a common syntax

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
import contextlib

import numpy as np
import matplotlib.pyplot as plt


# ii) Definitions

n_data_1 = 5
n_data_2 = 3
n_data_total = n_data_1 + n_data_2
torch.manual_seed(1)
pyro.set_rng_seed(1)



"""
    2. Generate data
"""


# i) Set up data distribution (=standard normal))

mu_true = torch.zeros([1])
sigma_true = torch.ones([1])

extension_tensor = torch.ones([n_data_1, n_data_2])
data_dist = pyro.distributions.Normal(loc = mu_true * extension_tensor, scale = sigma_true)


# ii) Sample from dist to generate data

# Generate data
data = data_dist.sample()
# data_list = [(datapoint,) for datapoint in data]



"""
    3. Define stochastic model
"""


def indexfun(tuple_of_indices, vectorizable = True):
    """ Function to create a multiindex that can handle an arbitrary number of indices 
    (both integers and vectors), while preserving the shape of the original tensor.
    # Example usage 1 
    A = torch.randn([4,5])  # Tensor of shape [4, 5]
    i = torch.tensor([1])
    j = torch.tensor([3, 4])
    # Call indexfun to generate indices
    indices_A, symbol = indexfun((i, j))
    # Index the tensor
    result_A = A[indices_A]     # has shape [1,2]
    # Example usage 2
    B = torch.randn([4,5,6,7])  # Tensor of shape [4, 5, 6, 7]
    k = torch.tensor([1])
    l = torch.tensor([3, 4])
    m = torch.tensor([3])
    n = torch.tensor([0,1,2,3])
    # Call indexfun to generate indices
    indices_B, symbol = indexfun((k,l,m,n))
    # Index the tensor
    result_B = B[indices_B]     # has shape [1,2,1,4]
    """
    # Calculate the shape needed for each index to broadcast correctly
    if vectorizable == True:        
        idx = tuple_of_indices
        shape = [1] * len(idx)  # Start with all singleton dimensions
        broadcasted_indices = []
    
        for i, x in enumerate(idx):
            target_shape = list(shape)
            target_shape[i] = len(x)
            # Reshape the index to target shape
            x_reshaped = x.view(target_shape)
            # Expand to match the full broadcast size
            x_broadcast = x_reshaped.expand(*[len(idx[j]) if j != i else x_reshaped.shape[i] for j in range(len(idx))])
            broadcasted_indices.append(x_broadcast)
            indexsymbol = 'vectorized'
    else:
        broadcasted_indices = tuple_of_indices
        indexsymbol = broadcasted_indices
    
    # Convert list of broadcasted indices into a tuple for direct indexing
    return tuple(broadcasted_indices), indexsymbol       
        
    
class CalipyContext:
    def __init__(self, pyro_plate, vectorizable=True):
        self.vectorizable = vectorizable
        self.plate = pyro_plate
        self._iterator = None

    def __iter__(self):
        return self._entered_plate
    
    def __enter__(self):
        if self.vectorizable:
            # Enter the other context and store the returned value
            self._entered_plate = [self.plate.__enter__()]
            # Return the value from the other context
            return self._entered_plate
        else:
            self._entered_plate = self.plate.__enter__()
            # Return the value from the other context
            return self._entered_plate

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Exit the other context
        return self.plate.__exit__(exc_type, exc_val, exc_tb)


# Example usage:
plate_ex = pyro.plate('example_plate', size = 10)
# Non-vectorized mode: acts like an iterable (individual items)
print("Non-vectorized mode:")
strange_context = CalipyContext(plate_ex, vectorizable=False)
with strange_context as i:
    print(i)
    for k in i:
        print(k)  # Prints 0, 1, 2

# Vectorized mode: acts like a context and iterable (single iteration with the list)
print("\nVectorized mode:")
strange_context = CalipyContext(plate_ex, vectorizable=True)
with strange_context as i:
    print(i)
    for k in i:
        print(k)  # Prints [0, 1, 2]




def model(observations = None):
    # observations is data_list
    mu = pyro.param(name = 'mu', init_tensor = torch.tensor([5.0])) 
    sigma = pyro.param( name = 'sigma', init_tensor = torch.tensor([5.0]))
    
    
    # BASIC
    # # Set up sampling    
    # # vectorizable version - basic
    # obs_dist = pyro.distributions.Normal(loc = mu*extension_tensor, scale = sigma)
    # plate_1 = pyro.plate('batch_plate_1', size = n_data_1, dim = -2)
    # plate_2 = pyro.plate('batch_plate_2', size = n_data_2, dim = -1)
    # with plate_1, plate_2:
    #     obs = pyro.sample('observation', obs_dist, obs = observations)    
    
    # # sequential version - basic
    # obs_dist = pyro.distributions.Normal(loc = mu, scale = sigma)
    # plate_1 = pyro.plate('batch_plate_1', size = n_data_1)
    # plate_2 = pyro.plate('batch_plate_2', size = n_data_2)
    # for i in plate_1:
    #     for j in plate_2:
    #         obs_or_None = observations[i,j] if observations is not None else None
    #         obs = pyro.sample('observation_{}{}'.format(i,j), obs_dist, obs = obs_or_None)
    
    
    # INDEX
    # # vectorizable version - index
    # vectorizable = True
    # obs_dist = pyro.distributions.Normal(loc = mu, scale = sigma)
    # plate_1 = pyro.plate('batch_plate_1', size = n_data_1)
    # plate_2 = pyro.plate('batch_plate_2', size = n_data_2)
    # with plate_2 as j: 
    #     with plate_1 as i:
    #         indices, indexsymbol = indexfun((i,j), vectorizable = vectorizable)
    #         obs_or_None = observations[indices] if observations is not None else None
    #         obs = pyro.sample('observation_{}'.format(indexsymbol), obs_dist, obs = obs_or_None)
    
    # # sequential version - index
    # vectorizable = False
    # obs_dist = pyro.distributions.Normal(loc = mu, scale = sigma)
    # plate_1 = pyro.plate('batch_plate_1', size = n_data_1)
    # plate_2 = pyro.plate('batch_plate_2', size = n_data_2)
    # for j in plate_2:
    #     for i in plate_1:
    #         indices, indexsymbol = indexfun((i,j), vectorizable = vectorizable)
    #         obs_or_None = observations[indices] if observations is not None else None
    #         obs = pyro.sample('observation_{}'.format(indexsymbol), obs_dist, obs = obs_or_None)
    
    # # UNIFIED
    # vectorizable = True
    # obs_dist = pyro.distributions.Normal(loc = mu, scale = sigma)
    # plate_1 = pyro.plate('batch_plate_1', size = n_data_1)
    # plate_2 = pyro.plate('batch_plate_2', size = n_data_2)
    # with handle_plate(plate_2, vectorizable) as j:
    #     with handle_plate(plate_1, vectorizable) as i:
    #         print(i,j)
    #         indices, indexsymbol = indexfun((i,j), vectorizable = vectorizable)
    #         obs_or_None = observations[indices] if observations is not None else None
    #         obs = pyro.sample('observation_{}'.format(indexsymbol), obs_dist, obs = obs_or_None)
    

    # return obs

model_trace = pyro.poutine.trace(model)
spec_trace = model_trace.get_trace(data)
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
data_svi = data
for step in range(1000):
    loss = svi.step(data_svi)
    if step % 100 == 0:
        print('Loss = ' , loss)



"""
    5. Plots and illustrations
"""


# i) Print results

print('True mu = {}, True sigma = {} \n Inferred mu = {:.3f}, Inferred sigma = {:.3f} \n mean = {:.3f}'
      .format(mu_true, sigma_true, 
              pyro.get_param_store()['mu'].item(),
              pyro.get_param_store()['sigma'].item(),
              torch.mean(data)))


# ii) Plot data

fig1 = plt.figure(num = 1, dpi = 300)
plt.hist(data.flatten().detach().numpy())
plt.title('Histogram of data')
















