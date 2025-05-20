#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script is meant to test subbatching and the impact it has on training convergence.
We take a very simple problem involving an unknown mean that needs to be estimated
and solve it by subbatching the data. We investigate, how the amount of data in
the batch influences convergence speed.
For this, do the following:
    1. Imports and definitions
    2. Simulate some data
    3. Define model and guide
    4. Perform inference
    
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

torch.manual_seed(0)


"""
    2. Simulate some data
"""


# i)  Generate Normally distributed data

mu_true = 1
sigma_true = 2

n_data = 1001
subsample_size = 1000
data = torch.distributions.Normal(loc = mu_true, scale = sigma_true).sample([n_data])


# ii) Build dataloader

class SubbatchDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self. data = data
        
    def __getitem__(self, idx):
        return (self.data[idx, ...], idx)
    
    def __len__(self):
        return data.shape[0]


subbatch_dataset = SubbatchDataset(data)
subbatch_dataloader = torch.utils.data.DataLoader(subbatch_dataset, batch_size = subsample_size, shuffle = True)


# iii) Print diagnostic quantities

index_list = []
datablock_list = []
mean_list = []
for data_block, index_tensor in subbatch_dataloader:
    print("Data block shape:", data_block.shape)
    print("Index tensor shape:", index_tensor.shape)
    print("data block:", data_block)
    print("Indices:", index_tensor)
    index_list.append(index_tensor)
    datablock_list.append(data_block)
    mean_list.append(torch.mean(data_block,0))
    print("----")
    
    

"""
    3. Define model and guide
"""


# i) Defining the model 

# Can be done either by inferring the subsample size and passing it to plate
def model(observations = None, subsample_indices = None):
    # Declare parameters
    mu = pyro.param(name = 'mu', init_tensor = torch.tensor([0.0]))
    sigma = pyro.param(name = 'sigma', init_tensor = torch.tensor([1.0]), constraint = pyro.distributions.constraints.positive)
    
    # Declare distribution and sample n_data independent samples, condition on observations
    model_dist = pyro.distributions.Normal(loc = mu, scale = sigma)
    obs_subsample_size = len(subsample_indices) if subsample_indices is not None else subsample_size
    with pyro.plate('batch_plate', size = n_data, subsample_size = obs_subsample_size, dim =-1):    
        model_sample = pyro.sample('model_sample', model_dist, obs = observations)
    
#     return model_sample

# # ... or can be done by passing the subsample_indices directly to the plate
# def model(observations = None, subsample_indices = None):
#     # Declare parameters
#     mu = pyro.param(name = 'mu', init_tensor = torch.tensor([0.0]))
#     sigma = pyro.param(name = 'sigma', init_tensor = torch.tensor([1.0]), constraint = pyro.distributions.constraints.positive)
    
#     # Declare distribution and sample n_data independent samples, condition on observations
#     model_dist = pyro.distributions.Normal(loc = mu, scale = sigma)
#     with pyro.plate('batch_plate', size = n_data, subsample = subsample_indices, dim =-1):    
#         model_sample = pyro.sample('model_sample', model_dist, obs = observations)
    
#     return model_sample



"""
    4. Perform inference
"""


# iii) Setting up inference

# Variational distribution
def guide(observations = None, subsample_indice = None):
    pass

# Optimization options
adam = pyro.optim.Adam({"lr": 0.01})
elbo = pyro.infer.Trace_ELBO()
svi = pyro.infer.SVI(model, guide, adam, elbo)


# iv) Inference and print results

# Handle DataLoader case
loss_sequence = []
for epoch in range(3000):
    epoch_loss = 0
    for batch_data, subsample_indices in subbatch_dataloader:
        loss = svi.step(batch_data, subsample_indices)
        epoch_loss += loss
    
    epoch_loss /= len(subbatch_dataloader)
    loss_sequence.append(epoch_loss)

    if epoch % 100 == 0:
        print(f'epoch: {epoch} ; loss : {epoch_loss}')
    

print('True mu = {}, True sigma = {} \n Inferred mu = {:.3f}, Inferred sigma = {:.3f}, \n Mean = {:.3f}, \n Mean of batch means = {:.3f}'
      .format(mu_true, sigma_true, 
              pyro.get_param_store()['mu'].item(),
              pyro.get_param_store()['sigma'].item(),
              torch.mean(data),
              torch.mean(torch.hstack(mean_list))))

# Interpretation:
# We notice here a discrepancy between the arithmetic mean of the whole dataset
# and the inferred mean based on two batches of size (1000,1). This is not due
# to some error in pyro, which uses the standard scaling. This scaling ensures
# that an unbiased estimator for the log probs and the gradients are built - but
# not that their sum adds to the gradient derived from the full dataset. This is
# by design as multiple noisy updates per epoch often behave better than one 
# single update to avoid local minima and converge faster in terms of wall-clock 
# time. Overall this increase of variance in gradient estimators is intentional
# and rescaling and accumulation of the gradients would diminish the advantages
# gained by minibatch training (many but noisy updates, small memory footprint).
# In cases of problems with the subbatched training, try some subbatch-specific
# troubleshooting instead: drop the last batch, ensure the subbatches are evenly
# sized or increase the subbatch size.