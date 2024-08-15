""" 
The goal of this script is to perform simulation and inference using pyro. We want
to test the torchdim module with multiple batch_dims and event_dims
For this, do the following:
    1. Definitions and imports
    2. Build stochastic model
    3. Inference
    4. Plots and illustrations
"""



"""
    1. Definitions and imports ------------------------------------------------
"""


# ii) Imports


import torch
import numpy as np
import matplotlib.pyplot as plt

import pyro
import pyro.distributions as dist
# import functorch
from functorch.dim import dims
# from torch import where


# ii) Definitions

n_data_1 = 100
n_data_2 = 20


pyro.set_rng_seed(1)



"""
    2. Build stochastic model -------------------------------------------------
"""


# i) Simulate some data using true model

# mu_true = torch.ones([2,3])
mu_true = torch.tensor([[1.0,2.0,3.0],[4.0,5.0,6.0]])
sigma_true = 0.1
data=pyro.distributions.Normal(mu_true,sigma_true).sample([n_data_1, n_data_2])

# functorch dimensions
bd_1, bd_2 = dims(2)
ed_1, ed_2 = dims(2)
batch_dims = (bd_1, bd_2)
event_dims = (ed_1, ed_2)
dim_tuple = batch_dims + event_dims

data_fc = data[dim_tuple]


# ii) Invoke observations, latent variables, parameters, model

# def model(observations = None):
#     mu=pyro.param("mu", init_tensor = torch.zeros([1,1,ed_1.size, ed_2.size]))
    
    # with pyro.plate("batch_plate_1", n_data_1, dim = -4):
    #     with pyro.plate("batch_plate_2", n_data_2, dim = -3):    
    #         return pyro.sample("obs",dist.Normal(mu,sigma_true),obs = observations)

def model(observations = None):
    
    # trivial_dims = dims(sizes = [1 for k in range(len(batch_dims))])
    # mu_dims = trivial_dims + event_dims
    
    mu_dims = batch_dims + event_dims
    mu = pyro.param("mu", init_tensor = torch.zeros([dim.size for dim in mu_dims]))
        
    mu_fc = mu[(mu_dims)]
    obs_dist = dist.Normal(mu_fc.order(*mu_dims),sigma_true).to_event(len(event_dims))
    
    # dim_pos_batch = [-(k+1)- len(event_dims) for k in range(len(batch_dims))] # generates -4, -3
    # above is incorrect since plate measures to left of first event dim, not total position
    dim_pos_batch = [-(k+1) for k in range(len(batch_dims))]   # generates -2, -1 
    dim_pos_batch.reverse()
    
    with pyro.plate("batch_plate_1",bd_1.size, dim = dim_pos_batch[0]):
        with pyro.plate("batch_plate_2",bd_2.size, dim = dim_pos_batch[1]):
            return pyro.sample("obs",obs_dist,obs = observations)



# # Multiple dim assignment test:
# tens = torch.randn(2,3,4)
# batch = dims()
# event_1, event_2 = dims(2)
# event_dims = (event_1,event_2)

# # using iterative assignment
# tens_fc = tens
# for dim in (batch, *event_dims):
#     tens_fc = tens_fc[dim] 

# # using assignment function
# def assign_dims(tensor, dims):
#     for dim in dims:
#         tensor = tensor[dim]
#     return tensor

# tens_2_fc = assign_dims(tens, (batch, event_dims))


# # using direct tuple construction
# dim_tuple = (batch, *event_dims)
# tens_3_fc = tens[dim_tuple]





"""
    3. Inference --------------------------------------------------------------
"""


# i) Create the guide

def guide(observations = None):
    pass


# ii) Run the optimization

adam = pyro.optim.Adam({"lr": 0.02})
elbo = pyro.infer.Trace_ELBO()
svi = pyro.infer.SVI(model, guide, adam, elbo)

losses = []
for step in range(1000):  
    loss = svi.step(data_fc.order(*dim_tuple))
    losses.append(loss)
    if step % 100 == 0:
        print(loss)

for name, value in pyro.get_param_store().items():
    print(name, pyro.param(name).data.cpu().numpy())



"""
    4. Plots and illustrations -----------------------------------------------
"""


# i) Plot the loss

plt.figure(figsize=(5, 2), dpi=300)
plt.plot(losses)
plt.xlabel("SVI step")
plt.ylabel("ELBO loss");






















