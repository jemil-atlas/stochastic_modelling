""" 
The goal of this script is to perform simulation and inference using pyro.
For this, do the following:
    1. Definitions and imports
    2. Build stochastic model
    3. Inference
    4. Plots and illustrations
"""



"""
    1. Definitions and imports ------------------------------------------------
"""

# NEsted tensors as of now dont support autograd so you can forget about using it in
# pyro right now.

# ii) Imports


import torch
import numpy as np
import matplotlib.pyplot as plt

import pyro
import pyro.distributions as dist
# import functorch
from functorch.dim import dims
from torch import where


# ii) Definitions

n_batch = [3]
n_sample = [5,10,15]
n_event = [2]


pyro.set_rng_seed(1)


# # nested use
# t1 = torch.ones([5])
# t2 = torch.ones([10])
# tnested = torch.nested.nested_tensor([t1,t2], layout = torch.jagged, requires_grad = True)
# tnested.unbind()


"""
    2. Build stochastic model -------------------------------------------------
"""


# i) Simulate some data using true model

mu_true=torch.ones([2])
sigma_true=0.1
data_0=pyro.distributions.Normal(mu_true,sigma_true).sample([n_sample[0]])
data_1=pyro.distributions.Normal(mu_true,sigma_true).sample([n_sample[1]])
data_2=pyro.distributions.Normal(mu_true,sigma_true).sample([n_sample[2]])

data_nested = torch.nested.nested_tensor([data_0, data_1, data_2], layout = torch.jagged)




# ii) Invoke observations, latent variables, parameters, model

def model(observations = None):
    mu=pyro.param("mu", init_tensor = torch.zeros([1,n_event[0]]))
    mu_0 = mu*torch.ones([n_sample[0],1])
    mu_1 = mu*torch.ones([n_sample[1],1])
    mu_2 = mu*torch.ones([n_sample[2],1])
    mu_nested = torch.nested.nested_tensor([mu_0, mu_1, mu_2], layout = torch.jagged)
    
    with pyro.plate("batch_plate",n_data, dim = -2):
        return pyro.sample("obs",dist.Normal(mu,sigma_true),obs = observations)



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
    loss = svi.step(data_fc.order(batch_dim, event_dim))
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













dims()








