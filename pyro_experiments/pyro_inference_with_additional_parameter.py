""" 
The goal of this script is to perform simulation and inference using pyro. We 
will survey the impact of having the model depend on additional input parameters.
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


# ii) Definitions

pyro.clear_param_store()
n_simu=100


"""
    2. Build stochastic model -------------------------------------------------
"""


# i) Simulate some data using true model

mu_true = 1
sigma_true = 1
theta_sample = torch.tensor(np.random.normal(0,1,[n_simu,1]))
data = torch.tensor(np.random.normal(mu_true,sigma_true,[n_simu,1])) + theta_sample


# ii) Invoke observations, latent variables, parameters, model

def model(theta, data=None):
    mu = pyro.param("mu",lambda:torch.randn(()))
    
    with pyro.plate("data",n_simu, dim = -2):
        return pyro.sample("x",dist.Normal(theta + mu,sigma_true),obs=data)



"""
    3. Inference --------------------------------------------------------------
"""


# i) Create the guide

auto_guide = pyro.infer.autoguide.AutoNormal(model)


# ii) Run the optimization

adam = pyro.optim.Adam({"lr": 0.01})
elbo = pyro.infer.Trace_ELBO()
svi = pyro.infer.SVI(model, auto_guide, adam, elbo)

losses = []
for step in range(1000):  # Consider running for more steps.
    loss = svi.step(theta_sample, data)
    losses.append(loss)
    if step % 100 == 0:
        print("Elbo loss: {}".format(loss))

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






















