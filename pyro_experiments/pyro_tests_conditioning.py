"""
The goal of this script is to showcase how conditioning can be used to create
conditional distributions and sample from them and how this interacts and 
facilitates inference via mcmc.

For this, do the following:
    1. Imports and definitions
    2. Create stochastic model
    3. Condition on observation
    4. Plots and illustrations
"""


"""
    1. Imports and definitions
"""


# i) imports

import torch
import pyro
import numpy as np
import pyro.distributions as dist
import matplotlib.pyplot as plt


# ii) Definitions

num_samples = 1000



"""    
    2. Create stochastic model
"""


# i) Fixed parameters

z_mean = torch.zeros(1)
z_cov = torch.eye(1)


# ii) Model with two sample sites

def model():
    z = pyro.sample('z',dist.Normal(z_mean,z_cov))
    x = pyro.sample('x',dist.Normal(z,1))
    return x


# iii) Generate samples from the original

samples_original = np.zeros(num_samples)
for k in range(num_samples):
    samples_original[k] = model()



"""
    3. Condition on observation
"""


# i) Build model that is conditioned on having observed z = 1

observation = torch.ones(1)
conditional_model = pyro.condition(model, data={"z": observation})


# ii) Sample from the conditional
 
samples_conditional = np.zeros(num_samples)
for k in range(num_samples):
    samples_conditional[k] = conditional_model()



"""
    4. Plots and illustrations
"""


# i) Plot original dataset

plt.figure(1,dpi =300)
plt.hist(samples_original)    
plt.title('The original distribution')


# i) Plot conditional dataset

plt.figure(2,dpi =300)
plt.hist(samples_conditional)    
plt.title('The conditional distribution')

