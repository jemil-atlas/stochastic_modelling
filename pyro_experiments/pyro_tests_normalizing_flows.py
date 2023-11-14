#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The goal of this script is to test normalizing flows. A bimodal dataset is 
generated and normalizing flows are used to fit a bimodal distribution to this
dataset.
For this, do the following:
    1. Imports and definitions
    2. Stochastic model
    3. Normalizing flow
    4. Plots and illustrations
"""

"""
    1. Imports and definitions
"""


# i) imports

import torch
import pyro
import numpy as np
import matplotlib.pyplot as plt


# ii) Definitions

n_data = 1000



"""
    2. Stochastic model
"""

# i) Generate data


# Bimodal <- this does not work with Gaussian priors and therefore we need the 
# normalizing flow construction
y_data = np.random.randint(0,2, size = [n_data,1]) + np.random.normal(0, 0.1, size = [n_data,1])


# ii) Set up base distribution
base_dist = pyro.distributions.Normal(torch.zeros(1), torch.ones(1))



"""
    3. Normalizing flow
"""


# i) Set up transforms

spline_transform = pyro.distributions.transforms.spline_coupling(1, count_bins=16)
flow_dist = pyro.distributions.TransformedDistribution(base_dist, [spline_transform])


# ii) Perform optimization

steps = 5000
dataset = torch.tensor(y_data, dtype=torch.float)
optimizer = torch.optim.Adam(spline_transform.parameters(), lr=5e-3)
for step in range(steps+1):
    optimizer.zero_grad()
    loss = -flow_dist.log_prob(dataset).mean()
    loss.backward()
    optimizer.step()
    flow_dist.clear_cache()

    if step % 500 == 0:
        print('step: {}, loss: {}'.format(step, loss.item()))

y_flow = flow_dist.sample(torch.Size([1000,])).detach().numpy()



"""
    4. Plots and illustrations
"""

# i) Histogram of data and flow_dist

plt.figure(1, figsize = (5,5), dpi = 300)
plt.hist(y_data, bins = 50, color = 'blue', alpha = 0.5, label = 'original data', density = True)
plt.hist(y_flow, bins = 50,  color = 'green', alpha = 0.5, label = 'flow samples', density = True)
plt.xlabel('value')
plt.ylabel('nr occurrences')
plt.legend()
plt.title('Histogram of data')










