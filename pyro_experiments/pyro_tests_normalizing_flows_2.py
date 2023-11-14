#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The goal of this script is to test normalizing flows in a multivariate setting. 
A bounded dataset is generated and normalizing flows are used to fit a a distribution
that should inherit the boundedness of the dataset.
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

n_time = 10
n_simu = 1000

t = np.linspace(0,1,n_time)



"""
    2. Stochastic model
"""

# i) Generate distributional parameters

mean_y = np.zeros([n_time])
cov_y = np.zeros([n_time,n_time])

d_t = 0.3
cov_fun = lambda s,t : 0.5*np.exp(-((s-t)/d_t)**2)

for k in range(n_time):
    for l in range(n_time):
        cov_y[k,l] = cov_fun(t[k], t[l])


# ii) Generate data & squeeze into interval [-1,1]

y_data = np.zeros([n_simu, n_time])
for k in range(n_simu):
    y_data[k,:] = np.random.multivariate_normal(mean_y, cov_y)

y_data = np.clip(y_data, -1, 1)


# iii) Set up base distribution
base_dist = pyro.distributions.MultivariateNormal(loc = torch.zeros([n_time]), covariance_matrix = torch.eye(n_time))



"""
    3. Normalizing flow
"""


# i) Set up transforms

spline_transform = pyro.distributions.transforms.spline_coupling(n_time, count_bins=8)
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

# i) Side-byside data comparison

# plt.figure(1, figsize=(10, 6), dpi = 300)
# plt.plot(y_data, label="y_data", color="blue")
# plt.plot(y_flow, label="y_flow", color="red")
# plt.xlabel("t")
# plt.ylabel("Value")
# plt.title("y_data and y_flow vs. t")
# plt.legend()
# plt.grid(True)
# plt.show()


# Create a vertically aligned subplot
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))

nr_plots = np.min([20, n_simu])
# Plot all timeseries of y_data on the first subplot
for idx in range(nr_plots):
    axes[0].plot(t, y_data[idx], label=f"y_data_{idx+1}")

# Plot all timeseries of y_flow on the second subplot
for idx in range(nr_plots):
    axes[1].plot(t, y_flow[idx], label=f"y_flow_{idx+1}")

# Setting labels and titles
axes[0].set_title("y_data timeseries")
axes[0].set_ylabel("Value")
axes[1].set_title("y_flow timeseries")
axes[1].set_xlabel("t")
axes[1].set_ylabel("Value")

# Displaying the plot
plt.tight_layout()
plt.show()


# ii) Histogram of data and flow_dist

plt.figure(2, figsize = (5,5), dpi = 300)
plt.hist(y_data[:,0], bins = 50, color = 'blue', alpha = 0.5, label = 'original data', density = True)
plt.hist(y_flow[:,0], bins = 50,  color = 'green', alpha = 0.5, label = 'flow samples', density = True)
plt.xlabel('value')
plt.ylabel('nr occurrences')
plt.legend()
plt.title('Histogram of data (first element)')










