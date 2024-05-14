#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The goal of this script is to illustrate 2D gaussian mixture model inference.
Multiple clusters can be found in 2D data and we formulate a model that leverages
discrete probability distributions to learn locations, covariances of the
individual clusters and use inference on the latent class assignement variables
for classification.
For this, do the following:
    1. Definitions and imports
    2. Simulate some data
    3. Model and guide
    4. Inference with pyro
    5. Plots and illustrations
"""



"""
    1. Definitions and imports
"""


# i) Imports

import torch
import pyro
import copy
import matplotlib.pyplot as plt

from pyro.infer import config_enumerate


# ii) Definitions

n_data = 100
n_dim = 2
n_clusters_true = 2
n_clusters = 2

pyro.set_rng_seed(1)



"""
    2. Simulate some data
"""


# i) Cluster centers and Covariances 

mu_clusters_true = pyro.distributions.Uniform(-5,5).sample([n_clusters_true, n_dim])
A_rand = pyro.distributions.Uniform(0,1).sample([n_clusters_true, n_dim,n_dim])
cov_clusters_true = 0.5*(torch.bmm(A_rand, A_rand.permute([0,2,1])) + 0.5*torch.eye(n_dim).repeat([n_clusters_true,1,1]))
    

# ii) Distributions and sampling

assignment_dist = pyro.distributions.Categorical(probs = (1/n_clusters_true) 
                                                 * torch.ones(n_clusters_true))
assignment_true = assignment_dist.sample([n_data])
data_locs = torch.zeros([n_data,n_dim])
data_cov = torch.zeros([n_data,n_dim, n_dim])
for k in range(n_data):
    data_locs[k,:] = mu_clusters_true[assignment_true[k],:]
    data_cov[k,:,:] = cov_clusters_true[assignment_true[k], :,:]
data_dist = pyro.distributions.MultivariateNormal(loc = data_locs, covariance_matrix = data_cov)
data = data_dist.sample()


# iii) Record classes

class_indices = []
for k in range(n_clusters_true):
    class_indices.append(torch.where(assignment_true == k)[0])



"""
    3. Model and guide
"""


# i) Model

@config_enumerate
def model(observations = None):
    # Global variables
    mu_clusters = pyro.param('mu_clusters', torch.rand([n_clusters, n_dim]))
    cov_clusters = pyro.param('cov_clusters', torch.eye(n_dim).repeat([n_clusters,1,1]), 
                              pyro.distributions.constraints.positive_definite)
    rel_probs = pyro.param('rel_probs', (1/n_clusters)*torch.ones(n_clusters),
                           pyro.distributions.constraints.simplex)
    
    # Local variables
    n_obs_or_n_data = observations.shape[0] if observations is not None else n_data
    with pyro.plate('batch_plate', size = n_obs_or_n_data, dim = -1):
        assignment_dist = pyro.distributions.Categorical(probs = rel_probs)
        assignment = pyro.sample('assignment', assignment_dist)
        obs_dist = pyro.distributions.MultivariateNormal(loc = mu_clusters[assignment,:],
                                                         covariance_matrix = cov_clusters[assignment,:,:])
        obs = pyro.sample('obs', obs_dist, obs = observations)
        
        # Diagnosis
        print("assignment.shape = {}".format(assignment.shape))
        print("assignment_dist.batch_shape = {}".format(assignment_dist.batch_shape))
        print("obs.shape = {}".format(obs.shape))
        print("obs_dist.batch_shape = {}".format(obs_dist.batch_shape))
        
        return obs, assignment

simulation_untrained, assignment_untrained = copy.copy(model())
simulation_untrained = simulation_untrained.detach()
assignment_untrained = assignment_untrained.detach()


# ii) Guide

def guide(observations = None):
    pass



"""
    4. Inference with pyro
"""


# i) Pyro inference setup

adam = pyro.optim.NAdam({"lr": 0.01})
elbo = pyro.infer.TraceEnum_ELBO(max_plate_nesting = 1)
svi = pyro.infer.SVI(model, guide, adam, elbo)

# Diagnosis - discrete variable shape enumerated, dependent dist enumerated
# data shape stays the same, gets broadcasted with enumerated shapes to logprobs
print("Sampling:")
_ = model(data)
print("Enumerated Inference:")
_ = elbo.loss(model, guide, data);


# ii) Optimization

loss_sequence = []
for step in range(1000):
    loss = svi.step(data)
    if step % 100 == 0:
        print('epoch: {} ; loss : {}'.format(step, loss))
    else:
        pass
    loss_sequence.append(loss)
    
simulation_trained, assignment_trained = copy.copy(model())
simulation_trained = simulation_trained.detach()
assignment_trained = assignment_trained.detach()
    

# iii) Infer_discrete for class inference

# Construct grid
x = torch.linspace(-7,7, 100)
y = torch.linspace(-7,7, 100)
xx, yy = torch.meshgrid(x,y, indexing = 'xy')
grid = torch.vstack((xx.flatten(), yy.flatten())).T

# Take the grid and pass it to the guide to construct the posteriors of the
# nondiscrete latents; then pass them to the model.
# This is trivial here since the guide is empty (only parameters)
guide_trace = pyro.poutine.trace(guide).get_trace(grid)  # record the globals
trained_model = pyro.poutine.replay(model, trace=guide_trace)  # replay the globals

# Define classifier by inferring discrete variables from trained model
def classifier(data, temperature=0):
    inferred_model = pyro.infer.infer_discrete(trained_model, temperature=temperature, 
                                    first_available_dim=-2)  # avoid conflict with data plate
    trace = pyro.poutine.trace(inferred_model).get_trace(data)
    return trace.nodes["assignment"]["value"]

class_predictions = classifier(grid)
class_predictions = class_predictions.reshape([100,100]).flipud()




"""
    5. Plots and illustrations
"""


# i) Illustration data

# extract class indices
class_indices_untrained = []
for k in range(n_clusters):
    class_indices_untrained.append(torch.where(assignment_untrained == k)[0])
    
class_indices_trained = []
for k in range(n_clusters):
    class_indices_trained.append(torch.where(assignment_trained == k)[0])


fig, ax = plt.subplots(4,1, figsize = (5,15), dpi = 300)

# data
for k in range(n_clusters_true):    
    ax[0].scatter(data[class_indices[k],0], data[class_indices[k],1], label = 'original data class {}'.format(k))
ax[0].legend()
ax[0].set_title('Data')

# untrained model
for k in range(n_clusters):    
    ax[1].scatter(simulation_untrained[class_indices_untrained[k],0], simulation_untrained[class_indices_untrained[k],1], label = 'untrained data class {}'.format(k))
ax[1].legend()
ax[1].set_title('Data from untrained model')

# trained model
for k in range(n_clusters):    
    ax[2].scatter(simulation_trained[class_indices_trained[k],0], simulation_trained[class_indices_trained[k],1], label = 'trained data class {}'.format(k))
ax[2].legend()
ax[2].set_title('Data from trained model')

# -> does mix the classes, probably because of config_enumerate since the model
# enumerates all posible choices for the discrete latent.

# decision boundaries
ax[3].imshow(class_predictions)
ax[3].set_title('Class predictions with posterior')
ax[3].set_xticks([])
ax[3].set_yticks([])




































