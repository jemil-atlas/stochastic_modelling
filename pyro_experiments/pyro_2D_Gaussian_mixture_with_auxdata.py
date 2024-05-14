#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The goal of this script is to illustrate 2D gaussian mixture model inference.To
the samples of the mixture model, we add values dependent on some auxiliary data.
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
time = torch.linspace(0,1,n_data)

pyro.set_rng_seed(10)



"""
    2. Simulate some data
"""


# i) Cluster centers and Covariances 

mu_clusters_true = pyro.distributions.Uniform(-5,5).sample([n_clusters_true, n_dim])
A_rand = pyro.distributions.Uniform(0,1).sample([n_clusters_true, n_dim,n_dim])
cov_clusters_true = 0.5*(torch.bmm(A_rand, A_rand.permute([0,2,1])) + 0.5*torch.eye(n_dim).repeat([n_clusters_true,1,1]))

alpha_true = 5*torch.ones([n_dim,1])
aux_effect = (alpha_true*time.unsqueeze(0).repeat([n_dim,1])).T


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
data = data_dist.sample() + aux_effect


# iii) Record classes

class_indices = []
for k in range(n_clusters_true):
    class_indices.append(torch.where(assignment_true == k)[0])



"""
    3. Model and guide
"""


# i) Aux effect function

def aux_effect_fun(time, alpha):
    effect = (alpha*time.unsqueeze(0).repeat([n_dim,1])).T
    return effect
    

# ii) Model

@config_enumerate
def model(time, observations = None):
    # Global variables
    mu_clusters = pyro.param('mu_clusters', torch.rand([n_clusters, n_dim]))
    cov_clusters = pyro.param('cov_clusters', torch.eye(n_dim).repeat([n_clusters,1,1]), 
                              pyro.distributions.constraints.positive_definite)
    rel_probs = pyro.param('rel_probs', (1/n_clusters)*torch.ones(n_clusters),
                           pyro.distributions.constraints.simplex)
    alpha = pyro.param('alpha', torch.rand([n_dim,1]))
    
    aux_effect = aux_effect_fun(time, alpha)
    
    # Local variables
    n_obs_or_n_data = observations.shape[0] if observations is not None else n_data
    with pyro.plate('batch_plate', size = n_obs_or_n_data, dim = -1) as ind:
        assignment_dist = pyro.distributions.Categorical(probs = rel_probs)
        assignment = pyro.sample('assignment', assignment_dist)
        obs_dist = pyro.distributions.MultivariateNormal(loc = mu_clusters[assignment,:] + aux_effect[ind,:],
                                                         covariance_matrix = cov_clusters[assignment,:,:])
        obs = pyro.sample('obs', obs_dist, obs = observations)
        
        # # Diagnosis
        # print("assignment.shape = {}".format(assignment.shape))
        # print("assignment_dist.batch_shape = {}".format(assignment_dist.batch_shape))
        # print("obs.shape = {}".format(obs.shape))
        # print("obs_dist.batch_shape = {}".format(obs_dist.batch_shape))
        # print("obs_dist.shape = {}".format(obs_dist.shape()))
        
        return obs, assignment

simulation_untrained, assignment_untrained = copy.copy(model(time))
simulation_untrained = simulation_untrained.detach()
assignment_untrained = assignment_untrained.detach()


# iii) Guide

def guide(time, observations = None):
    pass



"""
    4. Inference with pyro
"""


# i) Pyro inference

adam = pyro.optim.NAdam({"lr": 0.1})
elbo = pyro.infer.TraceEnum_ELBO(max_plate_nesting = 1)
svi = pyro.infer.SVI(model, guide, adam, elbo)

# Diagnosis
print("Sampling:")
_ = model(time, data)
print("Enumerated Inference:")
_ = elbo.loss(model, guide, *(time,data))

loss_sequence = []
for step in range(1000):
    loss = svi.step(*(time,data))
    if step % 100 == 0:
        print('epoch: {} ; loss : {}'.format(step, loss))
    else:
        pass
    loss_sequence.append(loss)
    
simulation_trained, assignment_trained = copy.copy(model(time))
simulation_trained = simulation_trained.detach()
assignment_trained = assignment_trained.detach()
    


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

# # decision boundaries
# ax[3].imshow(class_predictions)
# ax[3].set_title('Class predictions with posterior')
# ax[3].set_xticks([])
# ax[3].set_yticks([])




































