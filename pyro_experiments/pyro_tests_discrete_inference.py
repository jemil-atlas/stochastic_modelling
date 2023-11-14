#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The goal of this script is to demonstrate pyros functionality for inference in 
the presence of discrete random variables. This is done by fitting a gaussian
mixture model to some data, we compare the result to what scikit-learn produces.
For this, do the following:
    1. Imports and definitions
    2. Simulate data
    3. Stochastic model 
    4. SVI inference
    5. Compare with scikit learn
    6. Plots and illustrations
"""


"""
    1. Imports and definitions
"""


# i) Imports

import numpy as np
import torch
import matplotlib.pyplot as plt
import pyro
import sklearn as sk
import copy
from pprint import pprint


# ii) Definitions

pyro.clear_param_store()
torch.set_default_dtype(torch.float64)
n_data = 100
dim_data = 2
n_clusters = 2
n_mixture = 2
data_assignment = pyro.sample('data_assignment', pyro.distributions.Categorical(probs = (1/n_clusters)* torch.ones(size = [n_data,n_clusters])))



"""
    2. Simulate data
"""


# i) Random means

sigma = 5
mu_tensor = torch.normal(mean = torch.zeros([dim_data,n_clusters]),std = sigma*torch.ones([dim_data,n_clusters]))


# ii) Random covariances

# random_tensor = torch.normal(mean = torch.zeros([n_clusters, dim_data, dim_data]),std = sigma*torch.ones([n_clusters, dim_data, dim_data]))
# cov_batch = torch.bmm(random_tensor, torch.permute(random_tensor, [0,2,1]))
cov_batch = torch.repeat_interleave(torch.eye(dim_data).unsqueeze(0), repeats = n_clusters, dim = 0 )


# iii) Draw from multivariate Gaussians

data = torch.zeros([n_data,dim_data])
for k in range(n_data):
    data[k,:] = torch.distributions.MultivariateNormal(loc = mu_tensor[:,data_assignment[k]], covariance_matrix = cov_batch[data_assignment[k],:,:]).sample()



"""
    3. Stochastic model 
"""


# i) GaussianMixture object

class GaussianMixture(pyro.nn.PyroModule):
    def __init__(self, n_mixture, n_data, dim_data):
        # Initialize using invocation of base classes init method
        super().__init__()
        self.n_mixture = n_mixture
        self.n_data = n_data
        self.dim_data = dim_data
        # self.guide = pyro.infer.autoguide.AutoDiagonalNormal(self.model)
        # self.guide = pyro.infer.autoguide.AutoDiagonalNormal(pyro.poutine.block(self.model, hide=['assignment']))
        
    
    
    # Define the forward model
    @pyro.infer.config_enumerate
    def model(self, observations = None):
        # Instead of first defining the full probability distribution and afterwards
        # declaring the batch dimension to be independent, we directly construct and
        # sample in the independence plates to ensure that enumeration stays feasible.
        # Enumerating 2 categorical outcomes n_obs times is cheapers than a full
        # enumeration of all possible discrete combinations (2**n_obs possibilities).
        
        # This requires a structure where first general distributional params are
        # defined and the plates are invoked subsequently as early as possible
        # before assignments are made.
        self.n_observations = observations.shape[0] if observations is not None else self.n_data
        init_cluster_probs = (1/self.n_mixture)*torch.ones([1, n_mixture])
        cluster_probs = pyro.param('cluster_probs', init_tensor = init_cluster_probs, constraint = pyro.distributions.constraints.simplex)
        
        # Multivariate normal distribution
        mean_vecs = pyro.param('mean_vecs', init_tensor = torch.randn([self.n_mixture, self.dim_data]))
        cov_mats = pyro.param('cov_mats', init_tensor = torch.eye(dim_data).unsqueeze(0).repeat([n_mixture,1,1]))
        
        # Independent assignment to clusters and sampling
        with pyro.plate('batch_plate', size = self.n_observations, dim = -1):      
            assignment = pyro.sample('assignment', pyro.distributions.Categorical(probs = cluster_probs))
            mean_vec = mean_vecs[assignment,:]
            cov_mat = cov_mats[assignment,:,:]
            obs_dist = pyro.distributions.MultivariateNormal(loc = mean_vec, covariance_matrix = cov_mat)
            obs = pyro.sample('observations', obs_dist, obs = observations)
            return obs
        
    # Define the guide  
    # The guide can be empty. There is an unobserved latent variable (assignment)
    # but it is discrete and therefore needs to be blocked anyway in the guide.
    # In a more complex setting, we would construct a guide like this:
    #   self.guide = pyro.infer.autoguide.AutoDiagonalNormal(pyro.poutine.block(self.model, hide=['assignment']))
    def guide(self, observations = None):
        pass
        
    
# ii) Invoke a GaussianMixture object & analyze

gaussian_mixture = GaussianMixture(n_mixture, n_data, dim_data)
samples_pretrain = copy.copy(gaussian_mixture.model().detach().numpy())

model_trace = pyro.poutine.trace(gaussian_mixture.model).get_trace()
# pprint(model_trace.nodes)
print(model_trace.format_shapes())



"""
    4. SVI inference
"""


# i) Set up training

# specifying scalar options
learning_rate = 1*1e-1
num_epochs = 500
adam_args = {"lr" : learning_rate}

# Setting up svi
optimizer = pyro.optim.AdamW(adam_args)
elbo_loss = pyro.infer.TraceEnum_ELBO(max_plate_nesting=1)
svi = pyro.infer.SVI(model = gaussian_mixture.model, guide = gaussian_mixture.guide, optim = optimizer, loss= elbo_loss)


# ii) Execute training

train_elbo = []
for epoch in range(num_epochs):
    epoch_loss = svi.step(data)
    train_elbo.append(-epoch_loss)
    if epoch % 10 == 0:
        # log the data on tensorboard
        print("Epoch : {} train loss : {}".format(epoch, epoch_loss))


# iii) Results post training

samples_posttrain = copy.copy(gaussian_mixture.model().detach().numpy())
for name, value in pyro.get_param_store().items():
    print('{} : {}'.format(name, value))




"""
    5. Compare with scikit learn
"""




"""
    6. Plots and illustrations
"""


# i) Plot data

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# First scatter plot
axs[0].scatter(data[:,0], data[:,1])
axs[0].set_title('Scatterplot of data')

# Second scatter plot
axs[1].scatter(samples_pretrain[:,0], samples_pretrain[:,1])
axs[1].set_title('Scatterplot pretraining')

# # Third scatter plot
axs[2].scatter(samples_posttrain[:,0], samples_posttrain[:,1])
axs[2].set_title('Scatterplot posttraining')









