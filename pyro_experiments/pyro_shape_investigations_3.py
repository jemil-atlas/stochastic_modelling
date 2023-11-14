#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The goal of this script is to showcase inference in case of nontrivial batch_shape
and event_shape dims.
The inference task we are performing features some potentially observed data x
and some latent data z. Both are multivariate
"""

"""
    1. Imports and definitions
"""

import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from pyro.infer.autoguide import AutoMultivariateNormal
from pyro.util import check_traces_match

# Enable validation checks
pyro.enable_validation(True)


"""
    2. Generate some data and the model
"""

# Step 1: Generate x and y
# On this level things like to_event and expand do not have any meaning for the
# inference as only the data values are passed and no semantic information.
# x, and y both have dimension [10,2] where event_shape is 2 and batch-shape is 10.
z_dist = dist.Normal(0, 1).expand([10, 2])
z = z_dist.sample()
weights = torch.tensor([1,2])
x = torch.vstack((weights[0]*z[:, 0], weights[1]*z[:, 1])).T

# Step 2: Create the model
# We want to perform inference on this model and therefore need a proper shaping
# of batch_shapes, event_shapes and independence assertions.
def model(z, x=None):
    # z is assumed to be a tensor of shape [10, 2]

    # "weights_plate" is a plate of size 2. This declares that the dimension -1 (the last dimension)
    # of "weights" corresponds to 2 independent random variables.
    # "weights" is expected to have shape [2].
    with pyro.plate("weights_plate", size=2, dim=-1):
        weights = pyro.sample("weights", dist.Normal(0, 10)).expand([2])

    # "x_plate" is a plate of size 10. This declares that the dimension -1 (the last dimension)
    # of "mean" and "x_obs" corresponds to 10 independent random variables.
    # "mean" and "x_obs" are expected to have shape [10].
    with pyro.plate("x_plate", size=10, dim=-2):
        # The index operation z[k, :] leverages PyTorch's advanced indexing and results in a tensor of shape [10, 2]
        # The assignment operations broadcast the tensors to shape [10, 2].
        mean = z
        # mean = torch.tensor([weights[0]*z[:, 0], weights[1]*z[:, 1]])  # shape: [10,2]
        # "x_obs" is expected to have shape [10,2].
        observation = pyro.sample("x_obs", dist.Normal(mean, 0.01), obs=x)
        return observation


    
pyro.render_model(model, model_args=(), render_params=True)
model_trace = pyro.poutine.trace(model).get_trace(z)
model_trace.compute_log_prob()
model_trace.nodes
print(model_trace.format_shapes())

"""
    4. perform inference
"""

# Step 3: Create the guide
guide = AutoMultivariateNormal(model)

# Step 4: Perform SVI
svi = SVI(model, guide, Adam({"lr": 0.005}), loss=Trace_ELBO())
num_steps = 5000
for step in range(num_steps):
    svi.step(z, x)
    if step % 500 == 0:
        print("Step: ", step)

# Inspect the learned parameters
for name, value in pyro.get_param_store().items():
    print(name, value)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    