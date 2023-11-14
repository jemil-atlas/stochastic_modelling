#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The goal of this script is to investigate the impact of pyro plates when used on
different distribution shapes.
For this, do the following:
    1. Imports and definitions
    2. Model & inference mean for Gaussian. No independence statement
    3. Model & inference mean for Gaussian. IID with plate.
    4. Model & inference mean for Gaussian. Shaping batch_shape via arguments
    5. Model & inference mean for Gaussian. Shaping batch_shape via expand
"""

"""
    1. Imports and definitions
"""


# i) Imports

import numpy as np
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from pyro.infer.autoguide import AutoMultivariateNormal
from pyro.util import check_traces_match
from pprint import pprint

# Enable validation checks
pyro.enable_validation(True)

# ii) Data generation

mu_true = 0
sigma_true = 1
batch_shape = [10,2]
x_data = torch.tensor(np.random.normal(mu_true,sigma_true, batch_shape)) # all of them independent
test_input = torch.tensor(1)


# iii) Training function

def train(model, guide, x_data):
    pyro.clear_param_store()
    svi = SVI(model, guide, Adam({"lr": 0.005}), loss=Trace_ELBO())
    num_steps = 1000
    for step in range(num_steps):
        loss = svi.step(x_data)
        if step % 100 == 0:
            print("Step: ", step, "Loss: ", loss)
    print("Optimization terminated. Results follow")
    for name, value in pyro.get_param_store().items():
        print(name, pyro.param(name).data.cpu().numpy())
        

# iv) Analyzse and plot

def analyze(model, trace_input):
    model_trace = pyro.poutine.trace(model).get_trace(trace_input)
    # model_trace.compute_log_prob()
    pprint(model_trace.nodes)
    print(model_trace.format_shapes())
        

"""
    2. Model & inference mean for Gaussian. No independence statement
"""

# 2 different versions: i) no distribution shaping
#                       ii) to_event(2) to declare dependence
#
# # Define model & guide. We do not use any pyro.plate statements and try to figure
# # out how to write a statement using only the inbuilt sample statement without
# # any reshaping using plates, to_events, or expands.
# pyro.clear_param_store()
# def model(x_obs = None):
#     mu = pyro.param("mean",torch.tensor(2.0))
#     sigma = pyro.param("sigma",torch.tensor(2.0))
#     d = dist.Normal(mu, sigma)
#     x = pyro.sample("x",d, obs = x_obs)
#     # print( "batch_shape : ",  d.batch_shape, "event_shape : ", d.event_shape)
#     return x
# guide = pyro.infer.autoguide.AutoNormal(model)

# analyze(model, ())
# analyze(guide, ())
# model()
# guide()

# # The svi training does not work because the model that is passed to the svi-training
# # informs the inference, that only a single scalar is to be expected as output.
# # As I found out during discussions in the forum, every dimension must be declared
# # either as dependent or independent using pyro.plate or to_event. Otherwise
# # the svi checks fail. The relevant entry are the 'cond_indep_stack' entries in the 
# # trace nodes. They are left empty, if only expand or shaped parameters are used.
# train(model,guide,x_data)



# Define model & guide. We do not use any pyro.plate statements. Since we use 
# to_event, the distribution(2) the 2 rightmost dimensions are assumed as a single
#  event. d contains [10,2] independent copies of the Normal but considered as 
# one event -> d.batch shape is () and d.event_shape is [10,2]
pyro.clear_param_store()
def model(x_obs = None):
    mu = pyro.param("mean",torch.tensor(2.0))
    d = dist.Normal(mu,1).expand([10,2]).to_event(2)
    x = pyro.sample("x",d, obs = x_obs)
    # print( "batch_shape : ",  d.batch_shape, "event_shape : ", d.event_shape)
    return x
guide = pyro.infer.autoguide.AutoNormal(model)

analyze(model, ())
analyze(guide, ())
model()
guide()

train(model,guide,x_data)






"""
    3. Model & inference mean for Gaussian. IID with plate.
"""


# Define model & guide and use pyro.plate statements to declare independence.
# Since we use two plates the distribution d has the following properties
#  -> d.batch shape is [10,2] and d.event_shape is ()
pyro.clear_param_store()
def model(x_obs = None):
    mu = pyro.param("mean",torch.tensor(2.0))
    d = dist.Normal(mu,1)
    with pyro.plate("plate_1", size = 2, dim = -1):
        with pyro.plate("plate_2", size = 10, dim = -2):
            x = pyro.sample("x",d, obs = x_obs)
    return x
guide = pyro.infer.autoguide.AutoNormal(model)
# When looking at the trace, we find everything as expected: batch_shape of [10,2]
# and log_prob is also of shape [10,2]. 
analyze(model, x_data)
analyze(guide, ())
model()
guide()
# The optimization now works, everything is declared as independent by the plate statements
train(model,guide,x_data)



"""
    4. Model & inference mean for Gaussian. Shaping batch_shape via arguments
"""

# Originally envisioned
# 2 different versions: i) expand([10,2]) and plate to declare independence
#                       ii) extend distribution parameters and plate to declare independence
#
# Note that it is not possible to simply not use plate and let the expand() statement
# take over the work of declaring independence. The batch_shape is correct but during
# inference some complaints happen as the wrong log_prob shapes are expected.
# This is due to independence only being registered with the pyro.plate statement
# not with the expand statement or by passing shaped parameters.

# Define model & guide. Since we expand the distribution to [10,2] by multiplying
# with a ones tensor of that shape, the distribution d contains [10,2] independent
# copies of the Normal distribution. We still have to register them as independent though
# by using pyro.plate statements, that confer this information into the 'cond_indep_stack'
# entry and make it available for svi.

pyro.clear_param_store()
def model(x_obs = None):
    mu = pyro.param("mean",torch.tensor(2.0))
    extension_tensor = torch.ones([10,2])
    d = dist.Normal(mu*extension_tensor,1)
    with pyro.plate("plate_1", size = 2, dim = -1):
        with pyro.plate("plate_2", size = 10, dim = -2):
            x = pyro.sample("x",d, obs = x_obs)
    # print( "batch_shape : ",  d.batch_shape, "event_shape : ", d.event_shape)
    return x
guide = pyro.infer.autoguide.AutoNormal(model)
  
analyze(model, ())
analyze(guide, ())
model()
guide()

train(model,guide,x_data)



"""
    5. Model & inference mean for Gaussian. Shaping batch_shape via expand
"""

# Define model & guide. Since we expand the distribution to [10,2] by using the
# expand([10,2]) command, the distribution d contains [10,2] independent
# copies of the Normal distribution. We still have to register them as independent though
# by using pyro.plate statements, that confer this information into the 'cond_indep_stack'
# entry and make it available for svi.
pyro.clear_param_store()
def model(x_obs = None):
    mu = pyro.param("mean",torch.tensor(2.0))
    d = dist.Normal(mu,1).expand([10,2])
    with pyro.plate("plate_1", size = 2, dim = -1):
        with pyro.plate("plate_2", size = 10, dim = -2):
            x = pyro.sample("x",d, obs = x_obs)
    # print( "batch_shape : ",  d.batch_shape, "event_shape : ", d.event_shape)
    return x
guide = pyro.infer.autoguide.AutoNormal(model)
  
analyze(model, ())
analyze(guide, ())
model()
guide()

train(model,guide,x_data)

