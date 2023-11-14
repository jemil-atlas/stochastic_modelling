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
from pprint import pprint


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
        

# iv) Analyze and plot

def analyze(model, trace_input):
    model_trace = pyro.poutine.trace(model).get_trace(trace_input)
    model_trace.compute_log_prob()
    pprint(model_trace.nodes)
    print(model_trace.format_shapes())
        

"""
    2. Two stochastic models
"""

# 2 different versions: i) to_event(2) to declare dependence (event shape = [10,2])
#                       ii) expand([10,2]) to declare independence (batch shape = [10,2])
#
# i) Version 1 Works as I expect it
# Define model & guide. We do not use any pyro.plate statements. Since we use 
# to_event, the distribution(2) the 2 rightmost dimensions are assumed as a single
#  event. d contains [10,2] independent copies of the Normal but considered as 
# one event -> d.batch shape is () and d.event_shape is [10,2]. 
pyro.clear_param_store()
def model(x_obs = None):
    mu = pyro.param("mean",torch.tensor(2.0))
    d = dist.Normal(mu,1).expand([10,2]).to_event(2)
    x = pyro.sample("x",d, obs = x_obs)
    assert d.batch_shape == torch.Size([])
    assert d.event_shape == torch.Size([10, 2])
    return x
guide = pyro.infer.autoguide.AutoNormal(model)
# When looking at the trace, we find everything as expected: event_shape of [10,2]
# and log_prob is a single number. The guide is empty since no latents exist.
analyze(model, x_data)
analyze(guide, ())
model()
guide()
# The optimization to fit the parameter mu terminates and seems to deliver reasonable
# results.
train(model,guide,x_data)
#
# ii) Version 2 doesnt work as I expect it
# Define model & guide. We do not use any pyro.plate statements. Since we use 
# expand([10,2]), the distribution d contains [10,2] independent copies of the 
# Normal distribution. This is also recorded for later use in the properties of 
# the distribution d -> d.batch shape is [10,2] and d.event_shape is ()
pyro.clear_param_store()
def model(x_obs = None):
    mu = pyro.param("mean",torch.tensor(2.0))
    d = dist.Normal(mu,1).expand([10,2])
    x = pyro.sample("x",d, obs = x_obs)
    assert d.batch_shape == torch.Size([10, 2])
    assert d.event_shape == torch.Size([])
    return x
guide = pyro.infer.autoguide.AutoNormal(model)
# When looking at the trace, we find everything as expected: batch_shape of [10,2]
# and log_prob is also of shape [10,2]. The guide is empty since no latents exist.
analyze(model, x_data)
analyze(guide, ())
model()
guide()
# The optimization, however does not work. It raises a ValueError and says that
# it would expect input of shape [] even though the batch_shape and the log_prob 
# shape in the model are shown as [10,2].
train(model,guide,x_data)




"""
    3. Third stochastic model
"""


# iii) Version 3 Works partly as I expect it
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
# The optimization now works but seems to exhibit slower convergence even though
# I would have assumed convergence to be faster due to conditional independence
# reducing variance of the gradient estimator.
train(model,guide,x_data)


"""
    4. fourth stochastic model, also with nontrivial event dim
"""
# Build d by plates that automatically expand and declare independent.
def model(x_obs = None):
    mu = pyro.param("mean",torch.tensor(2.0))
    d = dist.Normal(mu,1).expand([5]).to_event(1)
    with pyro.plate("plate_1", size = 2, dim = -1):
        with pyro.plate("plate_2", size = 10, dim = -2):
            x = pyro.sample("x",d, obs = x_obs)
    return x
guide = pyro.infer.autoguide.AutoNormal(model)
analyze(model, torch.zeros([10,2,5]))

# Print out the independence structure
model_trace = pyro.poutine.trace(model).get_trace(torch.zeros([10,2,5]))
trace_nodes = model_trace.nodes
print(trace_nodes['x']['cond_indep_stack'])    
# mentions the two plates, their shapes and locations
print(trace_nodes['x']['fn'])  
# mentions the full size so whats not mentioned in the cond_indep_stack is dependent
print(model_trace.format_shapes())
# gives a compressed overview of shapes but is not binding w.r.t. (in)dependence

x_data_mod = torch.tensor(np.random.normal(0,1,[10,2,5]))
train(model,guide, x_data_mod)


"""
    5. fifth stochastic model, also with nontrivial event dim
"""
# Now build d fully, then afterwards declare some dims independent
def model(x_obs = None):
    mu = pyro.param("mean",torch.tensor(2.0)).expand(10, 2, 5)
    d = dist.Normal(mu,1).to_event(1)
    with pyro.plate("plate_1", size = 2, dim = -1):
        with pyro.plate("plate_2", size = 10, dim = -2):
            x = pyro.sample("x",d, obs = x_obs)
    return x
guide = pyro.infer.autoguide.AutoNormal(model)
analyze(model, torch.zeros([10,2,5]))

# Print out the independence structure
model_trace = pyro.poutine.trace(model).get_trace(torch.zeros([10,2,5]))
trace_nodes = model_trace.nodes
print(trace_nodes['x']['cond_indep_stack'])    
# mentions the two plates, their shapes and locations
print(trace_nodes['x']['fn'])  
# mentions the full size so whats not mentioned in the cond_indep_stack is dependent
print(model_trace.format_shapes())
# gives a compressed overview of shapes but is not binding w.r.t. (in)dependence

x_data_mod = torch.tensor(np.random.normal(0,1,[10,2,5]))
train(model,guide, x_data_mod)



