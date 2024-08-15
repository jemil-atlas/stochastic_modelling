""" 
The goal of this script is to perform simulation and inference using pyro.
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
# import functorch
from functorch.dim import dims
from torch import where


# ii) Definitions

n_data = 100


pyro.set_rng_seed(1)



"""
    2. Build stochastic model -------------------------------------------------
"""


# i) Simulate some data using true model

mu_true=torch.ones([2])
sigma_true=0.1
data=pyro.distributions.Normal(mu_true,sigma_true).sample([n_data])

# functorch dimensions
batch_dim, event_dim = dims(2)
data_fc = data[batch_dim, event_dim]


# ii) Invoke observations, latent variables, parameters, model

# def model(observations = None):
#     mu=pyro.param("mu", init_tensor = torch.zeros([1,event_dim.size]))
    
#     with pyro.plate("batch_plate",n_data, dim = -2):
#         return pyro.sample("obs",dist.Normal(mu,sigma_true),obs = observations)

def model(observations = None):
    mu=pyro.param("mu", init_tensor = torch.zeros([1,event_dim.size]))
    trivial_dim = dims(sizes = [1])
        
    mu_fc = mu[trivial_dim,event_dim]
    obs_dist = dist.Normal(mu_fc.order(trivial_dim,event_dim),sigma_true).to_event(event_dim.dim())
    
    dim_pos_batch = -1 - event_dim.dim()
    
    with pyro.plate("batch_plate",n_data, dim = dim_pos_batch):
        return pyro.sample("obs",obs_dist,obs = observations)


# # Multiple dim assignment test:
# tens = torch.randn(2,3,4)
# batch = dims()
# event_1, event_2 = dims(2)
# event_dims = (event_1,event_2)

# # using iterative assignment
# tens_fc = tens
# for dim in (batch, *event_dims):
#     tens_fc = tens_fc[dim] 

# # using assignment function
# def assign_dims(tensor, dims):
#     for dim in dims:
#         tensor = tensor[dim]
#     return tensor

# tens_2_fc = assign_dims(tens, (batch, event_dims))


# # using direct tuple construction
# dim_tuple = (batch, *event_dims)
# tens_3_fc = tens[dim_tuple]


# location of specific dim
bd, ed = dims(2)
dim_tuple = (bd,ed)
aa = torch.rand([5,2])
aa_fc = aa[bd,ed]


def get_position(tensor, dim):
    checkdims = tensor.dims
    checklist = [dim is checkdim for checkdim in checkdims]
    
    index = [i for i, x in enumerate(checklist) if x]
    
    return index, checklist

# # Multi assignment
# class DimCollector():
#     def __init__(self):
#         self.dim_list = []
    
#     def dim_multi_assignment(self, dim_shapes: list, name: str) -> tuple:
#         for k in range(len(dim_shapes)):
#             dim_shape = dim_shapes[k] 
#             dim_name = name + "_{}".format(k)
#             setattr(self, dim_name, dims(sizes = [dim_shape]))
#             self.dim_list.append(getattr(self, dim_name))
                  
# dim_shapes = [10,5]
# name = 'bd'

# batch_collector = DimCollector()
# batch_collector.dim_multi_assignment(dim_shapes, name)

# Multi_assignment by list

dim_shapes = [10,5]
dim_names = ['bd1', 'bd2']
name = 'bd'

# def multi_assignment_name(dim_shapes, name):
#     name_list = []
#     for k in range(len(dim_shapes)):
#         name_list.append("{}_{}".format(name, k))
#     exec_string = "{} = dims(sizs = {})".format(name_list, dim_shapes)
#     eval(exec_string)

# # multi_assignment_name(dim_shapes, name)

# def multi_assignment(dim_shapes, dim_names):
#     for k in range(len(dim_shapes)):
#         exec_string = "{} = dims()".format(dim_names[k])
#         print(exec_string)
#         exec(exec_string)
#     eval_string = '({})'.format(', '.join(dim_names))
#     print(eval_string)
#     dim_tuple = eval(eval_string)
#     return dim_tuple

# dt = multi_assignment(dim_shapes, dim_names)


# Safe multi-assignment

import re
from functorch.dim import dims

def restricted_exec(exec_string, allowed_locals):
    # Allow only simple assignment of `dims` using a regular expression
    if not re.match(r'^\w+\s*=\s*dims\(sizes=\[\d+(,\s*\d+)*\]\)$', exec_string):
        raise ValueError("Invalid exec command")
    
    # Execute the command in a very limited scope
    allowed_globals = {"dims": dims}
    exec(exec_string, allowed_globals, allowed_locals)

def dim_assignment(dim_shapes, dim_names):
    # Validate inputs
    if not all(isinstance(shape, int) and shape > 0 for shape in dim_shapes):
        raise ValueError("All dimension shapes must be positive integers.")
    
    if not all(isinstance(name, str) and name.isidentifier() for name in dim_names):
        raise ValueError("All dimension names must be valid Python identifiers.")
    
    # Create a local environment to hold the assigned dimensions
    dims_locals = {}
    for k in range(len(dim_shapes)):
        exec_string = f"{dim_names[k]} = dims(sizes=[{dim_shapes[k]}])"
        restricted_exec(exec_string, dims_locals)
    
    # Create a tuple of dimensions
    eval_string = f"({', '.join(dim_names)})"
    dim_tuple = safe_eval(eval_string, allowed_locals=dims_locals)
    
    return dim_tuple

# Safe eval function
def safe_eval(expr, allowed_globals=None, allowed_locals=None):
    if allowed_globals is None:
        allowed_globals = {}
    if allowed_locals is None:
        allowed_locals = {}
    return eval(expr, {"__builtins__": None, **allowed_globals}, allowed_locals)

# Example usage
dim_shapes = [10, 5]
dim_names = ['bd1', 'bd2']
dt = dim_assignment(dim_shapes, dim_names)
print(dt)




"""
    3. Inference --------------------------------------------------------------
"""


# i) Create the guide

def guide(observations = None):
    pass


# ii) Run the optimization

adam = pyro.optim.Adam({"lr": 0.02})
elbo = pyro.infer.Trace_ELBO()
svi = pyro.infer.SVI(model, guide, adam, elbo)

losses = []
for step in range(1000):  
    loss = svi.step(data_fc.order(batch_dim, event_dim))
    losses.append(loss)
    if step % 100 == 0:
        print(loss)

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













dims()








