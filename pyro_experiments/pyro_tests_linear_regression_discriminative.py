""" 
The goal of this script is to perform a simple linear regression using a discriminative
model in which we want to associate to an x value a y value by means of scaling
with an unknown factor. The goal is to infer the scale from the observations and
the model.
For this, do the following:
    1. Definitions and imports
    2. Build stochastic model
    3. Inference
    4. Plots and illustrations
"""


"""
    1. Definitions and imports
"""


# i) Imports

import numpy as np
import seaborn as sns
import pyro
import pyro.poutine as poutine
import pyro.distributions as dist
from pyro.infer import NUTS, MCMC
from pyro.infer import SVI, Trace_ELBO
import torch

import matplotlib.pyplot as plt


# ii) Definitions

n_out = 5
n_simu = 10

x = torch.linspace(0, 1, n_simu).reshape([n_simu,1])



"""
    2. Generate data ---------------------------------------------------------
"""


# Original stochastic model (assumed unknown later on)

cov_mat = 0.01*torch.eye(n_out)
scale_true = 1

f_simu = torch.zeros([n_simu, n_out])
for k in range(n_simu):
    mu_vec = (np.array(scale_true * x[k].detach().numpy()).reshape(1))*np.ones([n_out])
    f_simu[k,:] = torch.tensor(np.random.multivariate_normal(mu_vec, cov_mat))

xf_simu = torch.tensor(np.hstack((x.detach().numpy().reshape([n_simu,1]),f_simu)))


"""
    2. Build stochastic model
"""


# i) Initializations

pyro.clear_param_store()



#  ii) SVI Model for simulations and inference 

# # by iteration through the plate
# def stoch_model(x, f_simu = None):                
#     scaling = pyro.param('scaling', 3*torch.ones(1))
#     sigma = 0.01*torch.eye(n_out)
#     f = torch.zeros([x.shape[0],n_out])

#     for k in pyro.plate('x_simu', x.shape[0]):
#         f_simu_k = f_simu[k,:] if f_simu is not None else None
#         # dist of f with batch_shape = 10, event_shape = 5
#         f_dist = dist.MultivariateNormal(loc = scaling*x[k]*torch.ones([n_out]), covariance_matrix = sigma)
#         f[k,:] = pyro.sample('obs_f_{}'.format(k), f_dist, obs = f_simu_k)
#     return f


# # by pyro.plate on the n_out-multivariate distribution creating n_simu independent draws
# #   i) f is torch.size([10,5]) : achieved by expanding loc to [n_simu,n_out], 
# #   cov mat = [n_out,n_out], obs is [n_simu,n_out]
# #   ii) f_dist has event_shape = torch.size([5]) , batch_shape = torch.size([10])
# def stoch_model(x, f_simu = None):                
#     scaling = pyro.param('scaling', 3*torch.ones([1]))
#     sigma = 0.01*torch.eye(n_out)
    
#     # since the batch shape is going to be [10], and the event_shape = [5] 
#     # we declare the dim -2 independent (second from right)
#     with pyro.plate('x_simu',n_simu, dim = -1):
#         extension_tensor = torch.ones([n_simu,n_out])
#         # Notice the broadcasting rules: extension_tensor is [n_simu,n_out]
#         # and is broadcast over [n_simu, 1] 's trivial dim to create mu of shape [n_simu,n_out]
#         f_dist = dist.MultivariateNormal(loc = scaling*x.reshape([n_simu,1])*extension_tensor, covariance_matrix = sigma)
#         f = pyro.sample("obs_f", f_dist, obs = f_simu)
#         return f


# finally the conceptually cleanest solution consisting in first invoking the
# full distribution and then declaring dimensions to be dependent or independent.
#   i) invoking the distribution f_dist with batch_shape = 100, event_shape = [5]
#   ii) pyro.plate to declare independence on the batch_shape

def stoch_model(x, f_simu = None):                
    scaling = pyro.param('scaling', 3*torch.ones([1]))
    sigma = 0.01*torch.eye(n_out)
    
    extension_tensor = torch.ones([n_simu,n_out])
    f_dist= dist.MultivariateNormal(loc = scaling*x.reshape([n_simu,1])*extension_tensor, covariance_matrix = sigma)
    f_simu_or_None = f_simu if f_simu is not None else None
    with pyro.plate('x_plate', x.shape[0], dim = -1):
        f = pyro.sample("f_obs", f_dist, obs = f_simu_or_None)
        return f


# Test the model output and trace the model internals
x_test = torch.linspace(0,1,10)
stoch_model(x_test)

stoch_guide = pyro.infer.autoguide.AutoNormal(stoch_model)
trace = poutine.trace(stoch_model).get_trace(x_test)
trace.compute_log_prob()  # optional, but allows printing of log_prob shapes
print(trace.format_shapes())


# ii) Do svi inference

pyro.clear_param_store()
lr=0.01
n_steps=1000

adam_params = {"lr": lr}
adam = pyro.optim.Adam(adam_params)
svi = SVI(stoch_model, stoch_guide, adam, loss=Trace_ELBO())

for step in range(n_steps):
    loss = svi.step(x, f_simu)
    if step % 50 == 0:
        print('[iter {}]  loss: {:.4f}'.format(step, loss))

mle_estimate_scale = pyro.param("scaling").detach().numpy()
print(' The ml estimate of the scale is {}'.format(mle_estimate_scale))


# iii) Plots and illustrations

plt.figure(1, dpi = 300)
plt.plot(x, f_simu)
plt.figure(2, dpi = 300)
plt.plot(mle_estimate_scale*np.linspace(0,1,100))


# ppd=pyro.infer.Predictive(stoch_model,guide=stoch_guide, num_samples=1)
# svi_samples=ppd()
# svi_obs=svi_samples["obs"]



















