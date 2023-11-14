""" 
The goal of this script is to perform a simple linear regression using a generative
model in which we draw a random value and multiply it by an unknown scale to
yield an observation. The goal is to infer the scale from the observations and
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

n_out = 1
n_simu = 100

# t = np.linspace(0, n_t, n_t)



"""
    2. Generate data ---------------------------------------------------------
"""


# Original stochastic model (assumed unknown alater on)

x = torch.randint(low = 1,high = 12, size =[n_simu])
cov_mat = 0.001*torch.eye(n_out)
scale_true = 1

f_simu = torch.zeros([n_simu, n_out])
for k in range(n_simu):
    f_simu[k,:] = torch.tensor(np.random.multivariate_normal(np.array(scale_true * x[k].detach().numpy()).reshape(1), cov_mat))

xf_simu = torch.tensor(np.hstack((x.detach().numpy().reshape([n_simu,1]),f_simu)))


"""
    2. Build stochastic model
"""


# i) Initializations

pyro.clear_param_store()



#  ii) SVI Model for simulations and inference 

# # by iteration through the plate
# # doesnt even work correctly; only the first obersvation k = 0 is used
# def stoch_model(xf_simu = None, dim = -1):                
#     scale = pyro.param('scale', 3*torch.ones([1]))
#     sigma = 0.001*torch.eye(n_out)
    
#     for k in pyro.plate('x_simu',n_simu):
#         x = pyro.sample("obs_x_{}".format(k), dist.Uniform(low = 1, high = 12), obs = xf_simu[k,0])
#         f = pyro.sample('obs_f_{}'.format(k), dist.Normal(loc = scale*x, scale = sigma),obs = xf_simu[k,1].reshape([1,n_out]))
#         return f


# by pyro.plate on the ()() distribution creating n_simu independent draws
def stoch_model(xf_simu = None):                
    scale = pyro.param('scale', 3*torch.ones([1]))
    sigma = 0.001*torch.eye(n_out)
    
    with pyro.plate('x_simu',n_simu, dim = -1):
        x = pyro.sample("obs_x", dist.Uniform(low = 1, high = 12), obs = xf_simu[:,0])
        f = pyro.sample('obs_f', dist.Normal(loc = scale*x, scale = sigma),obs = xf_simu[:,1])
        return f


# by pyro.plate on a ()(n_simu) distribution creating a distribution object with 
# batch_dim n_simu and declaring the rightmost dimension to be independent
# def stoch_model(xf_simu = None):                
#     scale = pyro.param('scale', torch.ones([1]))
#     sigma = 0.001*torch.eye(n_out)
    
#     with pyro.plate('x_simu',n_simu):
#         x = pyro.sample("obs_x", dist.Uniform(low = torch.ones([n_simu]), high = 12 * torch.ones([n_simu])), obs = xf_simu[:,0])
#         f = pyro.sample('obs_f', dist.Normal(loc = scale*x, scale = sigma),obs = xf_simu[:,1])
#         return f


# by iteration through the plate
# def stoch_model(xf_simu = None):                
#     scale = pyro.param('scale', torch.ones([1]))
#     sigma = 0.001*torch.eye(n_out)
    
#     for k in pyro.plate('x_simu',n_simu):
#         x = pyro.sample("obs_x_{}".format(k), dist.Uniform(low = 1, high = 12), obs = xf_simu[k,0])
#         f = pyro.sample('obs_f_{}'.format(k), dist.MultivariateNormal(loc = scale*x, covariance_matrix = sigma),obs = xf_simu[k,1].reshape([1,1]))
#         return f


# by pyro.plate on the multivariate : reshape such that:
#   i) x is torch.Size([100]) : expand low and high, obs is torch.Size([100])
#   ii) dist(x) has event_shape = torch.Size([]) , batch_shape = torch.Size([100])
#   iii) f is torch.size([100,1]) : achieved by expanding loc to [n_simu,1], cov mat = [1,1], obs is [n_simu,1]
#   iv) dist(f) has event_shape =torch.size([1]) , batch_shape = torch.size([100])
# def stoch_model(xf_simu = None):                
#     scale = pyro.param('scale', torch.ones([1]))
#     sigma = 0.001*torch.eye(n_out)
    
#     with pyro.plate('x_simu',n_simu, dim = -1):
#         x = pyro.sample("obs_x", dist.Uniform(low = torch.ones([n_simu]), high = 12*torch.ones([n_simu])), obs = xf_simu[:,0])
#         f = pyro.sample('obs_f', dist.MultivariateNormal(loc = scale*x.reshape([n_simu,1]), covariance_matrix = sigma),obs = xf_simu[:,1].reshape([n_simu,1]))
#         return f


stoch_guide = pyro.infer.autoguide.AutoNormal(stoch_model)

# trace = poutine.trace(stoch_model).get_trace()
# trace.compute_log_prob()  # optional, but allows printing of log_prob shapes
# print(trace.format_shapes())


# ii) Do svi inference

lr=0.01
n_steps=1000

adam_params = {"lr": lr}
adam = pyro.optim.Adam(adam_params)
svi = SVI(stoch_model, stoch_guide, adam, loss=Trace_ELBO())

for step in range(n_steps):
    loss = svi.step(xf_simu)
    if step % 50 == 0:
        print('[iter {}]  loss: {:.4f}'.format(step, loss))

mle_estimate_scale = pyro.param("scale").detach().numpy()
print(' The ml estimate of the scale is {}'.format(mle_estimate_scale))



# iii) Plots and illustrations

plt.figure(1, dpi = 300)
plt.plot(f_simu.T)
plt.figure(2, dpi = 300)
plt.plot(mle_estimate_scale*np.linspace(0,12,100))


# ppd=pyro.infer.Predictive(stoch_model,guide=stoch_guide, num_samples=1)
# svi_samples=ppd()
# svi_obs=svi_samples["obs"]



















