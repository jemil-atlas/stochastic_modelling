""" 
The goal of this script is to perform simulation and inference using pyro.
A simple bernoulli trial model is implemented together with automated inference
using monte carlo and svi methods. The goal is to figure out the distribution
of a coin flip with a bent coin
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
import pyro.distributions as dist
from pyro.infer import NUTS, MCMC
from pyro.infer import SVI, Trace_ELBO
import torch


# ii) Definitions

n_sample=100
y_obs=dist.Bernoulli(0.5*torch.ones([n_sample])).sample()



"""
    2. Build stochastic model
"""


# i) Initializations

pyro.clear_param_store()



#  ii) SVI Model for simulations and inference 

def stoch_model_svi(y_obs=None):                
    theta=pyro.param('theta',0.5*torch.ones([1]),constraint=pyro.distributions.constraints.unit_interval)
    
    with pyro.plate('data',n_sample):
        sample=pyro.sample('obs', dist.Bernoulli(theta),obs=y_obs)
        return sample

def stoch_guide(y_obs=None):
    pass


# iii) MCMC

def stoch_model_mcmc(y_obs=None):
    theta=pyro.sample('theta', dist.Normal(torch.ones([1]), torch.ones([1])))
    y = pyro.sample('y', dist.Bernoulli(theta*torch.ones([n_sample])), obs=y_obs)
    return y



"""
    3. Inference
"""


# i) Do HMC inference

n_simu_hmc=500

nuts_kernel=NUTS(model=stoch_model_mcmc)         # Build hmc kernel
mcmc_results=MCMC(nuts_kernel,num_samples=n_simu_hmc,warmup_steps=200) # Build hmc setup
mcmc_results.run(y_obs)         # Run hmc and populate samples

samples = mcmc_results.get_samples()
point_estimate=torch.mean(samples['theta'])


# ii) Do svi inference

lr=0.01
n_steps=1000

adam_params = {"lr": lr}
adam = pyro.optim.Adam(adam_params)
svi = SVI(stoch_model_svi, stoch_guide, adam, loss=Trace_ELBO())

for step in range(n_steps):
    loss = svi.step(y_obs)
    if step % 50 == 0:
        print('[iter {}]  loss: {:.4f}'.format(step, loss))

mle_estimate = pyro.param("theta").item()


# iii) Evaluate ppd

ppd=pyro.infer.Predictive(stoch_model_svi,guide=stoch_guide, num_samples=1)
svi_samples=ppd()
svi_obs=svi_samples["obs"]




"""
    4. Plots and illustrations
"""


# data=y_obs
# def train(model, guide, lr=0.005, n_steps=1001):
#     pyro.clear_param_store()
#     adam_params = {"lr": lr}
#     adam = pyro.optim.Adam(adam_params)
#     svi = SVI(model, guide, adam, loss=Trace_ELBO())

#     for step in range(n_steps):
#         loss = svi.step(data)
#         if step % 50 == 0:
#             print('[iter {}]  loss: {:.4f}'.format(step, loss))
            
            
            
# def model_mle(data):
#     # note that we need to include the interval constraint;
#     # in original_model() this constraint appears implicitly in
#     # the support of the Beta distribution.
#     f = pyro.param("latent_fairness", torch.tensor(0.5),
#                    constraint=dist.constraints.unit_interval)
#     with pyro.plate("data", data.size(0)):
#         pyro.sample("obs", dist.Bernoulli(f), obs=data)
        
        
# def guide_mle(data):
#     pass

# train(model_mle, guide_mle)

# mle_estimate = pyro.param("latent_fairness").item()
# print("Our MLE estimate of the latent fairness is {:.3f}".format(mle_estimate))




















