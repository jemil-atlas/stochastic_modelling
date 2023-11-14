
"""
    1. Imports and definitions
"""


# i) imports

import matplotlib.pyplot as plt
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import seaborn as sns



"""
    2. Data & stochastic model
"""


# i) Generate data

true_mean = torch.tensor(4.0)
data = dist.Normal(true_mean, 1).sample([100])


# ii) Construct stochastic model

prior_dist = dist.Normal(0., 3.)
def model(data = None):
    # Prior over the mean
    mu = pyro.sample("mu", prior_dist)
    # Likelihood
    len_data = len(data) if data is not None else 1
    with pyro.plate("data", len_data):
        obs = pyro.sample("obs", dist.Normal(mu, 1), obs=data)
    return obs


# iii) construct variational distribution

def guide(data = None):
    # Variational parameters
    mu_loc = pyro.param("mu_loc", torch.tensor(0.))
    mu_scale = pyro.param("mu_scale", torch.tensor(1.), constraint=dist.constraints.positive)
    # Variational distribution over the mean
    mu_sample = pyro.sample("mu", dist.Normal(mu_loc, mu_scale))
    return mu_sample



"""
    3. SVI inference
"""


# i) Perform inference

pyro.clear_param_store()
svi = SVI(model, guide, Adam({"lr": 0.01}), loss=Trace_ELBO())
for _ in range(1000):
    svi.step(data)
    
# ii) Print out parameters and uncertainties

for name, param in pyro.get_param_store().items():
    print('{} : {}'.format(name, param))



"""
    4. Plots and illustrations
"""


# i) Sample from the prior and the posterior

prior_samples = torch.tensor([prior_dist.sample() for _ in range(1000)])
posterior_samples = torch.tensor([guide().detach() for _ in range(1000)])


#  ii) Sample from the model, from model with posterior swapped in 

model_samples = torch.tensor([model().detach() for _ in range(1000)])

def model_with_posterior():
    # Posterior over the mean
    mu = guide()
    # Likelihood
    obs = pyro.sample("obs", dist.Normal(mu, 1))
    return obs
ppd_samples = torch.tensor([model_with_posterior().detach() for _ in range(1000)])


# iv) Plotting empirical distributions

fig, axs = plt.subplots(3, 2, figsize=(10, 15))  # 3x2 grid of Axes

# data
sns.histplot(data, ax=axs[0, 0], kde=True)
axs[0, 0].set_title('Actual data')
# model
sns.histplot(model_samples, ax=axs[1, 0], kde=True)
axs[1, 0].set_title('Model samples of data using prior distribution')
# ppd
sns.histplot(ppd_samples, ax=axs[1, 1], kde=True)
axs[1, 1].set_title('Model samples of data using posterior distribution (ppd)')
# mu prior
sns.histplot(prior_samples, ax=axs[2, 0], kde=True)
axs[2, 0].set_title('Prior samples of mu')
# mu posterior
sns.histplot(posterior_samples, ax=axs[2, 1], kde=True)
axs[2, 1].set_title('Posterior samples of mu given data')

plt.tight_layout()
plt.show()


predictive = pyro.infer.Predictive(model = model, guide = guide, num_samples = 1000)
predictive_samples = predictive(())


# v) Use predictive functionality to produce values for all sample sites based
# on the posterior density encoded in guide i.e. p(z|x) (samples of mu) and
# p(x'|x) = int p(x'|z)p(z|x) dz (samples of new obs).

fig, axs = plt.subplots(2, 2, figsize=(10, 10))  # 3x2 grid of Axes

# predictive mu
sns.histplot(predictive_samples['mu'], ax=axs[0, 0], kde=True)
axs[0, 0].set_title('Predictive samples mu')
# predictive obs
sns.histplot(predictive_samples['obs'], ax=axs[0, 1], kde=True)
axs[0, 1].set_title('Predictive samples obs')

plt.tight_layout()
plt.show()




