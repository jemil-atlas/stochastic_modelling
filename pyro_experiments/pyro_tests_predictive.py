# Infer and plot posterior distribution of mu

import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.optim import Adam

def model(data):
    # Prior on the mean parameter
    mu = pyro.sample("mu", dist.Normal(0., 10.))
    # Likelihood of the data
    with pyro.plate("data", len(data)):
        pyro.sample("obs", dist.Normal(mu, 1.), obs=data)

def guide(data):
    # Variational parameters
    mu_loc = pyro.param("mu_loc", torch.tensor(0.))
    mu_scale = pyro.param("mu_scale", torch.tensor(1.), constraint=dist.constraints.positive)
    # Variational distribution on mu
    pyro.sample("mu", dist.Normal(mu_loc, mu_scale))

# Generate some data
torch.manual_seed(1)
data = torch.randn(100) + 3.  # centered around 3

# Set up the optimizer and the inference algorithm
pyro.clear_param_store()
optimizer = Adam({"lr": 0.01})
svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

# Run stochastic variational inference
num_steps = 2000
for step in range(num_steps):
    svi.step(data)

for param_name, param_value in pyro.get_param_store().items():
    print('{} : {}'.format(param_name, param_value))

# Generate samples from the posterior
predictive = Predictive(model, guide=guide, num_samples=1000)
samples = predictive(torch.rand(1))  # the argument is not used in the model or guide, but required by the syntax

# Analyze the posterior samples
mu_samples = samples["mu"]
print("Posterior mean:", mu_samples.mean())
print("Posterior std. deviation:", mu_samples.std())

# Visualize the posterior samples
import matplotlib.pyplot as plt
plt.hist(mu_samples.detach().numpy(), bins=30)
plt.title("Posterior distribution of mu")
plt.xlabel("mu")
plt.ylabel("Frequency")
plt.show()

# # Analyzing the ppd would be done as shown below. However, the model is not 
# # suitable for this construction as model needs to be invoked with some nontrivial
# # data and this then get passed as an observation thereby producing trivial
# # output in samples['obs'] that is equal to the invocation input. This is
# # why we will follow up with a second script that redefines the model to not
# # have these issues.
# obs_samples = samples["obs"]
# print("Posterior predictive obs mean:", obs_samples.mean())
# print("Posterior predictive obs std. deviation:", obs_samples.std())

# # Visualize the ppd samples
# plt.hist(obs_samples.detach().numpy(), bins=30)
# plt.title("Posterior distribution of obs")
# plt.xlabel("obs")
# plt.ylabel("Frequency")
# plt.show()


# # Infer posterior distribution and generate new data

# import torch
# import pyro
# import pyro.distributions as dist
# from pyro.infer import SVI, Trace_ELBO, Predictive
# from pyro.optim import Adam
# import matplotlib.pyplot as plt

# # Model
# def model(data):
#     mu = pyro.sample("mu", dist.Normal(0., 10.))
#     with pyro.plate("data", len(data)):
#         pyro.sample("obs", dist.Normal(mu, 1.), obs=data)

# # Guide
# def guide(data):
#     mu_loc = pyro.param("mu_loc", torch.tensor(0.))
#     mu_scale = pyro.param("mu_scale", torch.tensor(1.), constraint=dist.constraints.positive)
#     pyro.sample("mu", dist.Normal(mu_loc, mu_scale))

# # Generate some data
# data = torch.randn(100) + 2  # Data generated from a Normal distribution with mean 2

# # Clear parameter store
# pyro.clear_param_store()

# # Perform SVI
# svi = SVI(model, guide, Adam({"lr": 0.01}), loss=Trace_ELBO())
# for step in range(2000):
#     svi.step(data)

# for param_name, param_value in pyro.get_param_store().items():
#     print('{} : {}'.format(param_name, param_value))

# # To generate new data (posterior predictive distribution), build new model
# # in which the prior for the latents z (=mu) is exchanged for the posterior.
# # By plugging in the posterior over z as the prior in a new model, the new model
# # creates data that is parsimoneous with the latents implied by the original data.
# def new_model(data):
#     mu = pyro.sample("mu", dist.Normal(pyro.param("mu_loc"), pyro.param("mu_scale")))
#     return pyro.sample("obs", dist.Normal(mu, 1.), obs=None)

# # Generate new samples
# predictive = Predictive(new_model, guide=guide, num_samples=1000)
# samples = predictive(data)

# plt.figure(1, [5,5], dpi = 300)
# plt.hist(samples['obs'].detach().numpy(), bins = 20)
# plt.title('Posterior predictive - new samples')





