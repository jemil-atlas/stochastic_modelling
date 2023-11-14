# i) imports


import torch
import pyro
import pyro.distributions as dist


# ii) Model is multivariate Gaussian, condition on first element of event

# Fixed params of the model
xy_mean = torch.zeros(2)
xy_cov = torch.tensor([[1,0.5],[0.5,1]])

# Construct the model for [x,y] ~ N(mean, cov).
# x is a sample site referencing the first element of event [x,y]
def model(observations_x = None):
    xy = pyro.sample('xy', dist.MultivariateNormal(xy_mean,xy_cov))
    # x = pyro.deterministic('x', xy[0])
    x = pyro.sample('x',dist.Normal(xy[0],0.01), obs = observations_x)
    return xy

# Construct the conditional distribution of [x,y] given observation of x
observation_x = torch.ones(1)
conditional_model = pyro.condition(model, data={'x': observation_x})


# iii) Sample from the conditional -> but this does not work as imagined
realization = conditional_model()


# sequence_realizations = [conditional_model() for _ in range(1000)]
# print('Conditional_mean is {} <--wrong, should be {}'.format(torch.mean(sequence_realizations),1))
 


##################   
# Now do inference
import logging

# i) Create the guide

auto_guide = pyro.infer.autoguide.AutoNormal(model)


# ii) Run the optimization

adam = pyro.optim.Adam({"lr": 0.02})
elbo = pyro.infer.Trace_ELBO()
svi = pyro.infer.SVI(model, auto_guide, adam, elbo)

losses = []
for step in range(1000):  # Consider running for more steps.
    loss = svi.step(observation_x)
    losses.append(loss)
    if step % 100 == 0:
        logging.info("Elbo loss: {}".format(loss))
    
# iii) Sample from distribution
model()
