import os
import torch
import pyro
import pyro.distributions as dist
from torch.distributions import constraints
from pyro import poutine
from pyro.infer import SVI, Trace_ELBO, TraceEnum_ELBO, config_enumerate, infer_discrete
from pyro.infer.autoguide import AutoNormal
from pyro.ops.indexing import Vindex

smoke_test = ('CI' in os.environ)
# assert pyro.__version__.startswith('1.8.4')
pyro.set_rng_seed(0)

pyro.clear_param_store()


def model(data, num_components=3):
    print(f"  Running model with {len(data)} data points")
    p = pyro.sample("p", dist.Dirichlet(0.5 * torch.ones(num_components)))
    scale = pyro.sample("scale", dist.LogNormal(0, num_components))
    with pyro.plate("components", num_components):
        loc = pyro.sample("loc", dist.Normal(0, 10))
    with pyro.plate("data", len(data)):
        x = pyro.sample("x", dist.Categorical(p))
        print("    x.shape = {}".format(x.shape))
        pyro.sample("obs", dist.Normal(loc[x], scale), obs=data)
        print("    dist.Normal(loc[x], scale).batch_shape = {}".format(
            dist.Normal(loc[x], scale).batch_shape))

guide = AutoNormal(poutine.block(model, hide=["x", "data"]))

data = torch.randn(10)

pyro.clear_param_store()
print("Sampling:")
model(data)
print("Enumerated Inference:")
elbo = TraceEnum_ELBO(max_plate_nesting=1)
elbo.loss(model, guide, data);