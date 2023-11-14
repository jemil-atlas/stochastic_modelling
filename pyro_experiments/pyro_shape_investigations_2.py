#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The goal of this script is to perform even more investigations into tensor shapes.
"""


"""
    1. Imports and perparations
"""

import os
import torch
import pyro
from torch.distributions import constraints
from pyro.distributions import Bernoulli, Categorical, MultivariateNormal, Normal
from pyro.distributions.util import broadcast_shape
from pyro.infer import Trace_ELBO, TraceEnum_ELBO, config_enumerate
import pyro.poutine as poutine
from pyro.optim import Adam

# Enable validation checks
pyro.enable_validation(True)


smoke_test = ('CI' in os.environ)
# assert pyro.__version__.startswith('1.8.5')

# We'll ue this helper to check our models are correct.
def test_model(model, guide, loss):
    pyro.clear_param_store()
    loss.loss(model, guide)
    

"""
    2. Simple first investigations
"""
d = pyro.distributions.Normal(0,1)
    
x = d.sample()
assert x.shape == d.batch_shape + d.event_shape
assert d.log_prob(x).shape == d.batch_shape

sample_shape = torch.Size([10])
x2 = d.sample(sample_shape)
assert x2.shape == sample_shape + d.batch_shape + d.event_shape


"""
    3. Batch dims and event dims
"""

# Example univariate
d = Bernoulli(0.5)
assert d.batch_shape == ()
assert d.event_shape == ()
x = d.sample()
assert x.shape == ()
assert d.log_prob(x).shape == ()

# batched distribution by passing multidim params
d = Bernoulli(0.5 * torch.ones(3,4))
assert d.batch_shape == (3, 4)
assert d.event_shape == ()
x = d.sample()
assert x.shape == (3, 4)
assert d.log_prob(x).shape == (3, 4)

# batched distribution by expand method
d = Bernoulli(torch.tensor([0.1, 0.2, 0.3, 0.4])).expand([3, 4])
assert d.batch_shape == (3, 4)
assert d.event_shape == ()
x = d.sample()
assert x.shape == (3, 4)
assert d.log_prob(x).shape == (3, 4)

# introduce nontrivial event shape via multivaraite distribution
d = MultivariateNormal(torch.zeros(3), torch.eye(3, 3))
assert d.batch_shape == ()
assert d.event_shape == (3,)
x = d.sample()
assert x.shape == (3,)            # == batch_shape + event_shape
assert d.log_prob(x).shape == ()  # == batch_shape


"""
    4. Reshaping distributions & plates
"""

# declare 1 dim from the right as dependent
d = Bernoulli(0.5 * torch.ones(3,4)).to_event(1)
assert d.batch_shape == (3,)
assert d.event_shape == (4,)
x = d.sample()
assert x.shape == (3, 4)
assert d.log_prob(x).shape == (3,)


# investigate relationships in more complicated model
def model1():
    a = pyro.sample("a", Normal(0, 1))
    b = pyro.sample("b", Normal(torch.zeros(2), 1).to_event(1))
    with pyro.plate("c_plate", 2):
        c = pyro.sample("c", Normal(torch.zeros(2), 1))
    with pyro.plate("d_plate", 3):
        d = pyro.sample("d", Normal(torch.zeros(3,4,5), 1).to_event(2))
    assert a.shape == ()       # batch_shape == ()     event_shape == ()
    assert b.shape == (2,)     # batch_shape == ()     event_shape == (2,)
    assert c.shape == (2,)     # batch_shape == (2,)   event_shape == ()
    assert d.shape == (3,4,5)  # batch_shape == (3,)   event_shape == (4,5)

    x_axis = pyro.plate("x_axis", 3, dim=-2)
    y_axis = pyro.plate("y_axis", 2, dim=-3)
    with x_axis:
        x = pyro.sample("x", Normal(0, 1))
    with y_axis:
        y = pyro.sample("y", Normal(0, 1))
    with x_axis, y_axis:
        xy = pyro.sample("xy", Normal(0, 1))
        z = pyro.sample("z", Normal(0, 1).expand([5]).to_event(1))
    assert x.shape == (3, 1)        # batch_shape == (3,1)     event_shape == ()
    assert y.shape == (2, 1, 1)     # batch_shape == (2,1,1)   event_shape == ()
    assert xy.shape == (2, 3, 1)    # batch_shape == (2,3,1)   event_shape == ()
    assert z.shape == (2, 3, 1, 5)  # batch_shape == (2,3,1)   event_shape == (5,)

trace = poutine.trace(model1).get_trace()
trace.compute_log_prob()  # optional, but allows printing of log_prob shapes
trace.nodes
print(trace.format_shapes())



# batch dims | event dims
# -----------+-----------
#            |        a = sample("a", Normal(0, 1))
#            |2       b = sample("b", Normal(zeros(2), 1)
#            |                        .to_event(1))
#            |        with plate("c", 2):
#           2|            c = sample("c", Normal(zeros(2), 1))
#            |        with plate("d", 3):
#           3|4 5         d = sample("d", Normal(zeros(3,4,5), 1)
#            |                       .to_event(2))
#            |
#            |        x_axis = plate("x", 3, dim=-2)
#            |        y_axis = plate("y", 2, dim=-3)
#            |        with x_axis:
#         3 1|            x = sample("x", Normal(0, 1))
#            |        with y_axis:
#       2 1 1|            y = sample("y", Normal(0, 1))
#            |        with x_axis, y_axis:
#       2 3 1|            xy = sample("xy", Normal(0, 1))
#       2 3 1|5           z = sample("z", Normal(0, 1).expand([5])
#            |                       .to_event(1))

# data subsampling with a plate
data = torch.arange(100.)

def model2():
    mean = pyro.param("mean", torch.zeros(len(data)))
    with pyro.plate("data", len(data), subsample_size=10) as ind:
        assert len(ind) == 10    # ind is a LongTensor that indexes the subsample.
        batch = data[ind]        # Select a minibatch of data.
        mean_batch = mean[ind]   # Take care to select the relevant per-datum parameters.
        # Do stuff with batch:
        x = pyro.sample("x", Normal(mean_batch, 1), obs=batch)
        assert len(x) == 10
        print(ind)
        print(batch.shape)

trace = poutine.trace(model2).get_trace()
trace.compute_log_prob()  # optional, but allows printing of log_prob shapes
trace.nodes
print(trace.format_shapes())