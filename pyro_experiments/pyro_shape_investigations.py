#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The goal of this sscript is to investigate pyro shapes and learn to distinguish
between event_shape, batch_shape, sample_shape
"""

"""
    1. Imports and definitions
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

smoke_test = ('CI' in os.environ)
# assert pyro.__version__.startswith('1.8.4')

# We'll ue this helper to check our models are correct.
def test_model(model, guide, loss):
    pyro.clear_param_store()
    loss.loss(model, guide)




"""
    2. Univariate distribution
"""

#       |      iid     | independent | dependent
# ------+--------------+-------------+------------
# shape = sample_shape + batch_shape + event_shape
#
# See commented blob below for full set of examples


# With a single input leading to a trivial batch_shape
d = Bernoulli(0.5)
assert d.batch_shape == ()
assert d.event_shape == ()
x = d.sample()
assert x.shape == ()
assert d.log_prob(x).shape == ()

x = d.sample()
assert x.shape == d.batch_shape + d.event_shape

assert d.log_prob(x).shape == d.batch_shape

sample_shape = [3]
x2 = d.sample(sample_shape)
x2.shape


# With a batched input leading to anontrivial batch_shape
d = Bernoulli(0.5 * torch.ones(3,4))
assert d.batch_shape == (3, 4)
assert d.event_shape == ()
x = d.sample()
assert x.shape == (3, 4)
assert d.log_prob(x).shape == (3, 4)


# Use expand() to achieve the same result. Care for dimensions.
d = Bernoulli(torch.tensor([0.1, 0.2, 0.3, 0.4])).expand([3, 4])
assert d.batch_shape == (3, 4)
assert d.event_shape == ()
x = d.sample()
assert x.shape == (3, 4)
assert d.log_prob(x).shape == (3, 4)



"""
    3. Multivariate distribution
"""

# Multivariate normal leads to nontrivial event_shape
d = MultivariateNormal(torch.zeros(3), torch.eye(3, 3))
assert d.batch_shape == ()
assert d.event_shape == (3,)
x = d.sample()
assert x.shape == (3,)            # == batch_shape + event_shape
assert d.log_prob(x).shape == ()  # == batch_shape

# Multivariate normal can also be batched -> event_shape, batch_shape
d = MultivariateNormal(torch.zeros(5,3), torch.eye(3, 3))
assert d.batch_shape == (5,)
assert d.event_shape == (3,)
x = d.sample()
assert x.shape == (5,3)            # == batch_shape + event_shape
assert d.log_prob(x).shape == (5,)  # == batch_shape



"""
    4. Reshaping into dependence
"""

# Use to_event to declare the 1 rightmost dimension dependent
d = Bernoulli(0.5 * torch.ones(3,4)).to_event(1)
assert d.batch_shape == (3,)
assert d.event_shape == (4,)
x = d.sample()
assert x.shape == (3, 4)
assert d.log_prob(x).shape == (3,)




"""
    5. Independent dims with plate
"""








# import torch
# from torch.distributions.normal import Normal
# from torch.distributions.multivariate_normal import MultivariateNormal

# sample_shape, batch_shape, event_shape
# Row 1: [], [], []

# >>> dist = Normal(0.0, 1.0)
# >>> sample_shape = torch.Size([])
# >>> dist.sample(sample_shape)
# tensor(-1.3349)
# >>> (sample_shape, dist.batch_shape, dist.event_shape)
# (torch.Size([]), torch.Size([]), torch.Size([]))


# Row 2: [2], [], []

# >>> dist = Normal(0.0, 1.0)
# >>> sample_shape = torch.Size([2])
# >>> dist.sample(sample_shape)
# tensor([ 0.2786, -1.4113])
# >>> (sample_shape, dist.batch_shape, dist.event_shape)
# (torch.Size([2]), torch.Size([]), torch.Size([]))


# Row 3: [], [2], []

# >>> dist = Normal(torch.zeros(2), torch.ones(2))
# >>> sample_shape = torch.Size([])
# >>> dist.sample(sample_shape)
# tensor([0.0101, 0.6976])
# >>> (sample_shape, dist.batch_shape, dist.event_shape)
# (torch.Size([]), torch.Size([2]), torch.Size([]))


# Row 4: [], [], [2]

# >>> dist = MultivariateNormal(torch.zeros(2), torch.eye(2))
# >>> sample_shape = torch.Size([])
# >>> dist.sample(sample_shape)
# tensor([ 0.2880, -1.6795])
# >>> (sample_shape, dist.batch_shape, dist.event_shape)
# (torch.Size([]), torch.Size([]), torch.Size([2]))


# Row 5: [], [2], [2]

# >>> dist = MultivariateNormal(torch.zeros(2, 2), torch.eye(2))
# >>> sample_shape = torch.Size([])
# >>> dist.sample(sample_shape)
# tensor([[-0.4703,  0.4152],
#         [-1.6471, -0.6276]])
# >>> (sample_shape, dist.batch_shape, dist.event_shape)
# (torch.Size([]), torch.Size([2]), torch.Size([2]))


# Row 6: [2], [], [2]

# >>> dist = MultivariateNormal(torch.zeros(2), torch.eye(2))
# >>> sample_shape = torch.Size([2])
# >>> dist.sample(sample_shape)
# tensor([[ 2.2040, -0.7195],
#         [-0.4787,  0.0895]])
# >>> (sample_shape, dist.batch_shape, dist.event_shape)
# (torch.Size([2]), torch.Size([]), torch.Size([2]))


# Row 7: [2], [2], []

# >>> dist = Normal(torch.zeros(2), torch.ones(2))
# >>> sample_shape = torch.Size([2])
# >>> dist.sample(sample_shape)
# tensor([[ 0.2639,  0.9083],
#         [-0.7536,  0.5296]])
# >>> (sample_shape, dist.batch_shape, dist.event_shape)
# (torch.Size([2]), torch.Size([2]), torch.Size([]))


# Row 8: [2], [2], [2]

# >>> dist = MultivariateNormal(torch.zeros(2, 2), torch.eye(2))
# >>> sample_shape = torch.Size([2])
# >>> dist.sample(sample_shape)
# tensor([[[ 0.4683,  0.6118],
#          [ 1.0697, -0.0856]],

#         [[-1.3001, -0.1734],
#          [ 0.4705, -0.0404]]])
# >>> (sample_shape, dist.batch_shape, dist.event_shape)
# (torch.Size([2]), torch.Size([2]), torch.Size([2]))

