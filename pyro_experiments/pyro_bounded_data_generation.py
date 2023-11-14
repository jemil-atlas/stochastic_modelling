"""
The goal of this script is to test generating data that lies within fixed bounds.
For this, do the following:
    1. Imports and definitions
    2. Stochastic model
    3. Check
"""


"""
    1. Imports and definitions
"""

import torch
import pyro
import pyro.distributions as dist


"""
    2. Stochastic model
"""


def model():
    # Latent variable modeled with a Gaussian distribution
    z = pyro.sample("z", dist.Normal(0, 1))
    
    # Transformation to ensure output is between -1 and 1
    x = 2 * torch.sigmoid(z) - 1
    return x

"""
    3. Check
"""

# Generate samples to check if they are bounded between -1 and 1
samples = [model().item() for _ in range(1000)]

# Check if all samples are within desired bounds
all_within_bounds = all(-1 <= sample <= 1 for sample in samples)
all_within_bounds
