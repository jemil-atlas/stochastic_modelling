#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The goal of this script is to showcase tensorboard functionality for logging 
the progress during pyros svi.We illustrate the learning process during the
inference of a constant mean for some data.
For this, do the following:
    1. Definitions and imports
    2. Simulate some data
    3. Setup the pyro model
    4. Setup svi and run
        
"""



"""
    1. Definitions and imports
"""


# Importing required packages
import pyro
import pyro.distributions as dist
import pyro.optim as optim
from pyro.infer import SVI, Trace_ELBO
from torch.utils.tensorboard import SummaryWriter
import torch

# Initialize SummaryWriter
writer = SummaryWriter('runs/constant_mean_experiment')
pyro.clear_param_store()



"""
    2. Simulate some data
"""


# Generate synthetic data
data = torch.randn(100) + 10  # 100 data points centered around 10



"""
    3. Setup the pyro model
"""


# Model: A constant mean
def model(data):
    mean = pyro.param("mean", torch.tensor(0.))
    with pyro.plate("data", len(data)):
        pyro.sample("obs", dist.Normal(mean, 1), obs=data)

# Guide: Also a constant mean
guide = pyro.infer.autoguide.AutoNormal(model)



"""
    4. Setup svi and run
"""


# Set up SVI with Adam optimizer
pyro.clear_param_store()
adam = optim.Adam({"lr": 0.1})
svi = SVI(model, guide, adam, loss=Trace_ELBO())

# Training loop
num_epochs = 500
for epoch in range(num_epochs):
    loss = svi.step(data)
    
    # Log the loss and parameter values in TensorBoard
    writer.add_scalar("Loss/train", loss, epoch)
    writer.add_scalar("Mean/true", 10, epoch)  # The true mean

# Close SummaryWriter
writer.close()

# Final estimated mean
final_mean = pyro.param("mean").item()
print(f"Final estimated mean: {final_mean}")

# After running this script, you can start TensorBoard by running
# `tensorboard --logdir=runs` in the terminal, and go to `http://localhost:6006/`
# in your web browser to see the training progress.