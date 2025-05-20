#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The goal of this script is to use pyro for a very simple model for the levelling
rod calibration data. We start with a minimal dataset that contains only offset,
and scale and analyze the variation in these two parameters. We go for the simplest
possible model here: A single unknown pair (offset, scale) considered deterministic.
This only serves as a baseline model.
For this, do the following:
    1. Imports and definitions
    2. Preprocess data
    3. Build model and guide
    4. Inference with pyro
    5. Plots and illustrations
    
Dataset: minimal
Model: deterministic unknown
Role: baseline

Written by Dr. Jemil Avers Butt, Atlas optimization GmbH, www.atlasoptimization.com.
"""



"""
    1. Imports and definitions
"""


# i) Imports

import pyro
import torch
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# import data
with open("data_list.pkl", "rb") as f:
    data_list = pickle.load(f)
with open("minimal_list.pkl", "rb") as f:
    minimal_list = pickle.load(f)
minimal_tensor = torch.load("minimal_tensor.pt", weights_only=True)


# ii) Definitions

n_obs = len(minimal_list)



"""
    2. Preprocess data
"""

# 1. Extract staff types
staff_types = [entry["staff_type"] for entry in minimal_list]
staff_types_reduced = [entry["staff_type_reduced"] for entry in minimal_list]

# 2. Get unique staff types and assign a color per type
unique_types = sorted(set(staff_types_reduced))
type_to_color = {stype: plt.cm.tab10(i % 10) for i, stype in enumerate(unique_types)}



"""
    3. Build model and guide
"""


# i) Build model

def model(observations = None):
    mu = pyro.param("mu", init_tensor = 10*torch.ones([2]))
    Sigma = pyro.param("Sigma", init_tensor = 1000*torch.eye(2), constraint = 
                       pyro.distributions.constraints.positive_definite)
    
    param_dist = pyro.distributions.MultivariateNormal(loc = mu, covariance_matrix = Sigma) 
    with pyro.plate("batch_plate", size = n_obs):
        param_obs = pyro.sample("param_obs", param_dist, obs = observations)
    
    return param_obs

# ii) Build guide
def guide(observations = None):
    pass


# iii) illustrate model and guide
graphical_model = pyro.render_model(model = model, model_args= None,
                                    render_distributions=True,
                                    render_params=True)
graphical_guide = pyro.render_model(model = guide, model_args= None,
                                    render_distributions=True,
                                    render_params=True)

graphical_model
graphical_guide


# iv) Record example outputs of model and guide prior to training

n_model_samples = 10
predictive = pyro.infer.Predictive
prior_predictive_pretrain = predictive(model, num_samples = n_model_samples)()['param_obs']



"""
    4. Inference with pyro
"""

# i) Set up inference

adam = pyro.optim.Adam({"lr": 0.1})
elbo = pyro.infer.Trace_ELBO(num_particles = 10,
                                 max_plate_nesting = 2)
svi = pyro.infer.SVI(model, guide, adam, elbo)


# ii) Perform svi

data = (minimal_tensor,)
loss_sequence = []
for step in range(1000):
    loss = svi.step(*data)
    loss_sequence.append(loss)
    if step %100 == 0:
        print(f'epoch: {step} ; loss : {loss}')
    
    
# iii) Record example outputs of model and guide post training

prior_predictive_posttrain = predictive(model, num_samples = n_model_samples)()['param_obs']

for name, value in pyro.get_param_store().items():
    print(name, value)

"""
    5. Plots and illustrations
"""


# i) Plot loss
plt.figure(1, dpi = 300)
plt.plot(loss_sequence)
plt.yscale("log")
plt.title('ELBO loss during training (log scale)')
plt.xlabel('Epoch nr')
plt.ylabel('value')


# ii) Plot each group separately

plt.figure(dpi=300)

for stype in unique_types:
    indices = [i for i, t in enumerate(staff_types_reduced) if t == stype]
    points = minimal_tensor[indices]
    plt.scatter(points[:, 0], points[:, 1], label=stype, color=type_to_color[stype])

# 4. Labels and legend
plt.xlabel('Levelling rod offset [µm]')
plt.ylabel('Levelling rod scale [ppm]')
plt.title('Offset and scale of rods by staff type')
plt.legend(title='Staff Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# iii) Joint plot

offsets = minimal_tensor[:, 0].numpy()
scales = minimal_tensor[:, 1].numpy()
x_min = min(offsets) - 10
x_max = max(offsets) + 10
y_min = min(scales) - 10
y_max = max(scales) + 10

# Set up figure with 3 columns
fig, axs = plt.subplots(1, 3, figsize=(15, 5), dpi=300)

# Plot 1: Scatterplot
sns.scatterplot(x=offsets, y=scales, hue=staff_types_reduced, ax=axs[0], palette='tab10', legend=False)
axs[0].set_title("Offset vs Scale")
axs[0].set_xlabel("Offset [µm]")
axs[0].set_ylabel("Scale [ppm]")

# Plot 2: 2D KDE (Pretraining)
sns.kdeplot(
    x= prior_predictive_pretrain[:,:,0].flatten().numpy(), 
    y= prior_predictive_pretrain[:,:,1].flatten().numpy(), ax=axs[1],
    fill=True, cmap='Blues', bw_adjust=3
)
axs[1].set_title("Pretraining KDE")
axs[1].set_xlabel("Offset [µm]")
axs[1].set_ylabel("Scale [ppm]")

# Plot 3: 2D KDE (Posttraining)
sns.kdeplot(
    x= prior_predictive_posttrain[:,:,0].flatten().numpy(),
    y= prior_predictive_posttrain[:,:,1].flatten().numpy(), ax=axs[2],
    fill=True, cmap='Greens', bw_adjust=3
)
axs[2].set_title("Posttraining KDE")
axs[2].set_xlabel("Offset [µm]")
axs[2].set_ylabel("Scale [ppm]")

for ax in axs: ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)

plt.tight_layout()
plt.show()



