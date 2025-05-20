#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The goal of this script is to use pyro for a simple hierarchical model for the levelling
rod calibration data. We start with a minimal dataset that contains only offset,
and scale and analyze the variation in these two parameters dependent on rod class.
We go for a simple hierarchical deterministic model: We assume a parameter pair
(offset, scale) for each rod class. This means that the params are learned per class 
and we assume random rod construction but perfect measurement accuracy.
This only serves as a baseline model.
For this, do the following:
    1. Imports and definitions
    2. Preprocess data
    3. Build model and guide
    4. Inference with pyro
    5. Plots and illustrations
    
Dataset: minimal
Model: deterministic params for each rod class
Role: first hierarchical bayesian

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
n_types = len(unique_types)
type_to_color = {stype: plt.cm.tab10(i % 10) for i, stype in enumerate(unique_types)}
type_to_index = {stype: i for i, stype in enumerate(unique_types)}
index_to_type = {i: stype for i, stype in enumerate(unique_types)}

class_indices = torch.tensor([type_to_index[staff_type] for staff_type in staff_types_reduced])


"""
    3. Build model and guide
"""


# Model explanation
# The model is a generative process for the data. The story for this model goes
# like the following: 
#   1. Each class of rods is produced differently with alpha = (offset, scale) being 
#       different for each rod but the distribution of alpha is the same for rods
#       of the same class
#   ii. Each rod is then measured with 100% accuracy so that the observations are
#       the actual values of alpha.
# Overall this translates to the following probabilistic model 
# mu = UnknownParameter         [n_class,2]
# Sigma = UnknownVariance       [n_class,2,2]
# obs ~ N(mu, Sigma)

# i) Build model

def model(observations = None):
    # Invoke params
    mu = pyro.param("mu", init_tensor = 10*torch.ones([n_types, 2]))
    Sigma = pyro.param("Sigma", init_tensor = 10000 * (torch.eye(2).unsqueeze(0)).expand([n_types,2,2]),
                       constraint = pyro.distributions.constraints.positive_definite)
    
    # Extend params
    mu_extended = mu[class_indices,:]
    Sigma_extended = Sigma[class_indices,:,:]
    param_dist = pyro.distributions.MultivariateNormal(loc = mu_extended,
                                            covariance_matrix = Sigma_extended)
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

def build_dict_from_data(data):
    data_dict = {}
    for class_idx, class_name in index_to_type.items():
        # Find obs indices belonging to this class
        obs_mask = (class_indices == class_idx)                   # [n_obs] boolean
        obs_indices = obs_mask.nonzero(as_tuple=True)[0]          # [n_class_obs]
    
        # Select and stack observations for this class
        data_dict[class_name] = data[:, obs_indices, :]
        
    return data_dict

n_model_samples = 10
predictive = pyro.infer.Predictive
prior_predictive_pretrain = predictive(model, num_samples = n_model_samples)()['param_obs']
prior_predictive_pretrain_dict = build_dict_from_data(prior_predictive_pretrain)





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
prior_predictive_posttrain_dict = build_dict_from_data(prior_predictive_posttrain)

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
for stype in unique_types:
    indices = [i for i, t in enumerate(staff_types_reduced) if t == stype]
    points = minimal_tensor[indices]
    plt.scatter(points[:, 0], points[:, 1], label=stype, color=type_to_color[stype])

# 4. Labels and legend
axs[0].xlabel('Levelling rod offset [µm]')
axs[0].ylabel('Levelling rod scale [ppm]')
axs[0].title('Offset and scale of rods by staff type')
axs[0].legend(title='Staff Type', bbox_to_anchor=(1.05, 1), loc='upper left')
axs[0].tight_layout()
axs[0].show()

# Plot 2: 2D KDE (Pretraining)
for class_name, tensor in prior_predictive_pretrain_dict.items():
    # Flatten to [n_samples * n_obs_class, 2]
    x = tensor[:, :, 0].flatten().numpy()
    y = tensor[:, :, 1].flatten().numpy()
    
    sns.kdeplot(
        x=x, y=y, ax=axs[1],
        fill=False, bw_adjust=2, label=class_name
    )

axs[1].set_title("Pretraining KDE by Class")
axs[1].set_xlabel("Offset [µm]")
axs[1].set_ylabel("Scale [ppm]")
# axs[1].legend(title="Rod Type")

# Plot 3: 2D KDE (Posttraining)
for class_name, tensor in prior_predictive_posttrain_dict.items():
    # Flatten to [n_samples * n_obs_class, 2]
    x = tensor[:, :, 0].flatten().numpy()
    y = tensor[:, :, 1].flatten().numpy()
    
    sns.kdeplot(
        x=x, y=y, ax=axs[1],
        fill=False, bw_adjust=2, label=class_name
    )

axs[1].set_title("Posttraining KDE by Class")
axs[1].set_xlabel("Offset [µm]")
axs[1].set_ylabel("Scale [ppm]")
# axs[1].legend(title="Rod Type")


for ax in axs: ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)

plt.tight_layout()
plt.show()



