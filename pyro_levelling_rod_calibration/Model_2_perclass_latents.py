#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The goal of this script is to use pyro for a simple hierarchical model for the levelling
rod calibration data. We start with a minimal dataset that contains only offset,
and scale and analyze the variation in these two parameters dependent on rod class.
Since we also use the rod id, we can also incorporate the uncertainty of the
calibration procedure itself. We go for a simple hierarchical bayesian model: 
We assume the true rod params (offset, scale) sampled from a production distribution
that is different for each rod class. The observed rod params are then true rod
params plus some noise on top. This means that a latent distribution for the rod
params is learned per class, together with the calibration uncertainty.
For this, do the following:
    1. Imports and definitions
    2. Preprocess data
    3. Build model and guide
    4. Inference with pyro
    5. Plots and illustrations
    
Dataset: minimal
Model: latent vars for each rod class, measurement uncertainty
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
with open("../data_stochastic_modelling/data_levelling_rod_calibration/data_list.pkl", "rb") as f:
    data_list = pickle.load(f)
with open("../data_stochastic_modelling/data_levelling_rod_calibration/minimal_list.pkl", "rb") as f:
    minimal_list = pickle.load(f)
minimal_tensor = torch.load("../data_stochastic_modelling/data_levelling_rod_calibration/minimal_tensor.pt", weights_only=True)


# ii) Definitions

n_obs = len(minimal_list)



"""
    2. Preprocess data
"""

# 1. Extract staff types
staff_types_reduced = [entry["staff_type_reduced"] for entry in minimal_list]
staff_ids = [entry["staff_id"] for entry in minimal_list]


# 2. Get unique staff types and build obs -> class index tensor
unique_types = sorted(set(staff_types_reduced))
n_types = len(unique_types)
type_to_color = {stype: plt.cm.tab10(i % 10) for i, stype in enumerate(unique_types)}
type_to_index = {stype: i for i, stype in enumerate(unique_types)}
index_to_type = {i: stype for i, stype in enumerate(unique_types)}
obs_class_indices = torch.tensor([type_to_index[staff_type] for staff_type in staff_types_reduced])

# 3. Get unique rod ids and build obs -> id index tensor
unique_ids = sorted(set(staff_ids))
obs_id_indices = torch.tensor(staff_ids)
id_class_indices_dict = {}               # {staff_id: class_idx}
for row in minimal_list:            
    sid  = row["staff_id"]
    cidx = type_to_index[row["staff_type_reduced"]]
    id_class_indices_dict[sid] = cidx  
id_class_indices_dict = dict(sorted(id_class_indices_dict.items()))

# Build id -> class indextensor
n_ids = max(id_class_indices_dict) +1       # assuming staff_id starts at 0 and is dense
id_class_indices = torch.empty(n_ids, dtype=torch.long)
for sid, cidx in id_class_indices_dict.items():
    id_class_indices[sid] = cidx

# obs_class_indices = tensor. For each observation, the corresponding class index [n_obs]
# obs_id_indices = tensor. For each observation, the corresponding id index [n_obs]
# id_class_indices = tensor. For each id, the corresponding class index [n_ids]


"""
    3. Build model and guide
"""


# Model explanation
# The model is a generative process for the data. The story for this model goes
# like the following: 
#   1. Each class of rods is produced differently with alpha = (offset, scale) being 
#       different for each rod but the distribution of alpha is the same for rods
#       of the same class
#   ii. Each rod is then measured with unknown calibration accuracy Sigma_cal,
#       so that the observations are the actual values of alpha.
# We want the posterior distributions of the params (offset, scale) for each class
# given the observations and the calibration uncertainty Sigma_cal.
# Overall this translates to the following probabilistic model 
# mu = UnknownParameter         [n_class,2]
# Sigma = UnknownVariance       [n_class,2,2]
# alpha ~ N(mu, Sigma)
# Sigma_cal = UnknownVariance   [2,2]
# obs ~ N(alpha, Sigma_cal)

# i) Build model

def model(observations = None):
    # Invoke params
    mu_alpha = pyro.param("mu", init_tensor = 10*torch.ones([n_types, 2]))
    Sigma_alpha = pyro.param("Sigma", init_tensor = 10000 * (torch.eye(2).unsqueeze(0)).expand([n_types,2,2]),
                       constraint = pyro.distributions.constraints.positive_definite)
    
    # Extend params
    mu_extended = mu_alpha[obs_class_indices,:]
    Sigma_extended = Sigma_alpha[obs_class_indices,:,:]
    
    # Sample alpha, one sample per staff_id
    alpha_dist = pyro.distributions.MultivariateNormal(loc = mu_extended,
                                            covariance_matrix = Sigma_extended)
    with pyro.plate("id_plate", size = n_obs):
        alpha = pyro.sample("alpha", alpha_dist)
    
    # Extend params and Sample obs
    mu_extended = mu_alpha[obs_class_indices,:]
    Sigma_extended = Sigma_alpha[obs_class_indices,:,:]
    obs_dist = pyro.distributions.MultivariateNormal(loc = mu_extended,
                                            covariance_matrix = Sigma_extended)
    with pyro.plate("obs_plate", size = n_obs):
        param_obs = pyro.sample("obs", obs_dist, obs = observations)
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
        obs_mask = (obs_class_indices == class_idx)                   # [n_obs] boolean
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
    axs[0].scatter(points[:, 0], points[:, 1], label=stype, color=type_to_color[stype])

# 4. Labels and legend
axs[0].set_xlabel('Levelling rod offset [µm]')
axs[0].set_ylabel('Levelling rod scale [ppm]')
axs[0].set_title('Offset and scale of rods by staff type')
axs[0].legend(title='Staff Type', bbox_to_anchor=(1.05, 1), loc='upper left')


# Plot 2: 2D KDE (Pretraining)
for class_name, tensor in prior_predictive_pretrain_dict.items():
    # Flatten to [n_samples * n_obs_class, 2]
    x = tensor[:, :, 0].flatten().numpy()
    y = tensor[:, :, 1].flatten().numpy()
    
    sns.kdeplot(
        x=x, y=y, ax=axs[1],
        fill=True, bw_adjust=2, alpha = 0.3, label=class_name
    )

axs[1].set_title("Pretraining KDE by Class")
axs[1].set_xlabel("Offset [µm]")
axs[1].set_ylabel("Scale [ppm]")

# Plot 3: 2D KDE (Posttraining)
for class_name, tensor in prior_predictive_posttrain_dict.items():
    # Flatten to [n_samples * n_obs_class, 2]
    x = tensor[:, :, 0].flatten().numpy()
    y = tensor[:, :, 1].flatten().numpy()
    
    sns.kdeplot(
        x=x, y=y, ax=axs[2],
        fill=True, bw_adjust=2, alpha = 0.3, label=class_name
    )

axs[2].set_title("Posttraining KDE by Class")
axs[2].set_xlabel("Offset [µm]")
axs[2].set_ylabel("Scale [ppm]")
# axs[1].legend(title="Rod Type")


for ax in axs: ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)

plt.tight_layout()
plt.show()



