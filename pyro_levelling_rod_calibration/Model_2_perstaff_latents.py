#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The goal of this script is to use pyro for a simple hierarchical model for the levelling
rod calibration data. We start with a minimal dataset that contains only offset,
and scale and analyze the variation in these two parameters dependent on rod class.
Since we also use the rod id, we can also incorporate the uncertainty of the
calibration procedure itself. We go for a simple hierarchical bayesian model: 
We assume the true rod params (offset, scale) sampled for each rod separately
from a production distribution that is different for each rod class. The observed
rod params are then true rod params plus some noise on top. This means that a latent
distribution for the rod params is learned for each rod, together with the global
calibration uncertainty and a (mu, Sigma) pair for production distribution of 
each rod class.
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
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import arviz as az


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
# alpha ~ N(mu_alpha, Sigma_alpha)
# Sigma_cal = UnknownVariance   [2,2]
# obs ~ N(alpha, Sigma_cal)

# i) Build model

def model(observations = None):
    # Invoke params
    mu = pyro.param("mu", init_tensor = 10*torch.ones([n_types, 2]))
    Sigma = pyro.param("Sigma", init_tensor = 10000 * (torch.eye(2).unsqueeze(0)).expand([n_types,2,2]),
                       constraint = pyro.distributions.constraints.positive_definite)
    Sigma_cal = pyro.param("Sigma_cal", init_tensor = 10000 * (torch.eye(2)),
                       constraint = pyro.distributions.constraints.positive_definite)
    
    # Extend params to rod ids
    mu_alpha = mu[id_class_indices,:]
    Sigma_alpha = Sigma[id_class_indices,:,:]
    
    # Sample alpha, one sample per staff_id
    alpha_dist = pyro.distributions.MultivariateNormal(loc = mu_alpha,
                                            covariance_matrix = Sigma_alpha)
    with pyro.plate("id_plate", size = n_ids):
        alpha = pyro.sample("alpha", alpha_dist)
    # alpha is ordered as alpha_0, alpha_1, ... , alpha_i [n_ids,2] where i is id
    
    # Extend alpha and Sample obs
    alpha_extended = alpha[obs_id_indices,:]
    obs_dist = pyro.distributions.MultivariateNormal(loc = alpha_extended,
                                            covariance_matrix = Sigma_cal)
    with pyro.plate("obs_plate", size = n_obs):
        param_obs = pyro.sample("obs", obs_dist, obs = observations)
    return param_obs


# ii) Build guide

def guide(observations = None):
    # Set up posterior parameters
    mu_alpha_post = pyro.param("mu_alpha_post", init_tensor = 10*torch.ones([n_ids, 2]))
    Sigma_alpha_post = pyro.param("Sigma_alpha_post", init_tensor = 10000 * (torch.eye(2).unsqueeze(0)).expand([n_ids,2,2]),
                       constraint = pyro.distributions.constraints.positive_definite)  
    
    # Sample from posterior distribution
    alpha_post_dist = pyro.distributions.MultivariateNormal(loc = mu_alpha_post,
                                            covariance_matrix = Sigma_alpha_post)
    with pyro.plate("id_plate", size = n_ids):
        alpha_post_sample = pyro.sample('alpha', alpha_post_dist)
    return alpha_post_sample


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
n_guide_samples = 100

predictive = pyro.infer.Predictive
prior_predictive_pretrain_dict = predictive(model, num_samples = n_model_samples)()
posterior_pretrain_dict = predictive(guide, num_samples = n_guide_samples)()
posterior_predictive_pretrain_dict = predictive(model, guide = guide, num_samples = n_model_samples)()

# evaluate per-id and per-class data
def build_dicts_from_data(data):
    data_dict_class = {}
    for class_idx, class_name in index_to_type.items():
        # Find obs indices belonging to this class
        obs_mask = (obs_class_indices == class_idx)                   # [n_obs] boolean
        obs_indices = obs_mask.nonzero(as_tuple=True)[0]              # [n_types]
    
        # Select and stack observations for this class
        data_dict_class[class_name] = data[:, obs_indices, :]
        
    data_dict_id = {}
    for staff_id in id_class_indices_dict.keys():
        # Find obs indices belonging to this staff_id
        obs_mask = (obs_id_indices == staff_id)                     # [n_obs] boolean
        obs_indices = obs_mask.nonzero(as_tuple=True)[0]            # [n_ids]
    
        # Select and stack observations for this class
        data_dict_id[staff_id] = data[:, obs_indices, :]
        
    return {'data_dict_class' : data_dict_class,
            'data_dict_id' : data_dict_id}



"""
    4. Inference with pyro
"""

# i) Set up inference

adam = pyro.optim.Adam({"lr": 0.03})
elbo = pyro.infer.Trace_ELBO(num_particles = 10,
                                 max_plate_nesting = 2)
svi = pyro.infer.SVI(model, guide, adam, elbo)


# ii) Perform svi

data = (minimal_tensor,)
loss_sequence = []
for step in range(3000):
    loss = svi.step(*data)
    loss_sequence.append(loss)
    if step %100 == 0:
        print(f'epoch: {step} ; loss : {loss}')
    
    
# iii) Record example outputs of model and guide post training

prior_predictive_posttrain_dict = predictive(model, num_samples = n_model_samples)()
posterior_posttrain_dict = predictive(guide, num_samples = n_guide_samples)()
posterior_predictive_posttrain_dict = predictive(model, guide = guide, num_samples = n_model_samples)()

# for name, value in pyro.get_param_store().items():
#     print(name, value)



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
# In this plot, we showcase the impact of training onto the parameter mu, which
# determines the prior distribution of the parameters per class.

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
prior_obs_dicts = build_dicts_from_data(prior_predictive_pretrain_dict['obs'])
prior_class_obs_dict = prior_obs_dicts['data_dict_class']
prior_id_obs_dict = prior_obs_dicts['data_dict_id']
for class_name, tensor in prior_class_obs_dict.items():
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
prior_obs_dicts = build_dicts_from_data(prior_predictive_posttrain_dict['obs'])
prior_class_obs_dict = prior_obs_dicts['data_dict_class']
prior_id_obs_dict = prior_obs_dicts['data_dict_id']
for class_name, tensor in prior_class_obs_dict.items():
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


# iii) Pre/ Posttrain prior
# Plot prior for different classes before and after training

def to_df(samples, cls_labels, label):
    """
    samples : Tensor [n_samp, n_obs, 2]
    cls_labels : list[str] length n_obs
    label : 'prior'|'post'
    """
    n_samp, n_obs, _ = samples.shape
    flat = samples.reshape(-1, 2)
    df = pd.DataFrame(flat, columns=["offset", "scale"])
    df["class"] = cls_labels * n_samp
    df["which"] = label
    return df

# raw observations' class labels
cls_labels = staff_types_reduced

df_pretrain  = to_df(prior_predictive_pretrain_dict["obs"],  cls_labels, "pretrain")
df_posttrain   = to_df(prior_predictive_posttrain_dict["obs"], cls_labels, "posttrain")
df_long   = pd.concat([df_pretrain, df_posttrain], ignore_index=True)

g = sns.FacetGrid(
        df_long, col="class", row="which",
        height=3, aspect=1, despine=False
    )
g.map_dataframe(
        sns.kdeplot, x="offset", y="scale",
        fill=True, thresh=0.02, bw_adjust=1.2
    )
g.set_titles(row_template="{row_name}", col_template="{col_name}")
g.set(xlabel="Offset [µm]", ylabel="Scale [ppm]")
xmin, xmax = -200,  200    #  example numbers
ymin, ymax = -50,  50

for ax in g.axes.flat:
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
plt.tight_layout()


# iv) Caterpillar plot shows posterior distribution for each rod

alpha_samples = posterior_posttrain_dict["alpha"]    # [n_samp, n_ids, 2]
mean_off  = alpha_samples[:, :, 0].mean(0).numpy()   # [n_ids]
hdi_off   = az.hdi(alpha_samples[:, :, 0].numpy(), hdi_prob=0.95)  # [n_ids,2]

fig, ax = plt.subplots(figsize=(6, len(mean_off)*0.25))
ypos = np.arange(len(mean_off))

for i, (m, h) in enumerate(zip(mean_off, hdi_off)):
    cls_idx = id_class_indices[i]                 # class index for staff_id=i
    color   = type_to_color[index_to_type[int(cls_idx)]]
    ax.errorbar(
        x=m, y=i,
        xerr=[[m - h[0]], [h[1] - m]],
        fmt='o', color=color, capsize=3
    )

ax.set_yticks(ypos)
ax.set_yticklabels([f"id {i}" for i in ypos])
ax.set_xlabel("Offset [µm]")
ax.set_title("Posterior mean ± 95 % HPDI per rod")
plt.tight_layout()


# v) Arrow plot shows the difference between trained prior and the posterior

alpha_mean = posterior_posttrain_dict["alpha"].mean(0)   # [n_ids,2]
mu_learned = pyro.param("mu").detach()

fig, ax = plt.subplots(figsize=(5,5))
ax.scatter(mu_learned[:,0], mu_learned[:,1], c='k', marker='s', label='class µ')

for sid in range(n_ids):
    cls   = id_class_indices[sid]
    color = type_to_color[index_to_type[int(cls)]]
    ax.arrow(mu_learned[cls,0], mu_learned[cls,1],
             alpha_mean[sid,0] - mu_learned[cls,0],
             alpha_mean[sid,1] - mu_learned[cls,1],
             head_width=3, length_includes_head=True,
             color=color, alpha=.6)

ax.set_xlabel("Offset [µm]")
ax.set_ylabel("Scale [ppm]")
ax.set_title("Each rod’s posterior mean α̂ vs. its class centre µ")
plt.tight_layout(); plt.show()




# vi) Plot with learned priors

# Plot raw observations, distinguished by corresponding staff class
plt.figure(dpi=300)

for stype in unique_types:
    idx   = [i for i, t in enumerate(staff_types_reduced) if t == stype]
    pts   = minimal_tensor[idx]
    plt.scatter(pts[:, 0], pts[:, 1],
                s=12, alpha=.5,
                label=f"obs {stype}", color=type_to_color[stype])

# Plot mu and sigma prior learned from data
mu_hat     = pyro.param("mu").detach()          # [n_classes, 2]
Sigma_hat  = pyro.param("Sigma").detach()       # [n_classes, 2, 2]

for i, cls in enumerate(unique_types):
    plt.scatter(mu_hat[i,0], mu_hat[i,1],
                marker='X', s=90, lw=1.5,
                color=type_to_color[cls],
                label='mu ' + cls)   

# Plot 95 % confidence ellipses
CHI2_95 = 5.991  # χ²_{2,0.95}

def add_cov_ellipse(ax, mean, cov, color, **kwargs):
    # eigen-decomposition
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]          # big → small
    vals, vecs = vals[order], vecs[:, order]

    # 95 % χ² quantile for 2 dof
    width, height = 2 * np.sqrt(vals * CHI2_95)
    angle = np.degrees(np.arctan2(vecs[1,0], vecs[0,0]))  # major-axis vector

    e = Ellipse(mean, width, height, angle,
                facecolor='none', edgecolor=color,
                linewidth=2, **kwargs)
    ax.add_patch(e)

ax = plt.gca()
for cls in range(n_types):
    add_cov_ellipse(ax,
                    mean=mu_hat[cls].numpy(),
                    cov =Sigma_hat[cls].numpy(),
                    color=type_to_color[index_to_type[cls]],
                    ls='--')

# Adjust some details
plt.xlabel('Levelling-rod offset [µm]')
plt.ylabel('Levelling-rod scale [ppm]')
plt.title('Offset & scale: observations  |  posterior μ̂  |  95 % ellipse')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
xmin, xmax = -100,  100    #  example numbers
ymin, ymax = -50,  50

ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
plt.tight_layout()
plt.show()



# vii) Plot posterior samples of parameters

# Plot raw observations, distinguished by corresponding staff class
plt.figure(dpi=300)

for stype in unique_types:
    idx = [i for i, t in enumerate(staff_types_reduced) if t == stype]
    pts = minimal_tensor[idx]
    plt.scatter(pts[:, 0], pts[:, 1],
                s=12, alpha=.5,
                label=f"obs {stype}", color=type_to_color[stype])

# Plot samples from the alpha posterior
alpha_samples = posterior_posttrain_dict["alpha"]      # [n_samp, n_ids, 2]

# choose how many samples per ID to plot (thin for readability)
NSHOW = 100
idx_show = torch.randperm(alpha_samples.shape[0])[:NSHOW]

alpha_thin = alpha_samples[idx_show]                   # [NSHOW, n_ids, 2]
alpha_thin = alpha_thin.reshape(-1, 2).numpy()         # [NSHOW*n_ids, 2]

# we need matching colours: repeat each id’s colour NSHOW times
# colours = []
# for sid in range(n_ids):
#     cls   = id_class_indices[sid].item()
#     c     = type_to_color[index_to_type[cls]]
#     colours.extend([c]*NSHOW)

plt.scatter(alpha_thin[:,0], alpha_thin[:,1],
            s=14, alpha=.15, marker='o',
            edgecolors='none',
            # c=colours,
            label='posterior α samples')

# Plotting adjustments
plt.xlabel('Levelling-rod offset [µm]')
plt.ylabel('Levelling-rod scale [ppm]')
plt.title('Observed data  |  learned class μ  |  posterior α samples')

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')

xmin, xmax = -100, 100
ymin, ymax = -50, 50
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)

plt.tight_layout()
plt.show()
