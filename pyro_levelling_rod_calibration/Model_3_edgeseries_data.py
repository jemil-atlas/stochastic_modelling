#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The goal of this script is to perform levelling rod calibration with edgeseries
data. For this, we import synthetically generated data and use it to train a 
probabilistic model that involves latent per-staff variables and a generative 
process metadist -> perstaff latent -> edgeseries.
For this, do the following:
    1. Imports and definitions
    2. Support functions
    3. Build Model
    4. Build guide
    5. Perform inference
    6. Plots and illustration
    
Since we work with synthetic data for which the ground truth alpha is known, we
can evaluate the success of this model by measuring how close we come to the
true parameters and latents.

Written by Dr. Jemil Avers Butt, Atlas optimization GmbH, www.atlasoptimization.com.
"""


"""
    1. Imports and definitions
"""


# i) Imports

import string
import pyro
import torch
import pickle
import pandas as pd
import seaborn as sns
import numpy as np
import contextlib
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import chi2 
import arviz as az
from pathlib import Path


# ii) Import and format data

with open("../data_stochastic_modelling/data_levelling_rod_calibration/synthetic_data_list_edgeseries.pkl", "rb") as f:
    data_list = pickle.load(f)

# Sort data_list according to staff_ids
data_list = sorted(data_list, key=lambda d: d['staff_id'])

# make input vars into a data list that contains metainfo on each datapoint

def dropped_copy(list_of_dicts, key_list):
    keys = set(key_list)
    return [
        {k: v for k, v in d.items() if k not in keys}
        for d in list_of_dicts
    ]

gt_alpha = [torch.hstack((datapoint['gt_offset'], datapoint['gt_scale'])) 
            for datapoint in data_list]
input_vars = dropped_copy(data_list, ['gt_offset', 'gt_scale'])

observations = torch.vstack([datapoint['edgeseries'].unsqueeze(0) for datapoint in data_list])
n_obs = len(data_list)


# iii) Metadistribution for staff class alpha




"""
    2. Support functions
"""


# i) Recompile info along job nr
job_nr_perjob = [input_var['job_nr'] for input_var in input_vars]
staff_id_perjob = [input_var['staff_id'] for input_var in input_vars]
staff_type_perjob = [input_var['staff_type_reduced'] for input_var in input_vars]
tilt_type_perjob = [input_var['tilt_type'] for input_var in input_vars]
edgeseries_perjob = [input_var['edgeseries'] for input_var in input_vars]
n_edge_perjob = [(~torch.isnan(input_var['edgeseries'])).sum() for input_var in input_vars]



# ii) Recompile info along rod ids
staff_list = list(set([input_var['staff_id'] for input_var in input_vars]))
n_meas_perstaff = [ sum(1 for iv in input_vars if iv['staff_id'] == sid)
    for sid in staff_list]
staff_type_perstaff = [ [iv['staff_type_reduced'] for iv in input_vars if iv['staff_id'] == sid][0]
    for sid in staff_list]
n_edge_perstaff = [ [torch.sum(~torch.isnan(iv['edgeseries'])) for iv in input_vars if iv['staff_id'] == sid][0]
    for sid in staff_list]
staff_len_perstaff = [ [iv['staff_len'] for iv in input_vars if iv['staff_id'] == sid][0]
    for sid in staff_list]
tilt_type_perstaff = [ [iv['tilt_type'] for iv in input_vars if iv['staff_id'] == sid]
    for sid in staff_list]
job_nr_perstaff = [ [iv['job_nr'] for iv in input_vars if iv['staff_id'] == sid]
    for sid in staff_list]

# Derived quantities
n_staff = len(staff_list)
n_types = len(set(staff_type_perstaff))
tilt_types = torch.unique(torch.stack(tilt_type_perjob))
n_tilt_types = len(tilt_types)

# Extract some lengths adn setup masks
n_meas_max = max(n_meas_perstaff)
n_edge_max = max(n_edge_perjob)



# iii) Get unique staff types and build obs -> class index tensor
unique_types = sorted(set(staff_type_perstaff))
type_to_color = {stype: plt.cm.tab10(i % 10) for i, stype in enumerate(unique_types)}
type_to_index = {stype: i for i, stype in enumerate(unique_types)}
index_to_type = {i: stype for i, stype in enumerate(unique_types)}
obs_class_indices = torch.tensor([type_to_index[staff_type] for staff_type in staff_type_perjob])

# iv) Get unique rod ids and build obs -> id index tensor
unique_ids = sorted(set(staff_id_perjob))
obs_id_indices = torch.tensor(staff_id_perjob)
id_class_indices_dict = {}               # {staff_id: class_idx}
for row in input_vars:            
    sid  = row["staff_id"]
    cidx = type_to_index[row["staff_type_reduced"]]
    id_class_indices_dict[sid] = cidx  
id_class_indices_dict = dict(sorted(id_class_indices_dict.items()))

# Build id -> class indextensor
n_ids = max(id_class_indices_dict) +1       # assuming staff_id starts at 0 and is dense
id_class_indices = torch.empty(n_ids, dtype=torch.long)
for sid, cidx in id_class_indices_dict.items():
    id_class_indices[sid] = cidx
    
# Build obs indx -> job_nr
obsnr_to_jobnr_dict = {}
jobnr_to_obsnr_dict = {}
for i in range(n_obs):
    obsnr_to_jobnr_dict[i] = job_nr_perjob[i]
    jobnr_to_obsnr_dict[job_nr_perjob[i]] = i
    
obsnr_perstaff = [[jobnr_to_obsnr_dict[jobnr] for jobnr in jobnr_list] for jobnr_list in job_nr_perstaff]
# obsnr_to_index = {}
# for k in range(n_obs):
#     obsnr = k
#     jobnr = obsnr_to_jobnr_dict[k]
#     staff_id = staff_id_perjob[k]
#     obsnr_to_index[k] = staff_id
obs_idx_lookup = [None] * n_obs

# Fill in the inverse map
for staff_idx, obs_list in enumerate(obsnr_perstaff):
    for obs_idx_in_staff, obs in enumerate(obs_list):
        obs_idx_lookup[obs] = [staff_idx, obs_idx_in_staff]
        
        
# Build an extended tensor including nans
observations_extended = torch.full((n_staff, n_meas_max, n_edge_max), float('nan'),
                                   dtype=observations.dtype,
                                   device=observations.device)
for obs_idx, (staff_idx, meas_idx) in enumerate(obs_idx_lookup):
    observations_extended[staff_idx, meas_idx, :] = observations[obs_idx]
observations_extended = observations_extended.reshape([-1, n_edge_max])


"""
    3. Build Model
"""


# i) Tilt model - ANN

class ANN(torch.nn.Module):
    # Will putput a different edge series based on input tilt type.
    def __init__(self):
        # Initialize instance using init method from base class
        super().__init__()
                
        # Linear layers
        self.lin_1 = torch.nn.Linear(1,16)
        self.lin_2 = torch.nn.Linear(16,16)
        self.lin_3 = torch.nn.Linear(16,1)
        # nonlinear transforms
        self.nonlinear = torch.nn.Tanh()
        
        self.models = torch.nn.ModuleList([torch.nn.Sequential(
            self.lin_1, self.nonlinear, self.lin_2, self.nonlinear, self.lin_3)
            for k in range(n_tilt_types)])
        
    def forward(self, tilt_types, x):
        # Reshape to account for batch shape
        x = x.reshape([-1, n_edge_max])
        tilt_types = tilt_types.reshape([-1])

        # Choose model based on tilt type and apply
        n_obs, n_edge = x.shape
        nonlinear_drift = torch.zeros([n_obs, n_edge])
        for tt in range(n_tilt_types):
            # Find observations of this tilt type
            mask = (tilt_types == tt)
            
            # Shape x inputs and pass
            x_masked = x[mask,:].reshape([-1,1])      
            nonlinear_drift[mask,:] = self.models[tt](x_masked).reshape([-1, n_edge_max])
            
        return nonlinear_drift

tilt_ann = ANN()

def add_tilt_effect(tilt_type_list, n_meas, n_meas_max, n_edge_rod_k, n_edge_max):
    # ANN nonlinear effect dependent on tilt type
    x = (1/n_edge_rod_k)*torch.arange(0, n_edge_max)
    tilt_series = torch.zeros(n_meas_max, n_edge_max)
    
    for k, tilt_type in enumerate(tilt_type_list):
        tilt_series[k,:] = tilt_ann(tilt_type, x)
    
    return tilt_series

# i) Chain together the effects
# This will lead to a probabilistic model of the following type
# 
# mu_alpha = unknown_param      [n_rod, 2]
# sigma_alpha = unknown param   [n_rod, 2,2]
# alpha ~ N(mu_alpha, sigma_alpha)  [n_rod,2]
# tilt_effect = f(measurement)      [n_rod, n_meas]
# mu_edge = alpha[0] + alpha[1]*x + tilt_effect     [n_rod, n_meas, n_edge]
# sigma_edge = unknown_param    [1]
# edge_obs ~ N(mu_edge, sigma_edge) [n_rod, n_meas, n_edge]
#
# i.e. in words: For each class, there exist some production distribution with
# unknown params. Each rod is sampled from one of these production distributions.
# Each rod was sampled a variable number n_meas of times; different conditions
# lead to different means for the observed edge positions. In the end, we get
# an edge series for each measurement of each rod.

def model(input_vars, observations = None):
    # Mark the parameters inside of the ann for optimization
    pyro.module("tilt_ann", tilt_ann)
    
    # mu_edge   = torch.zeros(n_staff, n_meas_max, n_edge_max)   # dummy, fill in below
    # mask_edge = torch.zeros(n_staff, n_meas_max, n_edge_max, dtype=torch.bool)
    
    # Plate setup
    # type_plate = pyro.plate('type_plate', size = n_types)
    # rod_plate = pyro.plate('rod_plate', size = n_staff)
    meas_plate = pyro.plate('meas_plate', size = n_meas_max, dim = -2)
    edge_plate = pyro.plate('edge_plate', size = n_edge_max, dim = -1)

    # General params
    sigma_cal = pyro.param("sigma_cal", init_tensor = 100 * torch.eye(1),
                   constraint = pyro.distributions.constraints.positive)
 

    # Staff setup
    # Different production mean and cov params per staff type
    mu_alpha = pyro.param('mu_alpha_prod', init_tensor = torch.zeros([n_types,2]))
    Sigma_alpha = pyro.param('Sigma_alpha_prod', init_tensor = 10000 * (torch.eye(2).unsqueeze(0)).expand([n_types,2,2]),
                       constraint = pyro.distributions.constraints.positive_definite)

    # Different latent alpha per staff id
    
    mu_alpha_extended =  mu_alpha[id_class_indices,:]
    Sigma_alpha_extended = Sigma_alpha[id_class_indices,:,:]
    alpha_dist = pyro.distributions.MultivariateNormal(loc = mu_alpha_extended,
                                                       covariance_matrix = Sigma_alpha_extended)
    with pyro.plate('rod_plate_vect', size = n_staff):
        alpha = pyro.sample('alpha_rods', alpha_dist)
        
    
    # Observations
    obs_dict = {}
    for rod_k in pyro.plate('rod_plate', size = n_staff):
        rod_alpha = alpha[rod_k,:]
        n_meas_rod_k = n_meas_perstaff[rod_k]
        n_edge_rod_k = n_edge_perstaff[rod_k]
        
        indices_obs_rod_k = torch.tensor(obsnr_perstaff[rod_k])
        x_staff_rod_k = staff_len_perstaff[rod_k] * (1/n_edge_rod_k)*torch.arange(0, n_edge_max)
        
        with meas_plate:
            # print(rod_k)
            tilt_effect = add_tilt_effect(tilt_type_perstaff[rod_k], n_meas_rod_k,
                                          n_meas_max, n_edge_rod_k, n_edge_max)
            
            with edge_plate:
                # different edge, different impact of alpha
                mu_edge =  rod_alpha[0] + rod_alpha[1]*x_staff_rod_k
                
                # Extend to proper shape [n_meas_max, n_edge_max] pre-masking
                extension_tensor = torch.ones([n_meas_max, 1])
                mu_extended = extension_tensor * mu_edge.unsqueeze(0) + tilt_effect
                
                # mask construction: mask_tensor of shape [n_meas_max, n_edge_max]
                #    dim edge_plate True till n_edge_rod_k
                #    dim meas_plate True till n_meas_rod_k
                mask_tensor = torch.zeros([n_meas_max, n_edge_max])
                mask_tensor[0:n_meas_rod_k,0:n_edge_rod_k] = 1
                mask_tensor = mask_tensor.bool()
                
                # Set up obs
                # obs_or_none = observations[indices_obs_rod_k] if observations is not None else None
                obs_full = torch.zeros((n_meas_max, n_edge_max))
                if observations is not None:
                    obs_raw = observations[indices_obs_rod_k, :n_edge_rod_k]  # shape [n_meas_rod_k, n_edge_rod_k]
                    obs_full[:n_meas_rod_k, :n_edge_rod_k] = obs_raw
                    obs_or_none = obs_full
                else:
                    obs_or_none = None
                    
                # Build masked distribution
                edge_dist = (pyro.distributions.Normal(loc = mu_extended,scale = sigma_cal)
                             .mask(mask_tensor))
                edge_obs = pyro.sample('edge_obs_r{}'.format(rod_k), edge_dist, obs = obs_or_none)
                masked_obs = edge_obs.masked_fill(~mask_tensor, float("nan"))
        
        obs_dict[rod_k] = masked_obs
    
    observation_list = []
    for k in range(n_obs):
        rod_k, idx = obs_idx_lookup[k]
        observation_list.append(obs_dict[rod_k][idx])
    
    return obs_dict, observation_list



"""
    4. Build guide
"""


# i) Build the guide

def guide(input_vars, observations = None):
    # Guide contains posterior distributions for the unobserved latents, i.e.
    # the alphas for each rod
    
    # Set up posterior parameters
    mu_alpha_post = pyro.param("mu_alpha_post", init_tensor = 10*torch.ones([n_staff, 2]))
    Sigma_alpha_post = pyro.param("Sigma_alpha_post", init_tensor = 10000 * (torch.eye(2).unsqueeze(0)).expand([n_staff,2,2]),
                       constraint = pyro.distributions.constraints.positive_definite)  
    
    # Sample from posterior distribution
    alpha_post_dist = pyro.distributions.MultivariateNormal(loc = mu_alpha_post,
                                            covariance_matrix = Sigma_alpha_post)
    with pyro.plate("rod_plate", size = n_staff):
        alpha_post_sample = pyro.sample('alpha_rods', alpha_post_dist)
    return alpha_post_sample
    
    
    
    
    

# iii) illustrate model and guide

graphical_model = pyro.render_model(model = model, model_args= (input_vars,),
                                    render_distributions=True,
                                    render_params=True)
graphical_guide = pyro.render_model(model = guide, model_args= (input_vars,),
                                    render_distributions=True,
                                    render_params=True)

graphical_model
graphical_guide

# iv) Record example outputs of model and guide prior to training

n_model_samples = 2
n_guide_samples = 100

predictive = pyro.infer.Predictive
prior_predictive_pretrain_dict = predictive(model, num_samples = n_model_samples)(input_vars)
posterior_pretrain_dict = predictive(guide, num_samples = n_guide_samples)(input_vars)
posterior_predictive_pretrain_dict = predictive(model, guide = guide, num_samples = n_model_samples)(input_vars)


# Build tensor from sequential edge obs
def build_tensor_from_dict(obs_dict):
    keys = [key for key in obs_dict.keys() if 'obs' in key]
    data_tensor =  torch.zeros([n_model_samples, n_meas_max*n_staff, n_edge_max])
    k=0
    for key in keys:
        data_tensor[:, k*n_meas_max : (k+1)*n_meas_max] = obs_dict[key]
        k = k + 1
    return data_tensor


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
    5. Perform inference
"""


# i) Set up inference

adam = pyro.optim.Adam({"lr": 0.1})
elbo = pyro.infer.Trace_ELBO(num_particles = 5,
                                 max_plate_nesting = 2)
svi = pyro.infer.SVI(model, guide, adam, elbo)


# ii) Perform svi

data = (input_vars, observations)
loss_sequence = []
for step in range(10):
    loss = svi.step(*data)
    loss_sequence.append(loss)
    if step %10 == 0:
        print(f'epoch: {step} ; loss : {loss}')
    
    
# iii) Record example outputs of model and guide post training

prior_predictive_posttrain_dict = predictive(model, num_samples = n_model_samples)(input_vars)
posterior_posttrain_dict = predictive(guide, num_samples = n_guide_samples)(input_vars)
posterior_predictive_posttrain_dict = predictive(model, guide = guide, num_samples = n_model_samples)(input_vars)

for name, value in pyro.get_param_store().items():
    print(name, value)


"""
    6. Plots and illustration
"""




# i) Plot loss

plt.figure(1, dpi = 300)
plt.plot(loss_sequence)
plt.yscale("log")
plt.title('ELBO loss during training (log scale)')
plt.xlabel('Epoch nr')
plt.ylabel('value')


# ii) Plot each group separately

# Some renamings for plotting
staff_types_reduced = staff_type_perjob
# Build minimal tensor of fitted alphas

def estimate_alpha(data):
    batched = data.ndim == 3
    if not batched:
        data = data.unsqueeze(0)  # → [1, n_obs, n_edge_max]
    alpha_tensor = torch.zeros(tuple(data.shape[:-1]) + (2,))
    
    for k in range(n_obs):
        rod_k =  staff_id_perjob[k]
        n_meas_rod_k = n_meas_perstaff[rod_k]
        n_edge_rod_k = n_edge_perstaff[rod_k]
            
        indices_obs_rod_k = torch.tensor(obsnr_perstaff[rod_k])
        x_rod_k = staff_len_perstaff[rod_k] * (1/n_edge_rod_k)*torch.arange(0, n_edge_rod_k)
        A_k = torch.vstack((torch.ones(n_edge_rod_k), x_rod_k)).T
        pinv_A = torch.linalg.pinv(A_k.T @ A_k) @ A_k.T  # shape [2, n_edge_rod_k]

        obs_k = data[:, k, :n_edge_rod_k]  # shape [n_samples, n_edge_rod_k]
        alpha_k = torch.einsum('ij,sj->si', pinv_A, obs_k)  # [n_samples, 2]

        alpha_tensor[:, k, :] = alpha_k
        
        # alpha_k = torch.linalg.pinv(A_k.T@A_k)@A_k.T@observations[k,:n_edge_rod_k]
        # alpha_tensor[k,:] = alpha_k
    return alpha_tensor

minimal_tensor = estimate_alpha(observations)


plt.figure(dpi=300)

for stype in unique_types:
    indices = [i for i, t in enumerate(staff_types_reduced) if t == stype]
    points = minimal_tensor[0,indices]
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

offsets = minimal_tensor[:,:, 0].numpy()
scales = minimal_tensor[:,:, 1].numpy()
x_min = min(offsets.flatten()) - 10
x_max = max(offsets.flatten()) + 10
y_min = min(scales.flatten()) - 10
y_max = max(scales.flatten()) + 10

# Set up figure with 3 columns
fig, axs = plt.subplots(1, 3, figsize=(15, 5), dpi=300)

# Plot 1: Scatterplot
for stype in unique_types:
    indices = [i for i, t in enumerate(staff_types_reduced) if t == stype]
    points = minimal_tensor[0,indices]
    axs[0].scatter(points[:, 0], points[:, 1], label=stype, color=type_to_color[stype])

# 4. Labels and legend
axs[0].set_xlabel('Levelling rod offset [µm]')
axs[0].set_ylabel('Levelling rod scale [ppm]')
axs[0].set_title('Offset and scale of rods by staff type')
axs[0].legend(title='Staff Type', bbox_to_anchor=(1.05, 1), loc='upper left')


# Plot 2: 2D KDE (Pretraining)
prior_predictive_pretrain_tensor = build_tensor_from_dict(prior_predictive_pretrain_dict)
prior_predictive_pretrain_tensor_alpha = estimate_alpha(prior_predictive_pretrain_tensor[0,:,:])
prior_obs_dicts = build_dicts_from_data(prior_predictive_pretrain_tensor_alpha)
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
prior_predictive_posttrain_tensor = build_tensor_from_dict(prior_predictive_posttrain_dict)
prior_predictive_posttrain_tensor_alpha = estimate_alpha(prior_predictive_posttrain_tensor[0,:,:])
prior_obs_dicts = build_dicts_from_data(prior_predictive_posttrain_tensor_alpha)
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
cls_labels_extended = [label for label in staff_type_perstaff for _ in range(n_meas_max)]#staff_type_perstaff*n_meas_max

prior_predictive_pretrain_tensor = build_tensor_from_dict(prior_predictive_pretrain_dict)
prior_predictive_posttrain_tensor = build_tensor_from_dict(prior_predictive_posttrain_dict)
prior_predictive_pretrain_tensor_alpha = estimate_alpha(prior_predictive_pretrain_tensor)
prior_predictive_posttrain_tensor_alpha = estimate_alpha(prior_predictive_posttrain_tensor)
df_pretrain  = to_df(prior_predictive_pretrain_tensor_alpha,  cls_labels_extended, "pretrain")
df_posttrain   = to_df(prior_predictive_pretrain_tensor_alpha, cls_labels_extended, "posttrain")
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

alpha_samples = posterior_posttrain_dict["alpha_rods"]    # [n_samp, n_ids, 2]
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

alpha_mean = posterior_posttrain_dict["alpha_rods"].mean(0)   # [n_ids,2]
mu_learned = pyro.param("mu_alpha_prod").detach()

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
    pts   = minimal_tensor[:, idx,:]
    plt.scatter(pts[:, :, 0], pts[:,:, 1],
                s=12, alpha=.5,
                label=f"obs {stype}", color=type_to_color[stype])

# Plot mu and sigma prior learned from data
mu_hat     = pyro.param("mu_alpha_prod").detach()          # [n_classes, 2]
Sigma_hat  = pyro.param("Sigma_alpha_prod").detach()       # [n_classes, 2, 2]

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

plt.figure(dpi=300)
# Plot samples from the alpha posterior
alpha_samples = posterior_posttrain_dict["alpha_rods"]      # [n_samp, n_ids, 2]

# choose how many samples per ID to plot (thin for readability)
NSHOW = 100
idx_show = torch.randperm(alpha_samples.shape[0])[:NSHOW]

alpha_thin = alpha_samples[idx_show]                   # [NSHOW, n_ids, 2]
alpha_thin = alpha_thin.reshape(-1, 2).numpy()         # [NSHOW*n_ids, 2]

# we need matching colours: repeat each id’s colour NSHOW times
colours = []
for sid in range(n_ids):
    cls   = id_class_indices[sid].item()
    c     = type_to_color[index_to_type[cls]]
    colours.extend([c]*NSHOW)

plt.scatter(alpha_thin[:,0], alpha_thin[:,1],
            s=20, alpha=.1, marker='o',
            edgecolors='none',
            # c=colours,        # Something not right with color classes
            label='posterior α samples')

# Plot raw observations, distinguished by corresponding staff class
for stype in unique_types:
    idx   = [i for i, t in enumerate(staff_types_reduced) if t == stype]
    pts   = minimal_tensor[:, idx,:]
    plt.scatter(pts[:, :, 0], pts[:,:, 1],
                s=12, alpha=.5,
                label=f"obs {stype}", color=type_to_color[stype])
    
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


Sigma_cal_pyro = pyro.get_param_store()['sigma_cal']
print('Estimated edge uncertainty : ', Sigma_cal_pyro)
# print('Sigma offset calibration: ', torch.sqrt(Sigma_cal_pyro[0,0]) )
# print('Sigma scale calibration: ', torch.sqrt(Sigma_cal_pyro[1,1]) )


# viii) Plot calibration covariance

# plt.figure(dpi = 300)
# plt.imshow(Sigma_cal_pyro.detach().numpy())
# plt.title('Error covariance matrix calibration')

# ix) Plot edge observations per class for data, pretrain model, posttrain model

# Build posterior predictive distributions
posterior_predictive_pretrain_tensor = build_tensor_from_dict(posterior_predictive_pretrain_dict)
posterior_predictive_posttrain_tensor = build_tensor_from_dict(posterior_predictive_posttrain_dict)

# mask the nans
n_obs_extended  = len(cls_labels_extended)
rod_k_extended_obs = [staff_id for staff_id in staff_list for _ in range(n_meas_max)]
posterior_predictive_pretrain_tensor_nan = torch.full(posterior_predictive_pretrain_tensor.shape, float("nan"))
posterior_predictive_posttrain_tensor_nan = torch.full(posterior_predictive_posttrain_tensor.shape, float("nan"))
for k in range(n_obs_extended):
    rod_k = rod_k_extended_obs[k]
    n_edge_rod_k = n_edge_perstaff[rod_k]
    
    indices_obs_rod_k = torch.tensor(obsnr_perstaff[rod_k])
    x_staff_rod_k = staff_len_perstaff[rod_k] * (1/n_edge_rod_k)*torch.arange(0, n_edge_max)
    
    posterior_predictive_pretrain_tensor_nan[:, k, :n_edge_rod_k] = posterior_predictive_pretrain_tensor[:, k, :n_edge_rod_k]
    posterior_predictive_posttrain_tensor_nan[:, k, :n_edge_rod_k] = posterior_predictive_posttrain_tensor[:, k, :n_edge_rod_k]

def plot_data_model_grid(cls_labels_extended, observations,
                         predictive_pre, predictive_post,
                         n_types_to_plot=None, max_lines_per_cell=10):

    n_obs, n_edge = observations.shape
    n_samples = predictive_post.shape[0]

    # Unique rod types
    unique_classes = list(set(cls_labels_extended))
    if n_types_to_plot is not None:
        unique_classes = unique_classes[:n_types_to_plot]
    n_types = len(unique_classes)

    fig, axs = plt.subplots(3, n_types, figsize=(4 * n_types, 8), sharey=True, sharex=True)
    fig.suptitle("Edge observations from data and model", fontsize=16)
    if n_types == 1:
        axs = axs[:, None]  # make it 2D if only one type

    for col, rod_type in enumerate(unique_classes):
        idxs = [i for i, lbl in enumerate(cls_labels_extended) if lbl == rod_type][:max_lines_per_cell]

        # row 0: raw data
        for i in idxs:
            axs[0, col].plot(observations_extended[i].cpu(), color='black', alpha=0.6)

        # row 1: pretrain predictive mean
        pred_mean_pre = predictive_pre[:, idxs, :].mean(dim=0)
        for i in range(len(idxs)):
            axs[1, col].plot(pred_mean_pre[i].cpu(), color='blue', alpha=0.6)

        # row 2: posttrain predictive mean
        pred_mean_post = predictive_post[:, idxs, :].mean(dim=0)
        for i in range(len(idxs)):
            axs[2, col].plot(pred_mean_post[i].cpu(), color='green', alpha=0.6)

        axs[0, col].set_title(f"Rod Type {rod_type}")


    axs[0, 0].set_ylabel("Data")
    axs[1, 0].set_ylabel("Pretrain")
    axs[2, 0].set_ylabel("Posttrain")

    for ax in axs.flatten():
        ax.grid(True)

    plt.tight_layout()
    plt.show()
    
plot_data_model_grid(cls_labels_extended,
                 observations,
                 posterior_predictive_pretrain_tensor,
                 posterior_predictive_posttrain_tensor,
                 n_types_to_plot=5, max_lines_per_cell = 20)

    
    

# x) Plot nonlinear trends for different tilt classes

def plot_tilt_model_grid(tilt_labels_extended, observations,
                         predictive_pre, predictive_post,
                         n_types_to_plot=None, max_lines_per_cell=10,
                         suptitle=None):
    """
    Plots a 3-row grid:
        Row 1: Observations
        Row 2: Pretrain predictive mean
        Row 3: Posttrain predictive mean

    Grouped by unique tilt types.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Ensure tilt_labels_extended is a numpy array for fast comparison
    tilt_labels_extended = np.array(tilt_labels_extended)
    n_obs, n_edge = observations.shape
    n_samples = predictive_post.shape[0]

    unique_tilts = list(np.unique(tilt_labels_extended))
    if n_types_to_plot is not None:
        unique_tilts = unique_tilts[:n_types_to_plot]
    n_tilts = len(unique_tilts)

    fig, axs = plt.subplots(3, n_tilts, figsize=(4 * n_tilts, 8), sharey=True, sharex=True)

    if n_tilts == 1:
        axs = axs[:, None]

    for col, tilt_type in enumerate(unique_tilts):
        mask = (tilt_labels_extended == tilt_type)
        idxs = np.where(mask)[0][:max_lines_per_cell]
        idxs_torch = torch.tensor(idxs, device=observations.device)

        # Row 0: observations
        for i in idxs:
            axs[0, col].plot(observations[i].cpu(), color='black', alpha=0.6)

        # Row 1: pretrain mean
        pred_mean_pre = predictive_pre[:, idxs_torch, :].mean(dim=0)
        for i in range(len(idxs)):
            axs[1, col].plot(pred_mean_pre[i].cpu(), color='blue', alpha=0.6)

        # Row 2: posttrain mean
        pred_mean_post = predictive_post[:, idxs_torch, :].mean(dim=0)
        for i in range(len(idxs)):
            axs[2, col].plot(pred_mean_post[i].cpu(), color='green', alpha=0.6)

        axs[0, col].set_title(f"Tilt Type {tilt_type}")

    axs[0, 0].set_ylabel("Data")
    axs[1, 0].set_ylabel("Pretrain")
    axs[2, 0].set_ylabel("Posttrain")

    for ax in axs.flatten():
        ax.grid(True)

    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=16)

    plt.tight_layout()
    plt.show()

plot_tilt_model_grid(tilt_labels_extended,
                     observations,
                     predictive_pre_valid,
                     predictive_post_valid,
                     suptitle="Nonlinear Trends by Tilt Type")
