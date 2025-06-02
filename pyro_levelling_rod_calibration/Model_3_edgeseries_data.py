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
    # meas_plate = pyro.plate('meas_plate', size = n_meas_max, dim = -2)
    # edge_plate = pyro.plate('edge_plate', size = n_edge_max, dim = -1)

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
    with pyro.plate('rod_plate', size = n_staff):
        alpha = pyro.sample('alpha_rods', alpha_dist)
        
    
    # Observations
    obs_dict = {}
    for rod_k in pyro.plate('rod_plate', size = n_staff):
        rod_alpha = alpha[rod_k,:]
        n_meas_rod_k = n_meas_perstaff[rod_k]
        n_edge_rod_k = n_edge_perstaff[rod_k]
        
        x_staff_rod_k = staff_len_perstaff[rod_k] * (1/n_edge_rod_k)*torch.arange(0, n_edge_max)
        
        with pyro.plate('meas_plate', size = n_meas_max, dim = -2):
            # print(rod_k)
            tilt_effect = add_tilt_effect(tilt_type_perstaff[rod_k], n_meas_rod_k,
                                          n_meas_max, n_edge_rod_k, n_edge_max)
            
            with pyro.plate('edge_plate', size = n_edge_max, dim = -1):
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
                
                # Build masked distribution
                edge_dist = (pyro.distributions.Normal(loc = mu_extended,scale = sigma_cal)
                             .mask(mask_tensor))
                obs_or_none = observations[] if observations is not None else None
                edge_obs = pyro.sample('edge_obs_r{}'.format(rod_k), edge_dist, obs = obs_or_none)
                masked_obs = edge_obs.masked_fill(~mask_tensor, float("nan"))
        
        obs_dict[rod_k] = masked_obs
    
    observation_list = []
    for k in range(n_obs):
        observation_list.append()
    
    return obs_dict



"""
    4. Build guide
"""



# ii) Illustrate model and guide pre-training
aa = model(input_vars)


"""
    5. Perform inference
"""




"""
    6. Plots and illustration
"""