#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The goal of this script is to provide functionality for simulating data for the
levelling rod calibration. This script is built for modular simulation with multiple
different effects pre-built in pyro, which can then be chained for generating
data. This data is exported into excel format and can be imported and read later
on.
For this, do the following:
    1. Imports and definitions
    2. Base class construction
    3. Basic effects
    4. Specific effects
    5. Chaining for data generation
    6. Plots and illustrations
    7. Data export
    
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


# ii) Definitions

n_calibs = 100                  # nr of calibration measurements
n_staff_class_range = [5,10]    # range of nr of staff classes
n_staffs_range = [25, 75]       # range of nr of staffs that are measured
n_meas_range = [1, 5]           # range of nr of measurements per staff
n_edges_range = [50,100]        # range of nr of edges for rodtypes
n_dots_range = [100,200]        # range of nr of measurements on each edge

# torch.random.seed(42)
pyro.set_rng_seed(1)


# iii) Metadistribution for staff class alpha

meta_mu_mu = torch.zeros([2])
meta_mu_Sigma = torch.eye(2)
meta_mu_dist = pyro.distributions.MultivariateNormal(loc = meta_mu_mu, 
                                        covariance_matrix = meta_mu_Sigma)
meta_Sigma_dist = pyro.distributions.Wishart(df = 5,covariance_matrix = torch.eye(2))


# iii) Calibration and rod defaults

Sigma_cal = 100*torch.eye(2)
default_len = 2

"""
    2. Base class construction
"""

# i) Class for rods

len_dict = {f"Type {letter}": default_len
            for letter in string.ascii_uppercase[:n_staff_class_range[1]]}


class LevellingRodType():
    def __init__(self, name, n_edges = None):
        self.name = name
        self.n_edge = (n_edges if n_edges is not None else 
            torch.randint(low = n_edges_range[0], high = n_edges_range[1], size = (1,)))
        self.len_staff = len_dict[self.name]
        self.mu_alpha_class = 50 * pyro.sample('mu_alpha_class_' + self.name, meta_mu_dist)
        self.Sigma_alpha_class = 10 * pyro.sample('Sigma_alpha_class_' + self.name, meta_Sigma_dist)
        self.alpha_dist = pyro.distributions.MultivariateNormal(loc = self.mu_alpha_class,
                                                covariance_matrix = self.Sigma_alpha_class)
        
    def __repr__(self):
        repr_str = 'Levelling rod type' + self.name
        return repr_str
        

class LevellingRod():
    def __init__(self, rod_name, rod_type, n_meas):
        self.name = rod_name
        self.type = rod_type
        self.n_edge = rod_type.n_edge.item()
        self.n_meas = n_meas.item()
        self.tilt_types = torch.randint(0,3, [self.n_meas])
        self.len_staff = self.type.len_staff
        self.alpha_staff = pyro.sample('alpha_rod_' + self.name, self.type.alpha_dist)
        
        
    def __repr__(self):
        repr_str = 'Levelling rod ' + self.name + ' of type ' + self.type.name
        return repr_str


# ii) Class for environmental effects



# iii) Class for data

class DataList(list):
    def __init__(self, datapoint_list):
        self.datapoint_list = datapoint_list
        
    def get_sublist(self, key_list):
        # build new DataList containing only entries corresponding to keys in key_list
        subdatapoint_list = [dp.get_subdatapoint(key_list) for dp in self.datapoint_list]
        sub_datalist = DataList(subdatapoint_list)
        return sub_datalist

class DataPoint(dict):
    def __init__(self, data_dict):
        self.data = data_dict
    
    def get_subdatapoint(self, key_list):
        # build new DataPoint containing only entries corresponding to keys in key_list
        subdata_dict = {key: value for key, value in self.data.items() if key in key_list}
        sub_datapoint = DataPoint(subdata_dict)
        return sub_datapoint
        


# iv) Meta params for data generation




"""
    3. Basic effects
"""

# i) Noise addition

def add_noise(base_tensor, sigma_noise, obs = None):
    # Plate setup
    plate_sizes = base_tensor.shape
    n_plates = len(plate_sizes)
    plate_names = ['plate_{}'.format(k) for k in range(n_plates)]
    plate_stack = [pyro.plate(plate_names[k], dim = (- n_plates + k)) for k in range(n_plates)]
    
    # Dist setup
    noisy_dist = pyro.distributions.Normal(loc = base_tensor, scale = sigma_noise)
    
    # Sample from the distribution
    with contextlib.ExitStack() as stack:
        for plate in plate_stack:
            stack.enter_context(plate)

        noisy_sample = pyro.sample('noisy_sample', noisy_dist, obs = obs)

        return noisy_sample
    
    
# ii) Sample offset and scale

def measure_alpha(staff_list, n_meas_list):
    # Plate setup
    
    # Dist setup
    
    # Sample from distribution
    pass

# iii) Impact of tilting

def add_tilt_effect(tilt_type_list, n_meas, n_meas_max, n_edge_rod_k, n_edge_max):
    # assume 3 different tilt types:
    #   0. no impact
    #   1. positive curvature
    #   2. negative curvature
    
    x = (1/n_edge_rod_k)*torch.arange(0, n_edge_rod_k)
    
    tilt_series = torch.zeros(n_meas_max, n_edge_max)
    # tilt_series[n_meas : n_meas_max] = torch.nan
    for k, tilt_type in enumerate(tilt_type_list):
        if tilt_type == 0:
            pass
        elif tilt_type == 1:
            tilt_series[k, 0:n_edge_rod_k] = 50* x**2
        elif tilt_type == 2:
            tilt_series[k, 0:n_edge_rod_k] = -50* x**2
    
    return tilt_series
    


# iv) Generate edge series

def generate_edgeseries(staff_list, observations = None):
    # Info compilation
    n_meas_list = [staff.n_meas for staff in staff_list]
    n_edge_list = [staff.n_edge for staff in staff_list]
    n_staff = len(n_meas_list)
    
    # Extract some lengths adn setup masks
    n_meas_max = max(staff.n_meas for staff in staff_list)
    n_edge_max = max(staff.n_edge for staff in staff_list)
    
    mu_edge   = torch.zeros(n_staff, n_meas_max, n_edge_max)   # dummy, fill in below
    mask_edge = torch.zeros(n_staff, n_meas_max, n_edge_max, dtype=torch.bool)
    
    # Plate setup
    meas_plate = pyro.plate('meas_plate', size = n_meas_max, dim = -2)
    edge_plate = pyro.plate('edge_plate', size = n_edge_max, dim = -1)
    
    # Dist setup
    # Sample from distribution
    # rod plate : inside are effects different for each rod
    # meas_plate_rod :inside are effects differet for each measurement of a rod
    # edge_plate_rod_edge :inside are effects differet for each edge of a measurement of a rod
    
    obs_dict = {}
    for rod_k in pyro.plate('rod_plate', size = n_staff):
        # different_rods, different alpha
        staff = staff_list[rod_k]
        alpha = staff.alpha_staff
        n_meas_rod_k = staff.n_meas
        n_edge_rod_k = staff.n_edge
        
        # Following is x values of staff extended till max amount of edges for
        # vectorizability but true till n_edge_rod_k
        x_staff_rod_k = staff.len_staff * (1/n_edge_rod_k)*torch.arange(0, n_edge_max)
        tilt_types = staff.tilt_types
        
        with meas_plate:
            # different measurement, different tilt -> different 0
            tilt_effect = add_tilt_effect(tilt_types, n_meas_rod_k, n_meas_max, n_edge_rod_k, n_edge_max)
            
            with edge_plate:
                # different edge, different impact of alpha
                mu_edge =  alpha[0] + alpha[1]*x_staff_rod_k
                sigma_edge =  10*torch.eye(1)
                
                # Extendo to proper shape [n_meas_max, n_edge_max] pre-masking
                extension_tensor = torch.ones([n_meas_max, 1])
                mu_extended = extension_tensor * mu_edge.unsqueeze(0) + tilt_effect
                
                # mask construction: mask_tensor of shape [n_meas_max, n_edge_max]
                #    dim edge_plate True till n_edge_rod_k
                #    dim meas_plate True till n_meas_rod_k
                mask_tensor = torch.zeros([n_meas_max, n_edge_max])
                mask_tensor[0:n_meas_rod_k,0:n_edge_rod_k] = 1
                mask_tensor = mask_tensor.bool()
                
                # Build masked distribution
                edge_dist = (pyro.distributions.Normal(loc = mu_extended,scale = sigma_edge)
                             .mask(mask_tensor))
                edge_obs = pyro.sample('edge_obs_r{}'.format(rod_k), edge_dist, obs = observations)
                masked_obs = edge_obs.masked_fill(~mask_tensor, float("nan"))
        
        obs_dict[staff.name] = masked_obs
    
    return obs_dict
    
    

"""
    4. Specific effects
"""





"""
    5. Chaining for data generation
"""


# i) Build the model

# Set up staff classes
n_staff_class = torch.randint(low = n_staff_class_range[0], high = n_staff_class_range[1], size = (1,))
staff_class_names = [f"Type {letter}" for letter in string.ascii_uppercase[:n_staff_class]]
staff_classes_list = [LevellingRodType(name) for name in staff_class_names]

# Set up individual staffs
n_staff = torch.randint(low = n_staffs_range[0], high = n_staffs_range[1], size = (1,))
staff_names = ['{}'.format(k) for k in range(n_staff)]
random_class_index = torch.randint(low = 0, high = n_staff_class, size = (n_staff,))
random_staff_class_list = [staff_classes_list[random_class_index[k]] for k in range(n_staff)]
n_meas_list = [torch.randint(low = n_meas_range[0], high = n_meas_range[1], size = (1,)) for k in range(n_staff)]
staff_list = [LevellingRod(rod_name = staff_names[k], rod_type = random_staff_class_list[k],
                           n_meas = n_meas_list[k]) for k in range(n_staff)]

# ii) Build the model

def model(input_vars = None, obs = None):
    
    # Set up measurements 
    obs_dict = generate_edgeseries(staff_list)
    
    return obs_dict
    

# iii) Build the guide

def guide(input_vars = None, obs = None):
    pass


# vi) Illustrate model and guide

graphical_model = pyro.render_model(model = model, model_args= (None, None),
                                    render_distributions=True,
                                    render_params=True)
graphical_guide = pyro.render_model(model = guide, model_args= (None, None),
                                    render_distributions=True,
                                    render_params=True)

graphical_model
graphical_guide



# v) Use Model to generate data

data = model()


"""
    6. Plots and illustrations
"""


# i) Plot some edge datasets

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 3), sharex=True)

# Make subplots a bit tighter side-by-side
plt.subplots_adjust(wspace=0.25)

axes[0].plot(data['0'].T, lw=1.5)
axes[0].set_title("Edge measurements of some rod")
axes[0].set_ylabel("value")
axes[0].set_xlabel("edge nr")

axes[1].plot(data['1'].T, lw=1.5)
axes[1].set_title("Edge measurements of some rod")
axes[1].set_ylabel("value")
axes[1].set_xlabel("edge nr")

axes[2].plot(data['2'].T, lw=1.5)
axes[2].set_title("Edge measurements of some rod")
axes[2].set_ylabel("value")
axes[2].set_xlabel("edge nr")

# Optional overall title
fig.suptitle("Three examples of edge measurements", y=1.05, fontsize=14)

plt.show()


# ii) Plot rod alphas
rod_alpha_list = [staff.alpha_staff for staff in staff_list]
class_mu_list = [rod_class.mu_alpha_class for rod_class in staff_classes_list]
mus = torch.vstack(class_mu_list)
class_Sigma_list = [rod_class.Sigma_alpha_class for rod_class in staff_classes_list]
Sigmas = torch.vstack([sigma.unsqueeze(0) for sigma in class_Sigma_list])

def plot_gaussian_ellipses(mus, Sigmas, *, conf=0.95, ax=None,
                           facecolor="none", edgecolor="tab:blue", alpha=1.0):
    """
    Plot 2-D Gaussian means and confidence ellipses.
    """
    mus     = np.asarray(mus)
    Sigmas  = np.asarray(Sigmas)
    if mus.ndim != 2 or mus.shape[1] != 2:
        raise ValueError("mus must be of shape (N, 2)")
    if Sigmas.shape != mus.shape[:1] + (2, 2):
        raise ValueError("Sigmas must be of shape (N, 2, 2)")

    # χ² quantile for given confidence and 2 degrees of freedom
    chi2_val = chi2.ppf(conf, df=2)

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5), dpi = 300)
    ax.set_aspect("equal")

    # scatter plot of the means
    ax.scatter(mus[:, 0], mus[:, 1], color=edgecolor, zorder=3, label="means")

    # add one ellipse per Gaussian
    for mu, Sigma in zip(mus, Sigmas):
        # eigen-decomposition: Σ = Q Λ Qᵀ
        eigvals, eigvecs = np.linalg.eigh(Sigma)
        # major/minor axis lengths (sqrt of eigenvalues scaled by the quantile)
        width, height = 2.0 * np.sqrt(eigvals * chi2_val)
        # angle of the major axis (in degrees)
        angle = np.degrees(np.arctan2(*eigvecs[:, 1][::-1]))
        ell = Ellipse(xy=mu, width=width, height=height, angle=angle,
                      facecolor=facecolor, edgecolor=edgecolor,
                      alpha=alpha, lw=1.5)
        ax.add_patch(ell)

    ax.set_xlabel("$offset$")
    ax.set_ylabel("$scale$")
    # ax.legend(frameon=False)
    fig.suptitle("Rod classes and corresponding distributions", y=1.05, fontsize=14)
    return ax

plot_gaussian_ellipses(mus, Sigmas)


"""
    7. Data export
"""


# i) Convert to list of dicts
# Dict per datapoint should look similar to the following minimal dict that is
# exported from Roberts excel file.
# {'job_nr': '382',
#   'staff_type_reduced': 'l3m',
#   'staff_id': 61,
#   'overall_offset': -13.4319763017537,
#   'overall_scale': 1.41403091591386},
# however, we export here the ground truth offset and scale and instead as uncertain
# measurement data the edgeseries.
  
job_nr = 0
job_nr_list = []
staff_id = 0
staff_id_list = []

data_list = []
for staff in staff_list:
    for meas in range(staff.n_meas):
        # Compile info

        staff_type_reduced = staff.type.name
        staff_len = staff.len_staff
        overall_offset = staff.alpha_staff[0]
        overall_scale = staff.alpha_staff[1]
        edgeseries = data[str(staff_id)][meas,:]
        tilt_type = staff.tilt_types[meas]
        
        job_nr += 1
        job_nr_list.append(str(job_nr))
        
        # Save in data_list
        datapoint_dict_temp = {'job_nr' : job_nr,
                               'staff_id' : staff_id,
                               'staff_type_reduced' : staff_type_reduced,
                               'staff_len' : staff_len,
                               'gt_offset' : overall_offset,
                               'gt_scale' : overall_scale,
                               'edgeseries' : edgeseries,
                               'tilt_type' : tilt_type}
        data_list.append(datapoint_dict_temp)
    staff_id += 1
    staff_id_list.append(staff_id)


# ii) Set up dirs

out_dir = Path("../data_stochastic_modelling/data_levelling_rod_calibration")
out_dir.mkdir(parents=True, exist_ok=True)

with (out_dir / "synthetic_data_list_edgeseries.pkl").open("wb") as f:
    pickle.dump(data_list, f)


print(f"✨ Saved {len(data_list)} clean rows → {out_dir}")

