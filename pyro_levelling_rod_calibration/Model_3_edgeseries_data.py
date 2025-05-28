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


# ii) Import data
with open("../data_stochastic_modelling/data_levelling_rod_calibration/synthetic_data_list_edgeseries.pkl", "rb") as f:
    data_list = pickle.load(f)




# iii) Metadistribution for staff class alpha




"""
    2. Support functions
"""



"""
    3. Build Model
"""

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

"""
    4. Build guide
"""




"""
    5. Perform inference
"""




"""
    6. Plots and illustration
"""