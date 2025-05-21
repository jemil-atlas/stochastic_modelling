#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The goal of this script is investigate and illustrate the excel dataset containing
levelling rod calibration results.
For this, do the following:
    1. Imports and definitions
    2. Preprocess data
    3. Plots and illustrations
"""

"""
    1. Imports and definitions
"""


# i) Imports

import pyro
import torch
import pickle
import matplotlib.pyplot as plt

# import data
with open("../data_stochastic_modelling/data_levelling_rod_calibration/data_list.pkl", "rb") as f:
    data_list = pickle.load(f)
with open("../data_stochastic_modelling/data_levelling_rod_calibration/minimal_list.pkl", "rb") as f:
    minimal_list = pickle.load(f)
minimal_tensor = torch.load("../data_stochastic_modelling/data_levelling_rod_calibration/minimal_tensor.pt", weights_only=True)

# ii) Definitions


"""
    2. Preprocess data
"""


# 1. Extract staff types
staff_types_reduced = [entry["staff_type_reduced"] for entry in minimal_list]

# 2. Get unique staff types and assign a color per type
unique_types = sorted(set(staff_types_reduced))
type_to_color = {stype: plt.cm.tab10(i % 10) for i, stype in enumerate(unique_types)}





"""
    3. Plots and illustrations
"""


# i) Plot each group separately

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