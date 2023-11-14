#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The goal of this script is to perform a fitting of gnss data for all the stations
for which there is a sufficient amount of data available.
For this, do the following:
    1. Definitions and imports
    2. Set up dataset
    3. Build model and guide
    4. Inference via svi
    5. Plots and illustrations
"""


"""
    1. Definitions and imports -----------------------------------------------
"""


# i) Imports

import numpy as np
import pyro
import torch
import pandas as pd
import matplotlib.pyplot as plt


# ii) Definitions

gnss_df = pd.read_excel("./Data/GNSS_2018_2019_all_stations.xlsx", sheet_name = None)
list_stations = list(gnss_df)
list_properties = list(gnss_df[list_stations[0]].keys())

pyro.clear_param_store()
torch.set_default_tensor_type(torch.DoubleTensor)


"""
    2. Set up dataset --------------------------------------------------------
"""


# i) Kick out NAN's

for sheet, df in gnss_df.items():
    gnss_df[sheet] = df.dropna().reset_index(drop=True)


# ii) Reshape into single datatensor

x_data = torch.empty([0, 4])
y_data = torch.empty([0, 3])
name_data = []
for sheet, df in gnss_df.items(): 
    # x_data
    time_stack = df['YYYY'].to_numpy() + (1/12)*df['MM'].to_numpy() + (1/(12*31))*df['DD'].to_numpy()
    ones_extension = np.ones(df['E[m]'].to_numpy().shape)
    x_np_stack = np.vstack((time_stack, np.mean(df['E[m]'].to_numpy())*ones_extension, 
                            np.mean(df['N[m]'].to_numpy())*ones_extension, 
                            np.mean(df['H[m]'].to_numpy()) * ones_extension)).T
    x_tensor_stack = torch.tensor(x_np_stack)
    x_data = torch.vstack((x_data, x_tensor_stack))
    
    # y_data
    y_np_stack = np.vstack((df['E[m]'].to_numpy(), df['N[m]'].to_numpy(), df['H[m]'].to_numpy())).T
    y_tensor_stack = torch.tensor(y_np_stack)
    y_data = torch.vstack((y_data, y_tensor_stack))
   
    # name_data
    for nr in list(ones_extension):
        name_data.append(sheet)
    
# Remove unnecessary formatting from data (i.e. Coordinates of Bern)
subtraction_mat = torch.tensor([[2010, 2600000, 1200000, 0]])    
subtraction_mat = subtraction_mat.repeat(x_data.shape[0], 1)
x_data = x_data - subtraction_mat
y_data = y_data - subtraction_mat[:,1:4]
    
# Make into single entry (batch shape = 1)
x_data = x_data.unsqueeze(dim = 0)
y_data = y_data.unsqueeze(dim = 0)

# test: only the first station
x_data = x_data[:,0:400,:]
y_data = y_data[:,0:400,:]

n_batch, n_time, n_dims = y_data.shape



"""
    3. Build model and guide -------------------------------------------------
"""

# i) Prepare model construction

def construct_mu_vec(mu, dim):
    mu_vec = mu*torch.ones(dim)
    return mu_vec

def construct_k_mat(x_data, sigma, d_T, d_E, d_N, d_H):
    x = x_data.squeeze()
    x_expanded = x[:, None, :]
    x_transposed = x[None, :, :]
    
    # Compute the difference matrices shape [n,n,4]
    diff_matrix = x_expanded - x_transposed
        
    k_mat = sigma*torch.exp(-((diff_matrix[:,:,0]/d_T)**2 
                        + (diff_matrix[:,:,1]/d_E)**2 
                        + (diff_matrix[:,:,2]/d_N)**2 
                        + (diff_matrix[:,:,3]/d_H)**2 ))
    return k_mat
    
    

# ii) Stochastic model

def gnss_model(x_data, observations = None):
    # auxiliary
    n_data = x_data.shape[1]
    
    # # constant mean mu
    # mu = pyro.param('mu', 2629058*torch.ones(1))
    # mu parametric function of t
    alpha = pyro.param('alpha', torch.zeros(5))
    list_g_funs = [lambda t : 1,\
                    lambda t : t,\
                    lambda t : t**2,\
                    lambda t : torch.sin(2*np.pi*t),\
                    lambda t : torch.cos(2*np.pi*t),\
                       ]
    n_g_funs = len(list_g_funs)
    g_mat = torch.zeros([n_data, n_g_funs])
    for k in range(n_g_funs):
        g_mat[:,k] = list_g_funs[k](x_data[0,:,0])
    mu_vec = x_data[0,:,1] + g_mat @ alpha
    
    # variance sigma
    # sigma = 1e-3*torch.ones(1) 
    sigma = pyro.param('sigma', 1e-3*torch.ones(1), constraint = pyro.distributions.constraints.interval(0,1e-2))
    # correlation lengths d of time and the 3 space dims
    d_T = pyro.param('d_T', torch.ones(1))
    d_E = pyro.param('d_E', torch.ones(1))
    d_N = pyro.param('d_N', torch.ones(1))
    d_H = pyro.param('d_H', torch.ones(1))

    # construct distribution
    # mu_vec = construct_mu_vec(mu, n_data)
    k_mat = construct_k_mat(x_data, sigma, d_T, d_E, d_N, d_H)
    cov_reg = 0.00001*torch.eye(mu_vec.shape[0])
    obs_dist = pyro.distributions.MultivariateNormal(loc = mu_vec, covariance_matrix = k_mat + cov_reg)
    
    obs = pyro.sample('obs', obs_dist, obs = observations)
    
    return obs
    

# iii) Guide

def guide(x_data, observations = None):
    pass


# iv) Prior simulations
n_simu = 5
y_pre_train = torch.zeros([n_simu, n_time, n_dims])
for k in range(n_simu):
    y_pre_train[k,:,0] = gnss_model(x_data)




"""
    4. Inference via svi -----------------------------------------------------
"""


# i) Setup SVI

optimizer = pyro.optim.Adam({"lr": 0.01})
svi = pyro.infer.SVI(model=gnss_model, guide=guide, optim=optimizer, loss=pyro.infer.Trace_ELBO())


# ii) Train the model

losses = []
num_epochs = 1000
y_data_squeezed = y_data[0,:,0]
for epoch in range(num_epochs):
    loss = svi.step(x_data, y_data_squeezed)
    losses.append(loss)
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")


# iv) Posterior simulations
n_simu = 5
y_post_train = torch.zeros([n_simu, n_time, n_dims])
for k in range(n_simu):
    y_post_train[k,:,0] = gnss_model(x_data)



"""
    5. Plots and illustrations -----------------------------------------------
"""


# Fetching inference results
for name, value in pyro.get_param_store().items():
    print(name, pyro.param(name).data.cpu().numpy())


# i) Plot pre train data
plt.figure(1, figsize = (5,5), dpi = 300)
plt.plot(y_data[0,:,0].detach(), color = 'r')
plt.plot(y_pre_train[:,:,0].detach().T, color = 'k')
plt.title("Pre train")


# ii) Plot post train data
plt.figure(2, figsize = (5,5), dpi = 300)
plt.plot(y_data[0,:,0].detach(), color = 'r')
plt.plot(y_post_train[:,:,0].detach().T, color = 'k')
plt.title("Post train")




















































