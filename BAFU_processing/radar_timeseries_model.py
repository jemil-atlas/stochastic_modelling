#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The goal of this script is to train a stochastic model for TRI timeseries of
individual pixels. Learning variances and correlations lengths employs pyro's
svi formalism and real radar data from breithorn. The stochastic model is made up
primarily of a model for the covariance matrix of APS and noise that depends on
the coherence and time of day. Optimization yields parameters that establish a 
relationship between coherence, time of day, noise level, and correlation length.
For this, do the following:
    1. Imports and definitions
    2. Build auxiliary datasets
    3. Build stochastic model
    4. Inference
    5. Plots and illustrations
    
"""


"""
    1. Imports and definitions ------------------------------------------------
"""

# i) imports

import numpy as np
import pyro
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter
import copy
from scipy.io import loadmat


# ii) Load data

torch.set_default_dtype(torch.float64)
data = loadmat('./Data/submatrix_collection_1000pixels_2023_2days_mli_timeseries_no_def.mat')
data_struct = data['submatrix_collection']
complex_vals_data = data_struct[0][0][0]
info_mats_data = data_struct[0][0][1]

# reshape into 24h chunks
n_day = 719
index_set = [100, 200] # 
complex_vals_data = complex_vals_data[index_set[0]:index_set[1],:]
complex_vals_data = complex_vals_data.reshape([-1, n_day])
abs_vals_data = np.abs(complex_vals_data)
phase_vals_data = np.angle(complex_vals_data)
info_mats_data = info_mats_data[index_set[0]:index_set[1],:,:].reshape([-1,n_day,9])

# Subsample a portion of data (-> toy data for experimentation)
n_start = 0
n_end = n_day
n_skip = 2
n_step = np.floor((n_end - n_start)/n_skip).astype(int)
subsample_indices = np.linspace(n_start, n_end-1, n_step).astype(int) 
# n_cutoff = 100
complex_vals_data = complex_vals_data[:,subsample_indices]
abs_vals_data = abs_vals_data[:,subsample_indices]
phase_vals_data = phase_vals_data[:,subsample_indices]
info_mats_data = info_mats_data[:,subsample_indices]



# iii) Definitions

n_t = complex_vals_data.shape[1]
n_samples = complex_vals_data.shape[0]
n_illu = np.min([n_samples, 20])

t = np.linspace(0,24,n_t)

pyro.clear_param_store()
writer = SummaryWriter('./Tensorboard/Experiment')



"""
    2. Build auxiliary datasets -----------------------------------------------
"""


# i) Compile datasets from loaded data

# The sequence is range, az, x, y, z, coh, aoi, mean, time
range_mats = torch.tensor(info_mats_data[:,:,0]).double()
azimuth_mats = torch.tensor(info_mats_data[:,:,1]).double()
x_mats = torch.tensor(info_mats_data[:,:,2]).double()
y_mats = torch.tensor(info_mats_data[:,:,3]).double()
z_mats = torch.tensor(info_mats_data[:,:,4]).double()
coherence_mats = torch.tensor(info_mats_data[:,:,5]).double()
aoi_mats = torch.tensor(info_mats_data[:,:,6]).double()
meanphase_mats = torch.tensor(info_mats_data[:,:,7]).double()
time_mats = torch.tensor(info_mats_data[:,:,8]).double()

phase_mats = torch.tensor(phase_vals_data[:,:]).double()




"""
    3. Build stochastic model -------------------------------------------------
"""
    

# ii) Dataset preparation

# basic inputs to stochastic model
class BaseData():
    def __init__(self, range_mats, azimuth_mats, x_mats, y_mats, z_mats, coherence_mats, aoi_mats, meanphase_mats, time_mats):
        self.range_mats = range_mats
        self.azimuth_mats = azimuth_mats
        self.x_mats = x_mats
        self.y_mats = y_mats
        self.z_mats = z_mats
        self.coherence_mats = coherence_mats
        self.aoi_mats = aoi_mats
        self.meanphase_mats = meanphase_mats
        self.time_mats = time_mats
                
        def get_basic_attr_list(self):
            attributes = [attr for attr in dir(self) if not (attr.startswith('__'))]
            return attributes
        self.basic_attribute_list = get_basic_attr_list(self)
        self.num_basic_attributes = len(self.basic_attribute_list)
        
base_data = BaseData(range_mats, azimuth_mats,  x_mats, y_mats, z_mats, coherence_mats, aoi_mats, meanphase_mats, time_mats)


# full outputs of stochastic model
class FullData():
    def __init__(self, base_data, phase_mats, **optional_input):
        # load nr of samples
        self.n_samples = optional_input.get('n_samples', None)
        # load the variance scaling matrices
        self.A_mats = optional_input.get('A_mats', None)
        self.B_mats = optional_input.get('B_mats', None)
        # load the covariance matrices
        self.K_noise_mats = optional_input.get('K_noise_mats', None)
        self.K_smooth_mats = optional_input.get('K_smooth_mats', None)
        self.cov_mats_noise = optional_input.get('cov_mats_noise', None)
        if self.cov_mats_noise is not None:
                noise_variances = np.diagonal(self.cov_mats_noise.detach(), axis1=1, axis2=2)
                self.noise_variance_mats = noise_variances.reshape((self.n_samples, n_t))
        self.cov_mats_smooth = optional_input.get('cov_mats_smooth', None)
        if self.cov_mats_smooth is not None:
                smooth_variances = np.diagonal(self.cov_mats_smooth.detach(), axis1=1, axis2=2)
                self.smooth_variance_mats = smooth_variances.reshape((self.n_samples, n_t))
        # List distributional_data
        def get_distributional_attr_list(self):
            attributes = [attr for attr in dir(self) if not (attr.startswith('__') )]
            return attributes
        self.distributional_attribute_list = get_distributional_attr_list(self)
        self.num_distributional_attributes = len(self.distributional_attribute_list)
        # copy from base_data
        for name, attribute in base_data.__dict__.items():
            setattr(self, name, attribute)
        # Add simulations
        self.phase_mats = phase_mats            

full_data = FullData(base_data, phase_mats)

    
    
"""
    4. Build stochastic model -------------------------------------------------
"""

# i) Auxiliary function definitions

# Map into high dim space
class ANNMap(pyro.nn.PyroModule):
    # Initialize the module
    def __init__(self, dim_input, dim_hidden, dim_output):
        # Evoke by passing input, hidden, and output dimensions
        # Initialize instance using init method from base class
        super().__init__()
        self.dim_input = dim_input
        
        # linear transforms
        self.fc_1 = torch.nn.Linear(dim_input, dim_hidden)
        self.fc_2 = torch.nn.Linear(dim_hidden, dim_hidden)
        self.fc_3 = torch.nn.Linear(dim_hidden, dim_output)
        # nonlinear transforms
        self.nonlinear = torch.nn.Sigmoid()
        
    def forward(self, x, atypical_shape = False):
        # Define forward computation on the input data x
        if atypical_shape == False :
            # Shape the minibatch so that batch_dims are on left, argument_dims on right
            x = x.reshape([-1, n_t, self.dim_input])
        else:
            # Shape the minibatch for processing of a vector of inputs
            x = x.reshape([-1, x.shape[1], self.dim_input])
        
        # Then compute hidden units and output of nonlinear pass
        hidden_units_1 = self.nonlinear(self.fc_1(x))
        hidden_units_2 = self.nonlinear(self.fc_2(hidden_units_1))
        ann_output = self.fc_3(hidden_units_2)
        return ann_output



# ii) Covariance function definition

def construct_cov_mats(ann_map, coh_mats, t_mats):
        
    # fuse to input and compute coefficients
    input_mats = torch.cat((coh_mats.unsqueeze(2), t_mats.unsqueeze(2)), dim=2)
    feature_mats = ann_map(input_mats)
    n_rows, n_cols, n_output = feature_mats.shape
    A_mats = torch.zeros([n_rows, n_cols, n_cols])
    B_mats = torch.zeros([n_rows, n_cols, n_cols])
    for k in range(n_rows):
        A_mats[k] = torch.diag(feature_mats[k,:,0])
        B_mats[k] = torch.diag(feature_mats[k,:,1])
    
    # compute basis matrices
    K_noise_mats = torch.eye(n_cols).repeat(n_rows,1,1)
    
    K_smooth_mats = torch.zeros([n_rows, n_cols, n_cols])
    corr_lengths = pyro.param('corr_lengths', init_tensor = 1*torch.ones([2]), constraint = pyro.distributions.constraints.positive)
    rel_scales = pyro.param('scales', init_tensor = 0.5*torch.ones([2]), constraint = pyro.distributions.constraints.simplex)
    def kernel_fun_sqexp(v_1,v_2):
        # Calculate distance matrix dist_mat
        v_1_reshaped = v_1.unsqueeze(1)  # shape becomes [n_total, 1]
        v_2_reshaped = v_2.unsqueeze(0)  # shape becomes [1, n_total]
        # corr_lengths_reshaped = corr_lengths.unsqueeze(0)
        dist_mat = torch.abs((v_1_reshaped - v_2_reshaped) / corr_lengths[0])
        # squared exponential kernel function
        cov_mat = torch.exp(-dist_mat**2)
        return cov_mat
    
    def kernel_fun_exp(v_1,v_2):
        # Calculate distance matrix dist_mat
        v_1_reshaped = v_1.unsqueeze(1)  # shape becomes [n_total, 1]
        v_2_reshaped = v_2.unsqueeze(0)  # shape becomes [1, n_total]
        # corr_lengths_reshaped = corr_lengths.unsqueeze(0)
        dist_mat = torch.abs((v_1_reshaped - v_2_reshaped) / corr_lengths[1])
        # exponential kernel function
        cov_mat = torch.exp(-dist_mat**1)
        return cov_mat
        
    for k in range(n_rows):
        K_smooth_mats[k,:,:] = rel_scales[0]*kernel_fun_sqexp(t_mats[k,:], t_mats[k,:]) \
                                + rel_scales[1]*kernel_fun_exp(t_mats[k,:], t_mats[k,:])
        
    return K_noise_mats, K_smooth_mats, A_mats, B_mats
        

# iii) Stochastics class
subsample_size = 64

class TRIStochastics(pyro.nn.PyroModule):
    # Initialize the module
    def __init__(self, list_inputs, dim_hidden, base_data):
        super().__init__()  # Initialize the PyroModule superclass (otherwise module registration fails)
        # list_inputs contains a list of names of attributes that are used as 
        # inputs to construct a covariance function. Names must be attributes of
        # base_data class.
        self.dim_input = 2
        self.dim_hidden = dim_hidden
        self.dim_output = 2
        
        # combine base_data inputs
        self.list_inputs = list_inputs
        self.integrate_base_data(self.list_inputs, base_data)                       
        self.base_data = base_data
        self.ann_map = ANNMap(self.dim_input, self.dim_hidden, self.dim_output)
        
    
    # Model
    def model(self, base_data, observations = None, subsample_size = subsample_size):
        # integrate different base_data
        self.integrate_base_data(self.list_inputs,base_data)       
        # print("Samples:{}".format(self.n_samples))      
        
        with pyro.plate('batch_plate',size = self.n_samples, dim = -1, subsample_size = subsample_size) as ind:
            K_noise_mats, K_smooth_mats, A_mats, B_mats = construct_cov_mats(self.ann_map, 
                                    self.base_data.coherence_mats[ind,:], self.base_data.time_mats[ind,:]) 
            cov_regularizer = 1e-6*(torch.eye(n_t).repeat(subsample_size, 1, 1))
            cov_mats_noise_temp = torch.bmm(A_mats, K_noise_mats)
            cov_mats_noise = torch.bmm(cov_mats_noise_temp, A_mats.transpose(1,2))
            cov_mats_smooth_temp = torch.bmm(B_mats, K_smooth_mats)
            cov_mats_smooth = torch.bmm(cov_mats_smooth_temp, B_mats.transpose(1,2))   
            subsampled_observations = observations[ind] if observations is not None else None
            obs_dist = pyro.distributions.MultivariateNormal(loc = torch.zeros([subsample_size, n_t]), covariance_matrix = cov_mats_smooth + cov_mats_noise + cov_regularizer)
            # print("batch_shape : {}, event_shape : {}".format(obs_dist.batch_shape, obs_dist.event_shape))
            obs = pyro.sample("obs", obs_dist, obs = subsampled_observations)
        # data = FullData(base_data, obs.reshape([self.n_samples, n_x,n_y]).detach(), K_phase_mats = cov_mats)
        data = FullData(base_data, obs.reshape([subsample_size, n_t]).detach(), 
                        **{'K_noise_mats' : K_noise_mats, 'K_smooth_mats' : K_smooth_mats, \
                           'cov_mats_noise' : cov_mats_noise, 'cov_mats_smooth' : cov_mats_smooth, \
                           'A_mats' : A_mats, 'B_mats' : B_mats, 'n_samples' : len(ind)})
        return obs, data
    
    def integrate_base_data(self, list_inputs, base_data):
        self.n_samples = base_data.range_mats.shape[0]
        self.list_base_data_mats = [getattr(base_data, input_attr).reshape([self.n_samples, n_t, -1]) for input_attr in list_inputs]
        self.base_data_mats = torch.cat(self.list_base_data_mats, dim = -1)
        self.dim_arg = self.base_data_mats.shape[-1]
        
    # Guide
    def guide(self, base_data, observations = None):
        pass

list_inputs = ['coherence_mats', 'time_mats' ]

tri_stochastics = TRIStochastics(list_inputs, 100, base_data)
observations = full_data.phase_mats.reshape([n_samples, -1])


# iv) Simulation pretraining

simulation_pretrain, full_data_pretrain = copy.copy(tri_stochastics.model(base_data, subsample_size = n_illu))
# simulation_pretrain = copy.copy(simulation_pretrain.reshape([subsample_size, n_t]))



"""
    5. Inference -------------------------------------------------------------
"""


# i) Set up training

# specifying scalar options
learning_rate = 1*1e-2
num_epochs = 100
adam_args = {"lr" : learning_rate}

# Setting up svi
optimizer = pyro.optim.AdamW(adam_args)
elbo_loss = pyro.infer.Trace_ELBO()
svi = pyro.infer.SVI(model = tri_stochastics.model, guide = tri_stochastics.guide, optim = optimizer, loss= elbo_loss)


# ii) Execute training

train_elbo = []
for epoch in range(num_epochs):
    epoch_loss = svi.step(base_data, observations)
    train_elbo.append(-epoch_loss)
    if epoch % 10 == 0:
        # log the data on tensorboard
        print("Epoch : {} train loss : {}".format(epoch, epoch_loss))
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        # for name, param in pyro.get_param_store():
        #     writer.add_histogram(name, param, epoch)
        # writer.add_scalar('Some_Pyro_Parameter', pyro.param("some_param").item(), epoch)

# Howto tensorboard:
# tensorboard --logdir=Tensorboard
# go to http://localhost:6006/
writer.close()

# iii) Simulation posttraining

simulation_posttrain, full_data_posttrain = copy.copy(tri_stochastics.model(base_data, subsample_size = n_illu))
# simulation_posttrain = copy.copy(simulation_posttrain.reshape([subsample_size, n_t]))

    
for param_name, param_value in pyro.get_param_store().items():
    if not param_name.startswith('ann'):
        print('{} : {}'.format(param_name, param_value))
    
    

"""
    6. Plots and illustrations -----------------------------------------------
"""


# i) Illustrate data

fig, axs = plt.subplots(3, 4, figsize=(15, 9))

attributes = [full_data.phase_mats, full_data.coherence_mats, full_data.z_mats, full_data.aoi_mats]
titles = ["Phase Mats", "Coherence Mats", "Elevation Mats", "AOI Mats"]

for i in range(np.min([subsample_size, 3])):  # loop over samples
    for j in range(4):  # loop over attributes
        ax = axs[i, j]
        attribute = attributes[j]
        ax.plot(attribute[i, :])
        if i == 0:
            ax.set_title(titles[j])

        ax.axis('off')

plt.tight_layout()
plt.show()


# ii) Illustrate the ann mapping function and training process
# Training
plt.figure(1,figsize = (5,5),dpi =300)
plt.plot(-np.array(train_elbo)[10:])
plt.xlabel('train epoch')
plt.ylabel('elbo')
plt.title('Training progress')


# Mapping
n_map = 100
coh_base = torch.linspace(0,1,n_map).unsqueeze(0).unsqueeze(2)
time_base = torch.linspace(0,24,n_map).unsqueeze(0).unsqueeze(2)
coh_tensor, time_tensor = torch.meshgrid((coh_base.flatten(), time_base.flatten()))
time_coh_input_sequential = torch.cat((coh_tensor.unsqueeze(2), time_tensor.unsqueeze(2)), dim=2)
ann_output = tri_stochastics.ann_map(time_coh_input_sequential, atypical_shape = True).squeeze()
noise_var = ann_output[:,:,0].detach()**2
smooth_var = ann_output[:,:,1].detach()**2

morning_index = np.round(n_map/3).astype(int)
midday_index = np.round(n_map/2).astype(int)
afternoon_index = np.round(2*n_map/3).astype(int)
halfway_index = np.round(n_map/2).astype(int)

fig, axs = plt.subplots(3, 2, figsize=(10, 15))
# Image plots
ax = axs[0,0]
ax.imshow(noise_var)
ax.set_ylabel('Coherence')
ax.set_xlabel('Time')
ax.set_title('Noise variance')

ax = axs[0,1]
ax.imshow(smooth_var)
ax.set_ylabel('Coherence')
ax.set_xlabel('Time')
ax.set_title('Smooth variance')

# Lineplots over time
ax = axs[1,0]
ax.plot(time_base.flatten().detach(), noise_var[0,:], label = 'low_coh')
ax.plot(time_base.flatten().detach(), noise_var[halfway_index,:], label = 'mid_coh')
ax.plot(time_base.flatten().detach(), noise_var[-1,:], label = 'high_coh')
ax.set_ylabel('Noise var')
ax.set_xlabel('Time')
ax.legend()
ax.set_title('Noise variance over time')

ax = axs[1,1]
ax.plot(time_base.flatten().detach(), smooth_var[0,:], label = 'low_coh')
ax.plot(time_base.flatten().detach(), smooth_var[halfway_index,:], label = 'mid_coh')
ax.plot(time_base.flatten().detach(), smooth_var[-1,:], label = 'high_coh')
ax.set_ylabel('Smooth var')
ax.set_xlabel('Time')
ax.legend()
ax.set_title('Smooth variance over time')

# Lineplots over coherence
ax = axs[2,0]
ax.plot(coh_base.flatten().detach(), noise_var[:, morning_index], label = 'morning')
ax.plot(coh_base.flatten().detach(), noise_var[:, midday_index], label = 'midday')
ax.plot(coh_base.flatten().detach(), noise_var[:, afternoon_index], label = 'afternoon')
ax.plot(coh_base.flatten().detach(), noise_var[:, -1], label = 'night')
ax.set_ylabel('Noise var')
ax.set_xlabel('Coherence')
ax.legend()
ax.set_title('Noise variance over coherence')

ax = axs[2,1]
ax.plot(coh_base.flatten().detach(), smooth_var[:, morning_index], label = 'morning')
ax.plot(coh_base.flatten().detach(), smooth_var[:, midday_index], label = 'midday')
ax.plot(coh_base.flatten().detach(), smooth_var[:, afternoon_index], label = 'afternoon')
ax.plot(coh_base.flatten().detach(), smooth_var[:, -1], label = 'night')
ax.set_ylabel('Smooth var')
ax.set_xlabel('Coherence')
ax.legend()
ax.set_title('Smooth variance over coherence')

plt.tight_layout()
plt.show()


# iii) Illustrate data, initial simulations, trained simulations

# Realizations
fig, axs = plt.subplots(5, 3, figsize=(9, 15))

attributes = [full_data.phase_mats, simulation_pretrain.detach(), simulation_posttrain.detach()]
titles = ["Phase: data", "pretrain", " posttrain"]
for i in range(np.min([subsample_size, 5])):  # loop over samples
    for j in range(3):  # loop over attributes
        ax = axs[i, j]
        attribute = attributes[j]
        ax.plot(attribute[i, :])
        if i == 0:
            ax.set_title(titles[j])
        ax.axis('off')
plt.tight_layout()
plt.show()


# iv) Illustrate the variance scaling
# Covariance matrices
# Showcase K_smooth, K_noise and cov_mat_smooth, cov_mat_noise
fig, axs = plt.subplots(3, 2, figsize=(10, 15), dpi = 300)
# Noise kernel
ax = axs[0, 0]
ax.imshow(full_data_pretrain.K_noise_mats[0,:,:].detach())
ax.set_title('Kernel mat noise ')
ax.axis('off')
# Smooth kernel
ax = axs[0, 1]
ax.imshow(full_data_pretrain.K_smooth_mats[0,:,:].detach())
ax.set_title('Kernel mat smooth ')
ax.axis('off')
# Noise cov pretrain
ax = axs[1, 0]
ax.imshow(full_data_pretrain.cov_mats_noise[0,:,:].detach())
ax.set_title('Cov mat noise pretrain')
ax.axis('off')
# Smooth cov pretrain
ax = axs[1, 1]
ax.imshow(full_data_pretrain.cov_mats_smooth[0,:,:].detach())
ax.set_title('Cov mat smooth pretrain')
ax.axis('off')

# Noise cov posttrain
ax = axs[2, 0]
ax.imshow(full_data_posttrain.cov_mats_noise[0,:,:].detach())
ax.set_title('Cov mat noise posttrain')
ax.axis('off')
# Smooth cov posttrain
ax = axs[2, 1]
ax.imshow(full_data_posttrain.cov_mats_smooth[0,:,:].detach())
ax.set_title('Cov mat smooth posttrain')
ax.axis('off')

plt.tight_layout()
plt.show()


# Correlations coherence variance scales
# Showcase how the entries of the matrices A and B with cov = AK_noiseA.T +BK_smoothB.T
# are related to the coherence values. Do so via line plots.
fig, axs = plt.subplots(2, 2, figsize=(10, 10), dpi = 300)

# A_mat pretrain
ax = axs[0, 0]
ax.plot(torch.abs(torch.diag(full_data_pretrain.A_mats[0,:,:].detach())), color = 'k')
ax_twin = ax.twinx()
ax_twin.plot(full_data_pretrain.coherence_mats[0,:].flatten(), color = 'r')
ax.set_title('ANN scale noise pretrain')
ax.set_ylabel('Scale noise', color='k')
ax.set_xlabel('Time (nr interferogram)', color='k')
ax_twin.set_ylabel('Coherence', color='r')
# ax.axis('off')
# ax_twin.axis('off')

# B_mat pretrain
ax = axs[0, 1]
ax.plot(torch.abs(torch.diag(full_data_pretrain.B_mats[0,:,:].detach())), color = 'k')
ax_twin = ax.twinx()
ax_twin.plot(full_data_pretrain.coherence_mats[0,:].flatten(), color = 'r')
ax.set_title('ANN scale smooth pretrain')
ax.set_ylabel('Scale smooth', color='k')
ax.set_xlabel('Time (nr interferogram)', color='k')
ax_twin.set_ylabel('Coherence', color='r')
# ax.axis('off')
# ax_twin.axis('off')

# A_mat posttrain
ax = axs[1, 0]
ax.plot(torch.abs(torch.diag(full_data_posttrain.A_mats[0,:,:].detach())), color = 'k')
ax_twin = ax.twinx()
ax_twin.plot(full_data_pretrain.coherence_mats[0,:].flatten(), color = 'r')
ax.set_title('ANN scale noise posttrain')
ax.set_ylabel('Scale noise', color='k')
ax.set_xlabel('Time (nr interferogram)', color='k')
ax_twin.set_ylabel('Coherence', color='r')
# ax.axis('off')
# ax_twin.axis('off')

# B_mat posttrain
ax = axs[1, 1]
ax.plot(torch.abs(torch.diag(full_data_posttrain.B_mats[0,:,:].detach())), color = 'k')
ax_twin = ax.twinx()
ax_twin.plot(full_data_pretrain.coherence_mats[0,:].flatten(), color = 'r')
ax.set_title('ANN scale smooth posttrain')
ax.set_ylabel('Scale smooth', color='k')
ax.set_xlabel('Time (nr interferogram)', color='k')
ax_twin.set_ylabel('Coherence', color='r')
# ax.axis('off')
# ax_twin.axis('off')

plt.tight_layout()
plt.show()


# Showcase the empirical and predicted distributions
fig, axs = plt.subplots(2, 3, figsize=(15, 10), dpi = 300)

# Line plots
# Find the global min and max across all matrices
observations_emp = observations[0:n_illu,:].T.detach().numpy()
observations_pretrain = full_data_pretrain.phase_mats.T.detach().numpy()
observations_posttrain = full_data_posttrain.phase_mats.T.detach().numpy()
ymin_data = np.min([observations_emp.flatten(), observations_pretrain.flatten(), observations_posttrain.flatten()])
ymax_data = np.max([observations_emp.flatten(), observations_pretrain.flatten(), observations_posttrain.flatten()])

# Lineplots empirical
ax = axs[0, 0]
ax.plot(observations_emp)
ax.set_title('Observations')
ax.set_ylim([ymin_data,ymax_data])
ax.axis('off')

# Lineplots predicted pretrain
ax = axs[0, 1]
ax.plot(observations_pretrain)
ax.set_title('Simulations pretrain')
ax.set_ylim([ymin_data,ymax_data])
ax.axis('off')

# Lineplots predicted posttrain
ax = axs[0, 2]
ax.plot(observations_posttrain)
ax.set_title('Simulations posttrain')
ax.set_ylim([ymin_data,ymax_data])
ax.axis('off')


# Cov mats
# Find the global min and max across all matrices
cov_mat_emp = (1/n_samples)*(phase_vals_data.T@phase_vals_data)
cov_mat_pretrain = (full_data_pretrain.cov_mats_noise[0,:,:] + full_data_pretrain.cov_mats_smooth[0,:,:]).detach().numpy()
cov_mat_posttrain = (full_data_posttrain.cov_mats_noise[0,:,:] + full_data_posttrain.cov_mats_smooth[0,:,:]).detach().numpy()
vmin = np.min([cov_mat_emp.flatten(), cov_mat_pretrain.flatten(), cov_mat_posttrain.flatten()])
vmax = np.max([cov_mat_emp.flatten(), cov_mat_pretrain.flatten(), cov_mat_posttrain.flatten()])

# Empirical cov mat
ax = axs[1, 0]
ax.imshow(np.corrcoef(phase_vals_data.T), vmin=vmin, vmax=vmax)
ax.set_title('Cov mat empirical')
ax.axis('off')

# Predicted cov mat pretrain
ax = axs[1, 1]
ax.imshow(cov_mat_pretrain, vmin=vmin, vmax=vmax)
ax.set_title('Cov mat pretrain')
ax.axis('off')

# Predicted cov mat posttrain
ax = axs[1, 2]
ax.imshow(cov_mat_posttrain, vmin=vmin, vmax=vmax)
ax.set_title('Cov mat posttrain')
ax.axis('off')

plt.tight_layout()
plt.show()



# v) Multiple realizations for same base_data

def plot_realizations(base_data, full_data, **plotting_opts):
    # This function plots the base data and a bunch of realizations. This is used
    # to compare the realizations of the trained model and the true underlying model.
    # plotting opts include the 'n_realizations', the number of realizations to
    # be shown during plotting and 'scenario_index' the index of the scenario
    # that is shown (i.e. determines the randomly chosen base_data like elevation etc)
    
    # Fetch basic & distributional data
    base_attributes = [getattr(base_data,attr) for attr in base_data.basic_attribute_list]
    base_attribute_names = [attr for attr in base_data.basic_attribute_list]
    n_base_data = base_data.num_basic_attributes
    
    # distributional_attributes = [getattr(full_data, attr) for attr in full_data.distributional_attribute_list]
    # distributional_attribute_names = [attr for attr in full_data.distributional_attribute_list]
    # n_distributional_data = full_data.num_distributional_attributes
    
    distributional_attributes_posttrain = [getattr(full_data_posttrain, attr) for attr in full_data_posttrain.distributional_attribute_list]
    distributional_attribute_names_posttrain = [attr for attr in full_data_posttrain.distributional_attribute_list]
    n_distributional_data_posttrain = full_data_posttrain.num_distributional_attributes
    
    n_realizations = plotting_opts.get('n_realizations', 4)
    scenario_index = plotting_opts.get('scenario_index', 0)

    # Simulate new data


    # Support function    
    def eval_attribute_dims(attribute, scenario_index):
        attribute_dim = (attribute[0,:]).shape
        n_attribute_dim = len(attribute_dim)
        if n_attribute_dim ==3:
            image = attribute[scenario_index,:,0]
            message = " 1 of {} matrices".format(attribute_dim[2])
        else:
            image = attribute[scenario_index,:]
            message = ""
        return image, message
    
    
    # Plot data    
    n_rows_plot = 3
    n_cols_plot = np.max([n_base_data, n_realizations])
    base_data_slice = BaseData(torch.repeat_interleave((range_mats[scenario_index,:]).unsqueeze(0), subsample_size, dim = 0),
                               torch.repeat_interleave((azimuth_mats[scenario_index,:]).unsqueeze(0), subsample_size, dim = 0),
                               torch.repeat_interleave((x_mats[scenario_index,:]).unsqueeze(0), subsample_size, dim = 0),
                               torch.repeat_interleave((y_mats[scenario_index,:]).unsqueeze(0), subsample_size, dim = 0),
                               torch.repeat_interleave((z_mats[scenario_index,:]).unsqueeze(0), subsample_size, dim = 0),
                               torch.repeat_interleave((coherence_mats[scenario_index,:]).unsqueeze(0), subsample_size, dim = 0),
                               torch.repeat_interleave((aoi_mats[scenario_index,:]).unsqueeze(0), subsample_size, dim = 0),
                               torch.repeat_interleave((meanphase_mats[scenario_index,:]).unsqueeze(0), subsample_size, dim = 0),
                               torch.repeat_interleave((time_mats[scenario_index,:]).unsqueeze(0), subsample_size, dim = 0)
                               )
    
    tri_images_posttrain = (tri_stochastics.model(base_data_slice)[0]).reshape([subsample_size, n_t])
    
    fig, axs = plt.subplots(n_rows_plot, n_cols_plot, figsize=(n_cols_plot*3, n_rows_plot*3))
    for i in range(n_rows_plot):  # loop over rows of plot
        if i == 0:
            for j in range(n_base_data):
                ax = axs[i, j]
                ax.axis('off')
                attribute = base_attributes[j]
                image, message = eval_attribute_dims(attribute, scenario_index)
                ax.plot(image)
                ax.set_title(base_attribute_names[j] + message)
        if i == 1:
            for j in range(n_realizations):
                ax = axs[i, j]
                ax.axis('off')
                image = phase_mats[j,:]
                ax.set_title("Realization nr {}".format(j))
                image_example = ax.plot(image)
        if i == 2:
            for j in range(n_realizations):
                ax = axs[i, j]
                ax.axis('off')
                image = tri_images_posttrain[j,:]
                ax.plot(image.detach())
                ax.set_title("Realization nr {}".format(j))                
        
    plt.axis('off')
    plt.tight_layout()
    fig.suptitle("Adapt realizations when more data available - right now realizations for different base data")
    plt.show()


plot_realizations(base_data, full_data, n_realizations = 8, scenario_index = 1)















































