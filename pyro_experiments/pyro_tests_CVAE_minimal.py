#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The goal of this script is to showcase a very basic conditional variational 
autoencoder that can be used to encode and decode a very simple model featuring
two variables x,y where y is a function of x and some noise. We also sample 
from conditional distributions.
For this, do the following:
    1. Definitions and imports
    2. Generate data
    3. Encoder and decoder class
    4. Model, Guide, assembling the CVAE
    5. Training via svi
    6. Plots and illustrations
"""


"""
    1. Definitions and imports
"""


# i) Imports

import numpy as np
import pyro
import torch
import pyro.distributions as dist
import matplotlib.pyplot as plt


# ii) Definitions

n_data = 500
dim_data_x = 1
dim_data_y = 1
dim_data = dim_data_x + dim_data_y
dim_z = 2
dim_hidden = 10
use_cuda = False

 

"""
    2. Generate data
"""

# i) Draw samples for x and y

# Simple, linear mean
mu_fun = lambda x : x
sigma_fun = lambda x : 0.1

# # Quadratic mean
# mu_fun = lambda x : x**2
# sigma_fun = lambda x : 0.1

# # Dynamic variances
# mu_fun = lambda x : 1
# sigma_fun = lambda x : np.abs(x)

# # Sinusoidal
# mu_fun = lambda x : np.sin(2*np.pi*x)
# sigma_fun = lambda x : 0.2*np.abs(np.sin(2*np.pi*x))

# # Bimodal <- this does not work with Gaussian priors. This is a known problem
# mu_fun = lambda x : np.random.randint(0,2, size = [x.shape[0],1])
# sigma_fun = lambda x : 0.1



x_data = torch.linspace(-1,1, n_data).reshape([-1,1]).float()
y_true = mu_fun(x_data)
sigma_true = sigma_fun(x_data)
y_data = torch.tensor(np.random.normal(y_true, sigma_true)).float()


# ii) Fuse data together and create new dataset

xy_data = torch.hstack((x_data,y_data)).float()
xy_tensor_data = torch.utils.data.TensorDataset(x_data, y_data)

# Split the dataset into train and test parts: 80% for training and 20% for testing.
train_size = int(0.8 * len(xy_tensor_data))
test_size = len(xy_tensor_data) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(xy_tensor_data, [train_size, test_size])

# Wrap the datasets into DataLoaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)




"""
    3. Encoder and decoder class, baseline_net
"""

# Note that we do not do any masking here, so x and y are both of shape [n_data,1]
# with xy being of shape [n_data,2]. 

# i) Encoder neural network

class Encoder(pyro.nn.PyroModule):
    # Initialize the module
    def __init__(self, dim_z, dim_hidden):
        # Evoke by passing bottleneck and hidden dimension
        # Initialize instance using init method from base class
        super().__init__()
        
        # linear transforms
        self.fc_1 = torch.nn.Linear(dim_data, dim_hidden)
        self.fc_21 = torch.nn.Linear(dim_hidden, dim_z)
        self.fc_22 = torch.nn.Linear(dim_hidden, dim_z)
        # nonlinear transforms
        self.nonlinear = torch.nn.Tanh()
        
    def forward(self, x, y):
        # Define forward computation on the input data x and output data y
        xy = torch.hstack((x,y))
        
        # Then compute hidden units and output of nonlinear pass
        hidden_units = self.nonlinear(self.fc_1(xy))
        z_loc = self.fc_21(hidden_units)
        z_scale = torch.exp(self.fc_22(hidden_units))
        return z_loc, z_scale


# ii) Decoder neural network

class Decoder(pyro.nn.PyroModule):
    # Initialize the module
    def __init__(self, dim_z, dim_hidden):
        # Initialize using init method from base class
        super().__init__()
        
        # linear transforms
        self.fc_1 = torch.nn.Linear(dim_z, dim_hidden)
        self.fc_2 = torch.nn.Linear(dim_hidden, dim_hidden)
        self.fc_3 = torch.nn.Linear(dim_hidden, dim_data)
        # nonlinear transforms
        self.nonlinear = torch.nn.Tanh()
    
    def forward(self, z):
        # Define forward computation on the latent codes z
        # Compute hidden units and output of nonlinear pass
        hidden = self.nonlinear(self.fc_1(z))
        hidden = self.nonlinear(self.fc_2(hidden))
        xy_guess = self.fc_3(hidden)
        return xy_guess        


# iii) Build baseline_net

class BaselineNet(pyro.nn.PyroModule):
    def __init__(self, dim_input, dim_hidden, dim_output):
        super().__init__()  # __init__ method of base class
        self.fc1 = torch.nn.Linear(dim_data_x, dim_hidden)
        self.fc2 = torch.nn.Linear(dim_hidden, dim_hidden)
        self.fc3 = torch.nn.Linear(dim_hidden, dim_data_y)
        self.nonlinear = torch.nn.Tanh()

    def forward(self, x):
        hidden = self.nonlinear(self.fc1(x))
        hidden = self.nonlinear(self.fc2(hidden))
        y = self.fc3(hidden)
        return y


# iv) Train baseline_net

baseline_net = BaselineNet(dim_data_x, dim_hidden, dim_data_y)
optimizer_baseline = torch.optim.Adam(baseline_net.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss()

epochs = 1000
for epoch in range(epochs):
    for i, batch in enumerate(train_loader):
        inputs = batch[0]
        outputs = batch[1]

        optimizer_baseline.zero_grad()
        preds = baseline_net(inputs)
        loss = criterion(preds, outputs)
        loss.backward()
        optimizer_baseline.step()

    if epoch % 100 == 0:
        print('Epoch {} baseline train loss : {}'.format(epoch, loss.item()))


# v) Illustrate baseline on scatterplots

# Create predictions
y_pred_bl = torch.zeros([n_data,1])
for k in range(n_data):
    y_pred_bl[k,:] = baseline_net(x_data[k,:]).detach()
    
# Create scatterplots
fig, axs = plt.subplots(1, figsize=(5, 5)) 
axs.scatter(x_data, y_data, label = 'data')
axs.plot(x_data, y_pred_bl, color = 'red', label = 'baseline prediction')
axs.set_title('Original data and baseline prediction')
axs.legend()

plt.tight_layout()
plt.show()



"""
    4. Model, Guide, assembling the VAE
"""


# i) Build CVAE class with model and guide included for later inference

class CVAE(pyro.nn.PyroModule):
    def __init__(self, dim_z = dim_z, dim_hidden = dim_hidden, use_cude = use_cuda):
        super().__init__()      # __init__ method of base class
        # Initialize the prior_net, generation_net, and recognition_net that
        # perform the tasks of modulating prior p_theta(z|x), determining the
        # generative process p_theta(x|x,y) and encode a datapoint into latent
        # space via a variational distribution q_phi(z|x,y)
        self.prior_net = Encoder(dim_z, dim_hidden)
        self.generation_net = Decoder(dim_z, dim_hidden)
        self.recognition_net = Encoder(dim_z, dim_hidden)
        self.baseline_net = baseline_net
        
        self.dim_z = dim_z
        self.dim_hidden = dim_hidden     
        
        
    # ii) Define model - is built by the decoder as the decoder ties x to z via the
    # likelihood p_theta(x|z)p(z)
    
    def model(self, x, y = None):
        # register the ANN parameters with pyro
        pyro.module("generation_net", self.generation_net)
        pyro.module("prior_net", self.prior_net)
        batch_size = x.shape[0]
        
        # Now construct the x via sampling statements of a batch of z
        # First modulate the distribution of z based on x via the prior_net
        y_hat = self.baseline_net(x)        
        z_loc, z_scale = self.prior_net(x, y_hat)
        # Then sample the z's
        # z_dist.batch_shape = batch_size, z_dist.event_shape = dim_z 
        z_dist = dist.Normal(z_loc, z_scale).to_event(1)    
        with pyro.plate('batch_plate', size = batch_size, dim = -1):     
            # Broadcasting aligns to the left; leftmost dimension of samples from
            # z_dist are declared independent. Then pass z to generation_net.
            z = pyro.sample('latent_z', z_dist)
            xy_guess = self.generation_net(z)   
            
            # Pass y as an observation that ties the y segment of the xy_guess 
            # to some actual data
            y_guess = xy_guess[:,1].view(batch_size,-1) 
            dist_y = dist.Normal(y_guess, 0.01).to_event(1)
            pyro.sample('output_y', dist_y, obs = y)

        return xy_guess


    # iii) Define guide - is built by the recognition_net as the recognition_net
    # ties z to x,y via the the probability q_phi(z|x,y)
    
    def guide(self, x, y = None):
        # register the recognition_net parameters with pyro
        pyro.module("recognition_net", self.recognition_net)
        batch_size = x.shape[0]
        # construct z via sampling statements of a batch of x, y
        # use recognition_net to get distributional parameters
         
        if y is not None:
            # When training, pass x, y to recognition net to generate z and
            # declare q_phi(z| x,y) as variational distribution
            z_loc, z_scale = self.recognition_net(x,y)
            z_dist = dist.Normal(z_loc, z_scale).to_event(1)
        else:
            # When testing, pass x and a dummy y to the recognition_net to
            # generate a sample of z
            y_hat = self.baseline_net(x)  
            z_loc, z_scale = self.recognition_net(x,y_hat)
            z_dist = dist.Normal(z_loc, z_scale).to_event(1)
        
        # Sample z from the distribution
        with pyro.plate('batch_plate', size = batch_size, dim = -1):
            z = pyro.sample('latent_z', z_dist)
            
        return z

    

"""
    5. Training via svi
"""


# i) Set up training

cvae = CVAE()

# specifying scalar options
learning_rate = 1e-3
num_epochs = 1000
adam_args = {"lr" : learning_rate}
# Setting up svi
optimizer = pyro.optim.Adam(adam_args)
elbo_loss = pyro.infer.Trace_ELBO()
svi = pyro.infer.SVI(model = cvae.model, guide = cvae.guide, optim = optimizer, loss= elbo_loss)

pyro.clear_param_store()


# ii) Training function

def train(svi, train_loader, use_cuda = False):
    # Initialize loss and cycle through batches
    loss_accumulator = 0
    
    for k, data in enumerate(train_loader):
        # Extract batches
        x_batch = data[0]
        y_batch = data[1]
        # Perform svi gradient step
        temp_loss = svi.step(x_batch, y_batch)
        loss_accumulator = loss_accumulator + temp_loss
        # model_trace = pyro.poutine.trace(cvae.model).get_trace(x_batch)
        # print(model_trace.format_shapes())
        # model_trace.nodes
    
    epoch_loss = loss_accumulator/len(train_loader.dataset)
    return epoch_loss


# iii) Evaluation function

def evaluate(svi, test_loader, use_cuda = False):
    # Initialize loss and cycle through batches
    loss_accumulator = 0
    
    for k, data in enumerate(test_loader):
        # Extract batches
        x_batch = data[0]
        # Perform svi gradient step
        temp_loss = svi.evaluate_loss(x_batch)
        loss_accumulator = loss_accumulator + temp_loss
    
    epoch_loss = loss_accumulator/len(test_loader.dataset)
    return epoch_loss


# iv) Execute training

train_elbo = []
for epoch in range(num_epochs):
    epoch_loss = train(svi, train_loader)
    train_elbo.append(-epoch_loss)
    if epoch % 100 == 0:
        print("Epoch : {} train loss : {}".format(epoch, epoch_loss))



"""
    6. Plots and illustrations
"""

# i) Plot model outputs

# True distribution
n_simu = 1000
x_example = torch.tensor([0]).reshape([-1,1]).float()
y_example_true = torch.zeros([n_simu,1])
for k in range(n_simu):
        y_example_true[k,0] = torch.tensor(np.random.normal(mu_fun(x_example), sigma_fun(x_example)))
y_example_true = y_example_true.float().detach().numpy()

# Model simulations
y_example_model = torch.zeros([n_simu,1])
for k in range(n_simu):
    y_example_model[k,0] = cvae.model(x_example)[0,1].detach()
y_example_model = y_example_model.detach().numpy()


# 1D histograms 
fig, ax = plt.subplots(1, figsize=(5, 5))
ax.hist(y_example_true, bins=20, color='blue', alpha=0.5, label='y from true distribution')
ax.hist(y_example_model, bins=20, color='green', alpha=0.5, label='y from sampling cvae model')
# ax.hist(y_example_predictive, bins=20, color='red', alpha=0.5, label='y from sampling predictive model')
ax.set_title('1D Histograms')
ax.legend()


# ii) Plot predictive distribution

# The predictive as constructed here does the following: Take the model, take the guide
# Then sample all sample sites, i.e. latents z and output y and use the trained 
# parameters.Since latents are sampled with trained guide and output from the
# model, the flow is as follows:
# 1. Feed the value x_example to the guide. This produces a sample of the latents z.
# 2. Feed the value x_example to the model. Instead of sampling the latents 'z'
#   from the prior, take the samples from the guide and swap them in for the
#   sample statement. 
# 3. Then follow the generative process and build the output variables 'y' using
#   'z' samples from the guide.
# In our case, this means that the following happens in a bit more detail: 
# 1. We generate a sample z|x from the guide. First, y_hat is formed, then the
#   recognition net constructs mu_z, sigma_z = recognition_net(x_example, y_hat). 
#   Sampling from N(mu_z, sigma_z) generates values that are clustered around the mean.
# 2. The samples from N(mu_z, sigma_z) are passed to the model. They are swapped 
#   in replacing the sample statement z = N(prior_net(x_example, y_hat)). The
#   recognition_net is stronger than the prior net in inferring z as it is trained
#   on real data (x,y) instead of (x, y_hat), this does seem to translate
#   to a tighter clustering.
# 3. These samples of z are passed into the rest of the model, i.e. first into 
#   the generation net via xy_guess = generation_net(z) and then passed as an output.
# This explains the sharp peak around a certain value for the samples of the 
# posterior compared to the samples taken from the model (utilizing the prior net).
# The difference compared to taking samples from the model is in 'z' being swapped
# in from the approximate posterior q_phi(z|x) (via recognition net) therefore
# the peak is much sharper and focussed more closely around y_hat predicted from
# the baseline. Note that the variance of the synthetic y mirrors the std of the
# noisy observation. y = sample(y_dist) when the recognition_net is used as that
# net was trained to map y to y with the uncertainty being only due to observation
# noise. 


predictive_dist = pyro.infer.Predictive(cvae.model, posterior_samples = None, guide = cvae.guide, num_samples = n_simu)
y_example_predictive = predictive_dist(x_example)['output_y'].squeeze().detach().numpy()

fig, ax = plt.subplots(1, figsize=(5, 5))
ax.hist(y_example_predictive, bins=20, color='red', alpha=0.5, label='y from sampling predictive model')
ax.set_title('Predictive distribution')
ax.legend()


# iii) Plot the output of the guide and some conditional latents
# Compare output of prior_net (x, y_hat) with recognition net (x, y_hat) and (x,y)

# Result of sampling the guide
y_hat_example = cvae.baseline_net(x_example)  

z_example_guide = np.zeros([n_simu,dim_z])
xy_example_guide = np.zeros([n_simu,dim_data])
for k in range(n_simu):
    z_example_guide[k,:] = cvae.guide(x_example).detach()
    xy_example_guide[k,:] = cvae.generation_net(torch.tensor(z_example_guide[k,:]).float()).detach()

# Result of prior_net on (x_example, y_hat)
z_loc_example_prior, z_scale_example_prior = cvae.prior_net(x_example,y_hat_example)
z_dist_example_prior = dist.Normal(z_loc_example_prior, z_scale_example_prior).to_event(1)

z_example_prior = np.zeros([n_simu,dim_z])
xy_example_prior = np.zeros([n_simu,dim_data])
for k in range(n_simu):
    z_example_prior[k,:] = pyro.sample('latent_z_example_prior', z_dist_example_prior).detach()
    xy_example_prior[k,:] = cvae.generation_net(torch.tensor(z_example_prior[k,:]).float()).detach()

# Result of recognition_net on (x_example, y_hat), (x_example, y_example) 
z_loc_example_reco_1, z_scale_example_reco_1 = cvae.recognition_net(x_example,y_hat_example)
z_dist_example_reco_1 = dist.Normal(z_loc_example_reco_1, z_scale_example_reco_1).to_event(1)
z_loc_example_reco_2, z_scale_example_reco_2 = cvae.recognition_net(x_example,torch.tensor(y_example_true[0].reshape([1,1])))
z_dist_example_reco_2 = dist.Normal(z_loc_example_reco_1, z_scale_example_reco_1).to_event(1)

z_example_reco_1 = np.zeros([n_simu,dim_z])
z_example_reco_2 = np.zeros([n_simu,dim_z])
xy_example_reco_1 = np.zeros([n_simu,dim_data])
xy_example_reco_2 = np.zeros([n_simu,dim_data])
for k in range(n_simu):
    z_example_reco_1[k,:] = pyro.sample('latent_z_example_reco_1', z_dist_example_reco_1).detach()
    z_example_reco_2[k,:] = pyro.sample('latent_z_example_reco_2', z_dist_example_reco_2).detach()
    xy_example_reco_1[k,:] = cvae.generation_net(torch.tensor(z_example_reco_1[k,:]).float()).detach()
    xy_example_reco_2[k,:] = cvae.generation_net(torch.tensor(z_example_reco_2[k,:]).float()).detach()


# Plots and illustrations
fig, ax = plt.subplots(2,2, figsize=(15, 15), dpi = 300)
ax[0,0].scatter(z_example_prior[:,0], z_example_prior[:,1])
ax[0,0].set_title(('Prior net (x, y_hat) samples of z'))

ax[0,1].scatter(z_example_guide[:,0], z_example_guide[:,1])
ax[0,1].set_title(('Guide samples of z'))

ax[1,0].scatter(z_example_reco_1[:,0], z_example_reco_1[:,1])
ax[1,0].set_title(('Recognition net (x, y_hat) samples of z'))

ax[1,1].scatter(z_example_reco_2[:,0], z_example_reco_2[:,1])
ax[1,1].set_title(('Recognition net (x, y) samples of z'))




# iv) Samples of y based on different latent variables z

# Plots and illustrations
fig, ax = plt.subplots(2,2, figsize=(15, 15), dpi = 300)
ax[0,0].hist(xy_example_prior[:,1])
ax[0,0].set_title(('y - sample: generation net <- z sampled from prior net (x, y_hat)'))

ax[0,1].hist(xy_example_guide[:,1])
ax[0,1].set_title(('y - sample: generation net <- z sampled from guide'))

ax[1,0].hist(xy_example_reco_1[:,1])
ax[1,0].set_title(('y - sample: generation net <- z sampled from recognition net (x, y_hat)'))

ax[1,1].hist(xy_example_reco_2[:,1])
ax[1,1].set_title(('y - sample: generation net <- z sampled from prior net (x, y)'))


# v) Explore predictive methods - sampling from the posterior on z | x,y and
#   from the model 

predictive = pyro.infer.Predictive(model = cvae.model, posterior_samples = None, guide = cvae.guide, num_samples = 1)
samples = predictive(x_data)
samples_z = samples['latent_z'].squeeze()
samples_y_predictive = samples['output_y'].squeeze()
samples_xy_model = cvae.model(x_data).detach()

# Plot the predictive distributions
fig, axes = plt.subplots(2, 2, figsize=(15, 5))
# Scatter plot of samples_z
axes[0,0].scatter(xy_data[:, 0], xy_data[:, 1])
axes[0,0].set_title('Samples xy data')
axes[0,0].set_xlabel('Dimension 1')
axes[0,0].set_ylabel('Dimension 2')

# Scatter plot of samples_x,y
axes[0,1].scatter(x_data[:], samples_y_predictive[:])
axes[0,1].set_title('Samples of posterior x_data, y_posterior')
axes[0,1].set_xlabel('Dimension 1')
axes[0,1].set_ylabel('Dimension 2')

# Scatter plot of samples_x,y from model
axes[1,0].scatter(x_data, samples_xy_model[:,1])
axes[1,0].set_title('Samples of model x_data, y_model')
axes[1,0].set_xlabel('Dimension 1')
axes[1,0].set_ylabel('Dimension 2')

# Scatter plot of samples_x,y from model
axes[1,1].scatter(samples_xy_model[:,0], samples_xy_model[:,1])
axes[1,1].set_title('Samples of model x_model, y_model')
axes[1,1].set_xlabel('Dimension 1')
axes[1,1].set_ylabel('Dimension 2')

plt.tight_layout()
plt.show()



