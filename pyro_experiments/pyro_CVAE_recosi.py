#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The goal of this script is to showcase a very basic conditional variational 
autoencoder that can be used to encode and decode a simple model featuring
two variables x,y where x and why are generated from a joint distribution and 
might represent e.g. x = temperature, humidity and y = dispersion angle. We 
train the model to learn the conditional distributions and sample from them.
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
import seaborn as sns
import pandas as pd


# ii) Definitions

n_data_x = 30
n_data_y = 30
n_data = n_data_x*n_data_y
dim_data_x = 2
dim_data_y = 2
dim_data = dim_data_x + dim_data_y
dim_z = 10
dim_hidden = 10
use_cuda = False

np.random.seed(1)
torch.manual_seed(1)
 

"""
    2. Generate data
"""

# # i) Draw samples for x and y
# # Do this by generating synthetic data that exhibits a complicated relationship 
# # and split it into a part that belongs to the predictor variables x on which we 
# # will condition and the inferred variables y from which we want to sample y|x

# sigma_noise = 0.1
# A = np.random.normal(0,1, size = [dim_data, dim_data])
# A[0,:] = np.eye(dim_data)[0,:]
# def generate_data(xy_data_init):
#     n_pt = xy_data_init.shape[0]
#     rand_data = np.random.normal(0,1,size = [n_pt,dim_data-1])
#     xy_data_augmented = np.hstack((xy_data_init,rand_data))
#     noise = np.random.normal(0,sigma_noise,[n_pt,dim_data])
    
#     lin_trans = lambda xy : xy @ A.T
#     nonlin_trans = lambda xy : np.hstack((xy[:,0].reshape([-1,1]), np.tanh(xy[:,1:])))
    
#     xy_data_transformed = lin_trans(xy_data_augmented)
#     xy_data_transformed = nonlin_trans(nonlin_trans(xy_data_transformed))
#     xy_data_transformed = lin_trans(xy_data_transformed) + noise
#     return torch.tensor(xy_data_transformed)

# xy_data_init = np.linspace(-1,1,n_data).reshape([-1,1])
# xy_data = generate_data(xy_data_init)

# x_data = xy_data[:,0:dim_data_x].reshape([-1,dim_data_x]).float()
# y_data = xy_data[:,dim_data_x:].reshape([-1,dim_data_y]).float()

# Simple model where y is multivariate sample dependent on x
def mu_fun(x):
    y_1 = (x[:,0]*x[:,1]).reshape([-1,1])
    y_2 = (x[:,0] + x[:,1]).reshape([-1,1])
    y = np.hstack((y_1,y_2))
    return y

def sigma_fun(x):
    sigma_mat = np.zeros([x.shape[0], dim_data_y,dim_data_y])
    sigma_mat[:, 0,0] = np.abs(x[:,0])
    sigma_mat[:, 1,1] = np.abs(x[:,0]*x[:,1])
    return 0.1*sigma_mat


x_grid_1 = torch.linspace(-1,1,n_data_x)
x_grid_2 = torch.linspace(-1,1,n_data_y)
x_grid = torch.meshgrid(x_grid_1,x_grid_2,indexing = 'xy' )
x_data = torch.vstack((torch.flipud(x_grid[0]).flatten(), torch.flipud(x_grid[1]).flatten())).T
y_true = mu_fun(x_data)
sigma_true = sigma_fun(x_data)

y_data = torch.zeros([n_data,dim_data_y])
for k in range(n_data):
    y_data[k,:] = torch.tensor(np.random.multivariate_normal(y_true[k,:], sigma_true[k,:,:])).float()



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

# Create Pairgrid to illustrate data
visualization_frame = pd.DataFrame(xy_data.detach().numpy(),columns=['x_1', 'x_2', 'y_1', 'y_2'])
pairgrid = sns.PairGrid(visualization_frame)
# Map plots to the grid
pairgrid.map_diag(sns.histplot) # Diagonal plots as histograms
pairgrid.map_upper(sns.scatterplot) # Upper plots as scatter plots
pairgrid.map_lower(sns.kdeplot) # Lower plots as kde plots
plt.show()

# Create predictions and illustrate
y_pred_bl = torch.zeros([n_data,dim_data_y])
for k in range(n_data):
    y_pred_bl[k,:] = baseline_net(x_data[k,:]).detach()

# A square scatterplot showing 1d slices od predictions
# x_1, y_1
fig, axs = plt.subplots(2,2, figsize=(10, 10), dpi = 300) 
axs[0,0].scatter(x_data[:,0], y_data[:,0], label = 'x_1 y_1')
axs[0,0].plot(x_data[:,0], y_pred_bl[:,0], color = 'red', label = 'baseline')
axs[0,0].set_title('Data x_1, y_1 and baseline prediction')
axs[0,0].legend()
# x_1, y_2
axs[0,1].scatter(x_data[:,0], y_data[:,1], label = 'x_1 y_2')
axs[0,1].plot(x_data[:,0], y_pred_bl[:,1], color = 'red', label = 'baseline')
axs[0,1].set_title('Data x_1, y_2 and baseline prediction')
axs[0,1].legend()
# x_2, y_1
axs[1,0].scatter(x_data[:,1], y_data[:,0], label = 'x_2 y_1')
axs[1,0].plot(x_data[:,1], y_pred_bl[:,0], color = 'red', label = 'baseline')
axs[1,0].set_title('Data x_2, y_1 and baseline prediction')
axs[1,0].legend()
# x_2, y_2
axs[1,1].scatter(x_data[:,1], y_data[:,1], label = 'x_2 y_2')
axs[1,1].plot(x_data[:,1], y_pred_bl[:,1], color = 'red', label = 'baseline')
axs[1,1].set_title('Data x_2, y_2 and baseline prediction')
axs[1,1].legend()
plt.tight_layout()
plt.show()

# Create a new figure with subplots side by side to showcase data dependence
# and baseline predictions
n_pt_grid = 30
x_min = np.min(x_data.numpy(), axis = 0)
x_max = np.max(x_data.numpy(), axis = 0)
x_grid_1 = torch.linspace(x_min[0],x_max[0],n_pt_grid)
x_grid_2 = torch.linspace(x_min[1],x_max[1],n_pt_grid)
x_grid = torch.meshgrid(x_grid_1,x_grid_2,indexing = 'xy' )
x_vec = torch.vstack((torch.flipud(x_grid[0]).flatten(), torch.flipud(x_grid[1]).flatten())).T
baseline_pred_grid = baseline_net(x_vec)
y_grid_1 = baseline_pred_grid[:,0].reshape([n_pt_grid,n_pt_grid])
y_grid_2 = baseline_pred_grid[:,1].reshape([n_pt_grid,n_pt_grid])

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 15), dpi = 300)
# First scatterplot illustrating dependence of y_1 on x_1, x_2
sc1 = axs[0,0].scatter(xy_data[:,0], xy_data[:,1], c=xy_data[:,2])
axs[0,0].set_title('Dependence of y_1')
axs[0,0].set_xlabel('x_1')
axs[0,0].set_ylabel('x_2')
plt.colorbar(sc1, ax=axs[0,0], label='y_1')
# Second scatterplot illustrating dependence of y_2 on x_1,x_2
sc2 = axs[0,1].scatter(xy_data[:,0], xy_data[:,1], c=xy_data[:,3])
axs[0,1].set_title('Dependence of y_2')
axs[0,1].set_xlabel('x_1')
axs[0,1].set_ylabel('x_2')
plt.colorbar(sc2, ax=axs[0,1], label='y_2')
# Plot illustrating baseline predictions of y_1 on x_1, x_2
sc3 = axs[1,0].imshow(y_grid_1.detach(),extent = [x_min[0], x_max[0], x_min[1], x_max[1]], aspect='auto')
axs[1,0].set_title('Baseline predictions of y_1')
axs[1,0].set_xlabel('x_1')
axs[1,0].set_ylabel('x_2')
plt.colorbar(sc1, ax=axs[1,0], label='y_1')
# Plot illustrating baseline predictions of y_2 on x_1, x_2
sc4 = axs[1,1].imshow(y_grid_2.detach(),extent = [x_min[0], x_max[0], x_min[1], x_max[1]], aspect='auto')
axs[1,1].set_title('Baseline predictions of y_2')
axs[1,1].set_xlabel('x_1')
axs[1,1].set_ylabel('x_2')
plt.colorbar(sc1, ax=axs[1,1], label='y_2')

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
x_example_true = torch.zeros([n_simu,2])

y_example_true = torch.zeros([n_data,dim_data_y])
y_mean = mu_fun(x_example_true)
y_sigma = sigma_fun(x_example_true)
for k in range(n_data):
    y_example_true[k,:] = torch.tensor(np.random.multivariate_normal(y_mean[k,:], sigma_true[k,:,:])).float()
# Model simulations
y_example_model = cvae.model(x_example_true)[:,dim_data_x:].detach().numpy()


# 2D scatterplot
fig, ax = plt.subplots(2, figsize=(5, 5))
ax[0].scatter(y_example_true[:,0], y_example_true[:,1], alpha=1)
ax[0].set_title('y|x from true distribution')
ax[0].set_ylabel('y_2')
ax[1].scatter(y_example_model[:,0], y_example_model[:,1], alpha=1)
ax[1].set_title('y|x from model distribution')
ax[1].set_ylabel('y_2')
ax[1].set_xlabel('y_1')
plt.tight_layout()
plt.show()








