#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The goal of this script is to showcase a very basic variational autoencoder that
can be used to encode and decode multivariate Gaussian data.
For this, do the following:
    1. Definitions and imports
    2. Generate data
    3. Encoder and decoder class
    4. Model, Guide, assembling the VAE
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

n_data = 100
dim_data = 2
dim_z = 2
dim_hidden = 3
use_cuda = False

 

"""
    2. Generate data
"""

# i) Draw from multivariate gaussian

mu_true = np.zeros([2])
sigma_true = np.array([[1,0.9],[0.9,1]])

x_data_prel = np.random.multivariate_normal(mu_true,sigma_true, size = [n_data])


# ii) Apply nonlinear function

nonlinear_fun = lambda x : np.array([x[:,0],x[:,1]]).T
# nonlinear_fun = lambda x : np.array([x[:,0],x[:,1]**2]).T
x_data = torch.tensor(nonlinear_fun(x_data_prel)).float()


# iii) Create new dataset
# Create Dataset subclass
class VAEDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

# Invoke class instance
vae_dataset = VAEDataset(x_data)




"""
    3. Encoder and decoder class
"""


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
        self.nonlinear = torch.nn.Identity()
        
    def forward(self, x):
        # Define forward computation on the input data x
        # Shape the minibatch so that batch_dims are on left, event_dims on right
        x = x.reshape([-1,dim_data])
        
        # Then compute hidden units and output of nonlinear pass
        hidden_units = self.nonlinear(self.fc_1(x))
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
        self.fc_2 = torch.nn.Linear(dim_hidden, dim_data)
        # nonlinear transforms
        self.nonlinear_1 = torch.nn.Identity()
        self.nonlinear_2 = torch.nn.Identity()
    
    def forward(self, z):
        # Define forward computation on the latent codes z
        # Compute hidden units and output of nonlinear pass
        hidden = self.nonlinear_1(self.fc_1(z))
        x_guess = self.nonlinear_2(self.fc_2(hidden))
        return x_guess        



"""
    4. Model, Guide, assembling the VAE
"""


# i) Build VAE class with model and guide included later

class VAE(pyro.nn.PyroModule):
    def __init__(self, dim_z = dim_z, dim_hidden = dim_hidden, use_cude = use_cuda):
        super().__init__()      # __init__ method of base class
        # Initialize encoder and decoder
        self.encoder = Encoder(dim_z, dim_hidden)
        self.decoder = Decoder(dim_z, dim_hidden)
        
        # if use_cuda = true, then put all params into gpu memory
        if use_cuda == True:
            self.cuda()
        self.use_cuda = use_cuda
        self.dim_z = dim_z
        self.dim_hidden = dim_hidden     
        
        
    # ii) Define model - is built by the decoder as the decoder ties x to z via the
    # likelihood p_theta(x|z)p(z)
    
    def model(self, x):
        # register the decoder parameters with pyro
        pyro.module("decoder", self.decoder)
        # construct the x via sampling statements of a batch of z
        z_loc = torch.zeros([x.shape[0], self.dim_z])
        z_scale = torch.ones([x.shape[0], self.dim_z])
        z_dist = dist.Normal(z_loc, z_scale).to_event(1)
        with pyro.plate("batch_plate", size = x.shape[0], dim = -1):      
            # sample the latent codes z from a prior
            z = pyro.sample('latent_z', z_dist)
            # construct x via transforming z
            decoded_z = self.decoder(z)
            x_dist = dist.Normal(decoded_z, 0.1).to_event(1)
            x_guess = pyro.sample("x_obs" , x_dist, obs = x.reshape([-1,dim_data]))
        
        return x_guess


    # iii) Define guide - is built by the encoder as the encoder ties z to x via the
    # the probability q_phi(z|x)
    
    def guide(self, x):
        # register the encoder parameters with pyro
        pyro.module("encoder", self.encoder)
        # construct z via sampling statements of a batch of x
        # use encoder to get distributional parameters
        z_loc, z_scale = self.encoder(x)
        z_dist = dist.Normal(z_loc, z_scale).to_event(1)
        # sample independent batch elements
        with pyro.plate("batch_plate", size = x.shape[0], dim = -1):
            z = pyro.sample("latent_z", z_dist)

        return z


    # iv) Define some support and illustration functions

    def reconstruct_point(self, x):
        # encode datapoint x
        z_loc, z_scale = self.encoder(x)
        # sample in latent space
        z = dist.Normal(z_loc, z_scale).sample()
        # decode the latent code
        x_guess = self.decoder(z)
        return x_guess
    

"""
    5. Training via svi
"""


# i) Set up training

vae = VAE()

# specifying scalar options
learning_rate = 1e-2
num_epochs = 200
adam_args = {"lr" : learning_rate}
# Setting up svi
optimizer = pyro.optim.Adam(adam_args)
elbo_loss = pyro.infer.Trace_ELBO()
svi = pyro.infer.SVI(model = vae.model, guide = vae.guide, optim = optimizer, loss= elbo_loss)

vae_dataloader = torch.utils.data.DataLoader(vae_dataset, batch_size=10, shuffle=True, num_workers=0) 
pyro.clear_param_store()


# ii) Training function

def train(svi, train_loader, use_cuda = False):
    # Initialize loss and cycle through batches
    loss_accumulator = 0
    
    for k, x_batch in enumerate(train_loader):
        # pass minibatch to cuda
        if use_cuda == True:
            x_batch = x_batch.cuda()
        # Perform svi gradient step
        temp_loss = svi.step(x_batch)
        loss_accumulator = loss_accumulator + temp_loss
        # model_trace = pyro.poutine.trace(vae.model).get_trace(x_batch)
        # print(model_trace.format_shapes())
        # model_trace.nodes
    
    epoch_loss = loss_accumulator/len(train_loader.dataset)
    return epoch_loss


# iii) Evaluation function

def evaluate(svi, test_loader, use_cuda = False):
    # Initialize loss and cycle through batches
    loss_accumulator = 0
    
    for k, x_batch in enumerate(test_loader):
        # pass minibatch to cuda
        if use_cuda == True:
            x_batch = x_batch.cuda()
        # Perform svi gradient step
        temp_loss = svi.evaluate_loss(x_batch)
        loss_accumulator = loss_accumulator + temp_loss
    
    epoch_loss = loss_accumulator/len(test_loader.dataset)
    return epoch_loss


# iv) Execute training

train_elbo = []
for epoch in range(num_epochs):
    epoch_loss = train(svi, vae_dataloader)
    train_elbo.append(-epoch_loss)
    if epoch % 10 == 0:
        print("Epoch : {} train loss : {}".format(epoch, epoch_loss))



"""
    6. Plots and illustrations
"""


# i) Reconstruct data

x_data_rec = vae.reconstruct_point(x_data).detach()


# ii) Simulate new data 
# Do it by exploring predictive methods - sampling new data x

# Define new model for data prediction (without obs keyword)
def predictive_model(x):
    # construct the x via sampling statements of a batch of z
    z = vae.guide(x)
    # construct x via transforming z
    decoded_z = vae.decoder(z)
    x_dist = dist.Normal(decoded_z, 0.1).to_event(1)
    x_guess = pyro.sample("x_obs" , x_dist)
    
    return x_guess

x_data_new = predictive_model(x_data).detach().numpy()




# iii) Illustrate via scatterplot

fig, ax = plt.subplots(3, figsize=[10, 10], dpi=300)

ax[0].scatter(x_data[:,0], x_data[:,1])
ax[0].set_title('Original dataset')
ax[0].set_xlabel('x-axis')
ax[0].set_ylabel('y-axis')

ax[1].scatter(x_data_rec[:,0], x_data_rec[:,1])
ax[1].set_title('Reconstructed dataset')
ax[1].set_xlabel('x-axis')
ax[1].set_ylabel('y-axis')

ax[2].scatter(x_data_new[:,0], x_data_new[:,1])
ax[2].set_title('New dataset')
ax[2].set_xlabel('x-axis')
ax[2].set_ylabel('y-axis')

plt.tight_layout()
plt.show()


# iv) Explore predictive methods - sampling from the posterior on z

predictive = pyro.infer.Predictive(model = vae.model, posterior_samples = None, guide = vae.guide, num_samples = 1)
# samples = predictive()
samples = predictive(x_data)
# samples = predictive.get_samples(x_data)
samples_z = samples['latent_z'].squeeze()
samples_x = samples['x_obs'].squeeze()
x_data_new = samples_x


# Plot the predictive distributions
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Scatter plot of samples_z
axes[0].scatter(samples_z[:, 0], samples_z[:, 1])
axes[0].set_title('samples_z')
axes[0].set_xlabel('Dimension 1')
axes[0].set_ylabel('Dimension 2')

# Scatter plot of samples_x
axes[1].scatter(samples_x[:, 0], samples_x[:, 1])
axes[1].set_title('samples_x')
axes[1].set_xlabel('Dimension 1')
axes[1].set_ylabel('Dimension 2')

plt.tight_layout()
plt.show()



