#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The goal of this script is to train a variational autoencoder on tri data to
facilitate compression and simulation of radar data. Among other things, we
expect that the latent space corresponds to expressive features, that the 
posterior predictive allows for monte-carlo simulations, and that a distribution
over the latent codes might be useful for anomaly detection and data generation.
These investigations require crafting a procedure that is flexible w.r.t its 
possible inputs and the analyses that can be carried out after training.
For this, do the following:
    1. Definitions and imports
    2. Build auxiliary datasets
    3. Simulate data
    4. Encoder and decoder class
    5. Model, Guide, assembling the VAE
    6. Training via svi
    7. Plots and illustrations
  
What is different from the standard vae setup is that instead of learning a map
to distributional parameters in the bottleneck layer, we prescribe the bottleneck
layer to be standard gaussian distributed. Our approach should work out well
specifically for simulating new data whereas our capability for compression and 
understanding deep features might be lower.
"""


"""
    1. Imports and definitions
"""

# i) imports

import numpy as np
import pyro
import pyro.distributions as dist
import matplotlib.pyplot as plt
import torch
import copy


# ii) Definitions

n_x = 10
n_y = 10
n_total = n_x*n_y
n_samples = 10

dim_data = n_total
dim_z = 2
dim_hidden = 2


x = np.linspace(0,1,n_x)
y = np.linspace(0,1,n_y)

xx, yy = np.meshgrid(y,x)
x_vec = np.vstack((xx.flatten(), yy.flatten())).T

use_cuda = False
pyro.clear_param_store()



"""
    2. Build auxiliary datasets
"""


# i) Range

range_mats = np.zeros([n_samples, n_x, n_y])
r_start = np.zeros([n_samples])
r_end = np.zeros([n_samples])
for k in range(n_samples):
    r_start[k] = np.random.uniform(0,500)
    r_end[k] = r_start[k] + 500 
    range_mats[k,:,:] = np.flipud(np.repeat(np.linspace(r_start[k], r_end[k], n_x).reshape([n_x,1]),n_y,axis = 1))
range_mats = torch.tensor(range_mats).float()


# ii) Azimuth

azimuth_mats = np.zeros([n_samples, n_x, n_y])
az_start = np.zeros([n_samples])
az_end = np.zeros([n_samples])
for k in range(n_samples):
    az_start[k] = np.random.uniform(0,np.pi/2)
    az_end[k] = az_start[k] + np.pi/2
    azimuth_mats[k,:,:] = np.fliplr(np.repeat(np.linspace(az_start[k], az_end[k], n_y).reshape([1, n_y]),n_x,axis = 0))
azimuth_mats = torch.tensor(azimuth_mats).float()


# iii) Location

location_mats = np.zeros([n_samples, n_x, n_y,2])
for k in range(n_samples):
    location_mats[:,:,:,0] = np.cos(azimuth_mats)*range_mats
    location_mats[:,:,:,1] = np.sin(azimuth_mats)*range_mats
location_mats = torch.tensor(location_mats).float()


# iv) Coherence

d_coherence = 0.2
cov_fun_coherence = lambda x_vec1, x_vec2 : np.exp(-((np.linalg.norm(x_vec1-x_vec2,2)/d_coherence)**2))
K_coherence = np.zeros([n_total,n_total])
for k in range(n_total):
    for l in range(n_total):
        K_coherence[k,l] = cov_fun_coherence(x_vec[k,:], x_vec[l,:])
coherence_mats = np.zeros([n_samples, n_x, n_y ])
for k in range(n_samples):
    coherence_mats[k,:,:] = 1 + 1*np.clip(np.random.multivariate_normal(np.zeros(n_total), K_coherence).reshape([n_x,n_y]), -1,0)
coherence_mats = torch.tensor(coherence_mats).float()


# v) Elevation

d_elevation = 0.2
cov_fun_elevation = lambda x_vec1, x_vec2 : 100*np.exp(-((np.linalg.norm(x_vec1-x_vec2,2)/d_elevation)**2))
K_elevation = np.zeros([n_total,n_total])
for k in range(n_total):
    for l in range(n_total):
        K_elevation[k,l] = cov_fun_elevation(x_vec[k,:], x_vec[l,:])
elevation_mats = np.zeros([n_samples, n_x, n_y ])
for k in range(n_samples):
    elevation_mats[k,:,:] = np.random.multivariate_normal(np.zeros(n_total), K_elevation).reshape([n_x,n_y])
elevation_mats = torch.tensor(elevation_mats).float()



"""
    3. Simulate data
"""


# i) Initialize stochastic parameters

covariance_scenario_table = {'range' : 0,\
                             'location' : 1,\
                             'elevation' : 0,\
                             'coherence' : 0
                             }

def build_noise_variance_mats(coherence_mats, covariance_scenario_table):
    
    coherence_dependence = covariance_scenario_table['coherence']
    logical_multiplier = 1 if coherence_dependence else 0
    noise_variance_mats = logical_multiplier * 0.1*(1-coherence_mats**2)
    return noise_variance_mats

def phase_cov_fun(range_1, range_2, elevation_1, elevation_2, location_1, location_2, covariance_scenario_table):
    d_location = 200
    d_elevation = 50
    
    elevation_dependence = covariance_scenario_table['elevation']
    location_dependence = covariance_scenario_table['location']
    range_dependence = covariance_scenario_table['range']
    
    term_1 = (1/1000)*(np.minimum(range_1, range_2)) if range_dependence else 1
    term_2 = np.exp(-((np.linalg.norm(location_1-location_2,2)/d_location)**2)) if location_dependence else 1
    term_3 = (1/100)*np.sqrt(np.abs(elevation_1*elevation_2)) if elevation_dependence else 0
    term_4 = np.exp(-((np.abs(elevation_1 - elevation_2)/d_elevation)**2))
    
    phase_cov_val = term_1*term_2 + term_3*term_4
    return phase_cov_val
    

# ii) Dataset preparation

# basic inputs to stochastic model
class BaseData():
    def __init__(self, range_mats, azimuth_mats, location_mats, elevation_mats, coherence_mats, covariance_scenario_table):
        self.range_mats = range_mats
        self.azimuth_mats = azimuth_mats
        self.location_mats = location_mats
        self.elevation_mats = elevation_mats
        self.coherence_mats = coherence_mats
        self.covariance_scenario_table = covariance_scenario_table
        def get_basic_attr_list(self):
            attributes = [attr for attr in dir(self) if not (attr.startswith('__') or attr.startswith('covariance_scenario'))]
            return attributes
        self.basic_attribute_list = get_basic_attr_list(self)
        self.num_basic_attributes = len(self.basic_attribute_list)
base_data = BaseData(range_mats, azimuth_mats, location_mats, elevation_mats, coherence_mats, covariance_scenario_table)
        
# full outputs of stochastic model
class FullData():
    def __init__(self, base_data, phase_mats, noise_variance_mats, K_phase_mats):
        n_samples = base_data.range_mats.shape[0]
        
        # Integrate distributional_data
        self.noise_variance_mats = noise_variance_mats
        self.K_phase_mats = K_phase_mats
        variances = np.diagonal(K_phase_mats, axis1=1, axis2=2)
        self.aps_variance_mats = variances.reshape((n_samples, n_x, n_y))
        
        # List distributional_data
        def get_distributional_attr_list(self):
            attributes = [attr for attr in dir(self) if not (attr.startswith('__') or attr.startswith('covariance_scenario'))]
            return attributes
        self.distributional_attribute_list = get_distributional_attr_list(self)
        self.num_distributional_attributes = len(self.distributional_attribute_list)

        # copy from base_data
        for name, attribute in base_data.__dict__.items():
            setattr(self, name, attribute)
        # Add simulations
        self.phase_mats = phase_mats            

# iii) Define stochastic model for data generation
        
def simple_model(base_data):
    n_samples = base_data.range_mats.shape[0]
    
    range_mats = base_data.range_mats
    location_mats = base_data.location_mats
    elevation_mats = base_data.elevation_mats
    coherence_mats = base_data.coherence_mats
    covariance_scenario_table = base_data.covariance_scenario_table
    
    noise_variance_mats = build_noise_variance_mats(coherence_mats, covariance_scenario_table)
    phase_mats = np.zeros([n_samples, n_x, n_y ])
    K_phase_mats = np.zeros([n_samples, n_total, n_total])
    for m in range(n_samples):
        for k in range(n_total):
            for l in range(n_total):
                K_phase_mats[m,k,l] = phase_cov_fun(range_mats.reshape([n_samples, -1])[m,k], range_mats.reshape([n_samples, -1])[m,l],
                                                  elevation_mats.reshape([n_samples, -1])[m,k], elevation_mats.reshape([n_samples, -1])[m,l],
                                                  location_mats.reshape([n_samples, -1, 2])[m,k], location_mats.reshape([n_samples, -1, 2])[m,l],
                                                  covariance_scenario_table)
        K_phase_mats[m,:,:] = K_phase_mats[m,:,:] + np.diag(noise_variance_mats[m,:,:].flatten())
    for k in range(n_samples):
        phase_mats[k,:,:] = (np.random.multivariate_normal(np.zeros(n_total), K_phase_mats[k,:,:])).reshape([n_x,n_y])
        
    data = FullData(base_data, phase_mats, noise_variance_mats, K_phase_mats)
    
    return data


# iv) Apply stochastic model

full_data = simple_model(base_data)
observations = torch.tensor(full_data.phase_mats.reshape([n_samples, -1])).float()


# v) Create new dataset
# Create Dataset subclass
class VAEDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

# Invoke class instance
x_data = observations
vae_dataset = VAEDataset(x_data)



"""
    4. Encoder and decoder class
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
        self.fc_2 = torch.nn.Linear(dim_hidden, dim_z)
        # nonlinear transforms
        self.nonlinear = torch.nn.Identity()
        
    def forward(self, x):
        # Define forward computation on the input data x
        # Shape the minibatch so that batch_dims are on left, event_dims on right
        x = x.reshape([-1,dim_data])
        
        # Then compute hidden units and output of nonlinear pass
        hidden_units = self.nonlinear(self.fc_1(x))
        z = self.fc_2(hidden_units)
        return z


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
    5. Model, Guide, assembling the VAE
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

    
    # ii) Define model - is built as map from x to x_guess via deterministic 
    # "latent" z that are scored against standard normal
    
    def model(self, x):
        # register the decoder parameters with pyro
        pyro.module("decoder", self.decoder)
        pyro.module("encoder", self.encoder)
        # construct the x via sampling statements of a batch of z
        z_loc = torch.zeros([x.shape[0], self.dim_z])
        z_scale = 1*torch.ones([x.shape[0], self.dim_z])
        z_dist = dist.Normal(z_loc, z_scale).to_event(1)
        with pyro.plate("batch_plate", size = x.shape[0], dim = -1):      
            # set latent code to encoding, observe via prior
            encoded_x = self.encoder(x)
            z = pyro.sample('latent_z', z_dist, obs = encoded_x)
            # construct x via transforming z
            decoded_z = self.decoder(z)
            x_dist = dist.Normal(decoded_z, 0.01).to_event(1)
            x_guess = pyro.sample("x_obs" , x_dist, obs = x.reshape([-1,dim_data]))
        
        return x_guess


    # iii) Define guide - since in this VAE variant there are no stochastic 
    # latents, it is empty. We let it pass the encoded z though, so it integrates
    # with the rest of the script
    
    def guide(self, x):
        z_obs = self.encoder(x)
        return z_obs


    # iv) Define some support and illustration functions

    def reconstruct_point(self, x):
        # encode datapoint x
        z = self.encoder(x)
        # decode the latent code
        x_guess = self.decoder(z)
        return x_guess
    

"""
    6. Training via svi
"""


# i) Set up training

vae = VAE()

# specifying scalar options
learning_rate = 0.003
num_epochs = 5000
adam_args = {"lr" : learning_rate}
# Setting up svi
# optimizer = pyro.optim.Adam(adam_args)
optimizer = pyro.optim.ClippedAdam(adam_args)
elbo_loss = pyro.infer.Trace_ELBO()
svi = pyro.infer.SVI(model = vae.model, guide = vae.guide, optim = optimizer, loss= elbo_loss)

vae_dataloader = torch.utils.data.DataLoader(vae_dataset, batch_size=10, shuffle=True, num_workers=0) 
pyro.clear_param_store()


# ii) Training function       

def train(svi, train_loader, train_log, use_cuda = False):
    # Initialize loss and cycle through batches
    loss_accumulator = 0
    
    for k, x_batch in enumerate(train_loader):
        # pass minibatch to cuda
        if use_cuda == True:
            x_batch = x_batch.cuda()
        # Perform svi gradient step
        temp_loss = svi.step(x_batch)
        loss_accumulator = loss_accumulator + temp_loss
        
    # Add diagnostics
    epoch_loss = loss_accumulator/len(train_loader.dataset)
    model_trace = pyro.poutine.trace(vae.model).get_trace(x_batch)
    param_store = pyro.get_param_store()
    temp_train_info = TempTrainInfo(x_batch, vae, epoch_loss, model_trace, param_store)
    temp_train_info.fun_update_train_log(train_log)
    # train_log = log_train_info(temp_train_info, train_log)
    
    return epoch_loss, train_log


class TempTrainInfo():
    def __init__(self, x_data, vae, epoch_loss, model_trace, param_store):
        self.epoch_loss = torch.tensor(epoch_loss)
                
        mu_enc = vae.encoder(x_data)[0]
        sigma_enc = vae.encoder(x_data)[1]
        mu_dec = vae.decoder(mu_enc)
        
        self.mu_enc = mu_enc
        self.sigma_enc = sigma_enc
        self.mu_dec = mu_dec
        
        self.mu_enc_highest = torch.max(mu_enc)
        self.mu_enc_lowest = torch.min(mu_enc)
        self.mu_dec_highest = torch.max(mu_dec)
        self.mu_dec_lowest = torch.min(mu_dec)
        self.sigma_enc_highest = torch.max(sigma_enc)
        self.sigma_enc_lowest = torch.min(sigma_enc)
        
        self.attribute_list =[attr for attr in dir(self) if not (attr.startswith('__') or attr.startswith('fun')) ]
        
    def fun_update_train_log(self, train_log):
        train_dict = {}
        for attr in self.attribute_list:
            train_dict[attr] = getattr(self, attr)   
        train_log.append(train_dict)
        return train_log
    
    def fun_reformat_train_log(self, train_log):
        
        return train_log_reformatted
        

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

train_log = []
for epoch in range(num_epochs):
    epoch_loss, train_log = train(svi, vae_dataloader, train_log)
    if epoch % 100 == 0:
        print("Epoch : {} train loss : {}".format(epoch, epoch_loss))



"""
    7. Plots and illustrations
"""


# i) Plot inference history

# Reformat the train_log
train_log_reformatted = {}
n_train = len(train_log)
train_example = train_log[0]
for key, value in train_example.items():
    value_shape = value.shape if hasattr(value, 'shape') else [1]
    train_log_reformatted[key] = np.zeros([n_train, *value_shape])
for epoch in range(n_train):
    for key, value in train_log[epoch].items():
        save_val =  value.detach().numpy() if type(value) == torch.Tensor else value
        train_log_reformatted[key][epoch] = save_val


def visualize_training(train_log_reformatted):
    fig, axes = plt.subplots(4, 2, figsize=(10, 12))

    # Plot data on each subplot with a full plot and one zoomed in
    
    zoom_index = np.round(0.8*n_train).astype(int)
    epochs = np.linspace(0,n_train-1,n_train)
    epochs_zoom = np.linspace(epochs[zoom_index], epochs[-1], n_train - zoom_index)
    
    # Epoch loss
    axes[0,0].plot(epochs,train_log_reformatted['epoch_loss'])
    axes[0,0].set_title('ELBO objective')
    axes[0,0].set_ylabel('Elbo')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].grid(True)
    
    axes[0,1].plot(epochs_zoom, train_log_reformatted['epoch_loss'][zoom_index : ])
    axes[0,1].set_title('ELBO objective zoom-in')
    axes[0,1].set_ylabel('Elbo')
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].grid(True)
    
    # Encoder mean extreme elements
    axes[1,0].plot(epochs,train_log_reformatted['mu_enc_highest'])
    axes[1,0].plot(epochs,train_log_reformatted['mu_enc_lowest'])
    axes[1,0].set_title('Highest and lowest encoder means')
    axes[1,0].set_ylabel('Mean')
    axes[1,0].set_xlabel('Epoch')
    axes[1,0].grid(True)
    
    axes[1,1].plot(epochs_zoom, train_log_reformatted['mu_enc_highest'][zoom_index : ])
    axes[1,1].plot(epochs_zoom, train_log_reformatted['mu_enc_lowest'][zoom_index : ])
    axes[1,1].set_title('Encoder means zoom-in')
    axes[1,1].set_ylabel('Mean')
    axes[1,1].set_xlabel('Epoch')
    axes[1,1].grid(True)
    
    # Decoder mean extreme elements
    axes[2,0].plot(epochs,train_log_reformatted['mu_dec_highest'])
    axes[2,0].plot(epochs,train_log_reformatted['mu_dec_lowest'])
    axes[2,0].set_title('Highest and lowest decoder means')
    axes[2,0].set_ylabel('Mean')
    axes[2,0].set_xlabel('Epoch')
    axes[2,0].grid(True)
    
    axes[2,1].plot(epochs_zoom, train_log_reformatted['mu_dec_highest'][zoom_index : ])
    axes[2,1].plot(epochs_zoom, train_log_reformatted['mu_dec_lowest'][zoom_index : ])
    axes[2,1].set_title('Decoder means zoom-in')
    axes[2,1].set_ylabel('Mean')
    axes[2,1].set_xlabel('Epoch')
    axes[2,1].grid(True)

    # Encoder variance extreme elements
    axes[3,0].plot(epochs,train_log_reformatted['sigma_enc_highest'])
    axes[3,0].plot(epochs,train_log_reformatted['sigma_enc_lowest'])
    axes[3,0].set_title('Highest and lowest encoder variances')
    axes[3,0].set_ylabel('Var')
    axes[3,0].set_xlabel('Epoch')
    axes[3,0].grid(True)
    
    axes[3,1].plot(epochs_zoom, train_log_reformatted['sigma_enc_highest'][zoom_index : ])
    axes[3,1].plot(epochs_zoom, train_log_reformatted['sigma_enc_lowest'][zoom_index : ])
    axes[3,1].set_title('Decoder variances zoom-in')
    axes[3,1].set_ylabel('Var')
    axes[3,1].set_xlabel('Epoch')
    axes[3,1].grid(True)
    
    
    # Add some spacing between the subplots for better readability
    plt.tight_layout()
    
    plt.show()

visualize_training(train_log_reformatted)




# ii) Reconstruct & simulate new data 
x_data_rec = vae.reconstruct_point(x_data).detach()

# Define new model for data prediction by randomly sampling in the bottleneck layer
def predictive_model(x):
    # construct the x via sampling statements of a batch of z
    z_dist = pyro.distributions.Normal(loc = torch.zeros([x.shape[0], dim_z]), scale = torch.ones([x.shape[0], dim_z])).to_event(1)
    z = pyro.sample("latent_t", z_dist)
    # construct x via transforming z
    decoded_z = vae.decoder(z)
    x_dist = dist.Normal(decoded_z, 0.01).to_event(1)
    x_guess = pyro.sample("x_obs" , x_dist)
    
    return x_guess

x_data_new = predictive_model(x_data).detach().numpy()

# Illustrate
fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(10, 10))
original_images = x_data.reshape([-1,n_x,n_y])
new_images = x_data_new.reshape([-1,n_x,n_y])
for i, ax in enumerate(axes.flatten()):
    if i <= 3:
        ax.imshow(original_images[i,:,:])
        ax.set_title('original_data')
        ax.axis('off')
    else:
        ax.imshow(new_images[i,:,:])
        ax.set_title('synthetic_data')
        ax.axis('off')

plt.tight_layout()
plt.show()


# iii) Illustrate encoding and decoding
num_rows = 4
def plot_vae_results(vae_model, original_data):
    fig, axes = plt.subplots(num_rows, 5, figsize=(20, 4 * num_rows))

    for i in range(num_rows):
        # Panel 1: Original Data Point
        ax = axes[i, 0]
        ax.imshow(original_data[i,:].reshape([n_x,n_y]))
        ax.set_title("Original Data")
        ax.axis("off")

        # Panel 2: Scatterplot of Latent Codes from Original Data
        ax = axes[i, 1]
        latent_codes = [vae_model.guide(original_data[i,:].reshape([1,-1])).detach() for _ in range(50)]
        latent_codes = torch.cat(latent_codes).detach()
        ax.scatter(latent_codes[:, 0], latent_codes[:, 1])
        ax.set_title("Latent Codes Scatter")
        ax.set_xlabel("Latent Dim 1")
        ax.set_ylabel("Latent Dim 2")

        # Panels 3-5: Random Latent Codes Decoded
        for j in range(2, 5):
            ax = axes[i, j]
            random_latent_code = latent_codes[j,:]
            decoded_image = vae_model.decoder(random_latent_code).reshape([n_x,n_y])
            ax.imshow(decoded_image.detach())
            ax.set_title(f"Decoded Random Latent #{j-1}")
            ax.axis("off")

    plt.tight_layout()
    plt.show()
    
plot_vae_results(vae, x_data)
    
    
# iv) Investigate latent space

# Plot latent space dims
num_codesims = 30
latent_codes = torch.zeros([n_samples, num_codesims, dim_z])
for k in range(n_samples):
    latent_code_list = [vae.guide(x_data[k,:].reshape([1, -1])).reshape([1, 1, -1]) for _ in range(num_codesims)]
    latent_codes[k,:]  = torch.cat(latent_code_list, dim = 1)
latent_codes = latent_codes.reshape([-1,dim_z])
n_total_codes = latent_codes.shape[0]
latent_codes_2d = latent_codes[:, 0:2].detach().numpy()

# Create the 2D histogram plot
fig, axes = plt.subplots(1, 1, figsize=(8, 8))
axes.hist2d(latent_codes_2d[:, 0], latent_codes_2d[:, 1], bins=(50, 50), cmap='viridis')
axes.set_xlabel('Dimension 1')
axes.set_ylabel('Dimension 2')
axes.set_title('2D Histogram of Latent Codes - Panel 1')

plt.tight_layout()
plt.show()
    

































    
    
    
    
    
    
    
    
    
    
    
    
    
  