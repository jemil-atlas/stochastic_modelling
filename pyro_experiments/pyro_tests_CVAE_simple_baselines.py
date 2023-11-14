#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The goal of this script is to showcase a very basic conditional variational 
autoencoder that can be used to encode and decode multivariate Gaussian data 
and sample from conditional distributions.
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

n_data = 100
dim_data_x = 2
dim_data_y = 2
dim_data = dim_data_x + dim_data_y
dim_z = 2
dim_hidden = 10
use_cuda = False

 

"""
    2. Generate data
"""

# i) Draw from multivariate gaussian

mu_true = np.zeros([dim_data])
sigma_true = np.eye(dim_data)
np.fill_diagonal(sigma_true[1:], 0.5)  
np.fill_diagonal(sigma_true[:, 1:], 0.5)  

xy_data_prel = np.random.multivariate_normal(mu_true,sigma_true, size = [n_data])
xy_data_prel_unseen = np.random.multivariate_normal(mu_true,sigma_true, size = [n_data])
mask_with = -100


# ii) Apply nonlinear function

# nonlinear_fun = lambda x : x
nonlinear_fun = lambda x : torch.tensor([x[:,0],x[:,1],x[:,1],x[:,1]**2]).T
# nonlinear_fun = lambda x : torch.tensor([x[:,0], x[:,1], x[:,0]**2, x[:,1]**2]).T
xy_data = torch.tensor(nonlinear_fun(xy_data_prel)).float()
xy_data_unseen = torch.tensor(nonlinear_fun(xy_data_prel_unseen)).float()


# iii) Create new dataset
# Create Dataset subclass
class CVAEDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        datapoint = self.data[index]
        # The 'original' key contains the whole datapoint (both input and output features),
        # 'input' contains only input features, and 'output' only output features.
        return {'original': datapoint,
                'input': torch.hstack((datapoint[:dim_data_x], mask_with*torch.ones(dim_data_y))),
                'output': torch.hstack((mask_with*torch.ones(dim_data_x),datapoint[dim_data_x:]))
                }

    def __len__(self):
        return len(self.data)
    
    def get_x_data(self):
        x_data = torch.empty(size = [0,dim_data])
        for k in range(self.__len__()):
            x_data = torch.vstack((x_data,self.__getitem__(k)['input']))
        return x_data
    def get_y_data(self):
        y_data = torch.empty(size = [0,dim_data])
        for k in range(self.__len__()):
            y_data = torch.vstack((y_data,self.__getitem__(k)['output']))
        return y_data


# iv) Invoke class instance

# Initialize original dataset containing xy data
cvae_dataset = CVAEDataset(xy_data)
x_data = cvae_dataset.get_x_data()
y_data = cvae_dataset.get_y_data()

# unseen dataset
cvae_dataset_unseen = CVAEDataset(xy_data_unseen)
x_data_unseen = cvae_dataset_unseen.get_x_data()
y_data_unseen = cvae_dataset_unseen.get_y_data()

# Split the dataset into train and test parts: 80% for training and 20% for testing.
train_size = int(0.8 * len(cvae_dataset))
test_size = len(cvae_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(cvae_dataset, [train_size, test_size])


# Wrap the datasets into DataLoaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# Build a new dataset containing train test splits
cvae_all_datasets = dict()
cvae_all_datasets['original'] = cvae_dataset
cvae_all_datasets['train'] = train_dataset
cvae_all_datasets['test'] = test_dataset




"""
    3. Encoder and decoder class, baseline_net
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
        
    def forward(self, x, y):
        # Define forward computation on the input data x and output data y
        # Shape the minibatch so that batch_dims are on left, event_dims on right
        x = x.reshape([-1,dim_data_x]) # TODO Something might be wrong here, dimensions might not match (reshaped to [?,2] but input is [?,4]!
        y = y.reshape([-1,dim_data_y])
        xy = x.clone()
        xy[x==mask_with] = y[x==mask_with]
        xy = xy.reshape([-1, dim_data])
        
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
        self.nonlinear_1 = torch.nn.Identity()
        self.nonlinear_2 = torch.nn.Identity()
    
    def forward(self, z):
        # Define forward computation on the latent codes z
        # Compute hidden units and output of nonlinear pass
        hidden = self.nonlinear_1(self.fc_1(z))
        hidden = self.nonlinear_2(self.fc_2(hidden))
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
        x = x.view(-1, dim_data_x)
        hidden = self.nonlinear(self.fc1(x))
        hidden = self.nonlinear(self.fc2(hidden))
        y = self.fc3(hidden)
        return y


# iv) Train baseline_net

def reset_parameters(model):
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

baseline_net = BaselineNet(dim_data_x, dim_hidden, dim_data_y)
reset_parameters(baseline_net)
optimizer_baseline = torch.optim.Adam(baseline_net.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss()

epochs = 1000
for epoch in range(epochs):
    for i, batch in enumerate(train_loader):
        inputs = batch["input"][:, 0:dim_data_x]
        outputs = batch["output"][:, dim_data_x:]

        optimizer_baseline.zero_grad()
        preds = baseline_net(inputs)
        loss = criterion(preds, outputs)
        loss.backward()
        optimizer_baseline.step()

    if epoch % 50 == 0:
        print('Epoch {} baseline train loss : {}'.format(epoch, loss.item()))



"""
    4. Model, Guide, assembling the VAE
"""


# i) Build CVAE class with model and guide included later

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
        
        # if use_cuda = true, then put all params into gpu memory
        if use_cuda == True:
            self.cuda()
        self.use_cuda = use_cuda
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
        with torch.no_grad():
            # this ensures the training process does not change the
            # baseline network
            y_hat = torch.hstack((mask_with*torch.ones([x.shape[0], dim_data_x]),self.baseline_net(x[:,0:dim_data_x])))
        z_loc, z_scale = self.prior_net(x, y_hat)
        # Then sample the z's
        # z_dist.batch_shape = batch_size, z_dist.event_shape = dim_z 
        z_dist = dist.Normal(z_loc, z_scale).to_event(1)    
        with pyro.plate('batch_plate', size = batch_size, dim = -1):     
            # Broadcasting aligns to the left; leftmost dimension of samples from
            # z_dist are declared independent. Then pass z to generation_net.
            z = pyro.sample('latent_z', z_dist)
            xy_guess = self.generation_net(z)
         
            if y is not None:
                # When training, pass y as an observation that ties the y segment 
                # of the xy_guess to some actual data
                # To do this, 
                y_guess = xy_guess[(x == mask_with).view(-1,dim_data)].view(batch_size,-1) 
                y_extracted = y[(x == mask_with).view(-1,dim_data)].view(batch_size,-1) 
                dist_y = dist.Normal(y_guess, 0.01).to_event(1)
                pyro.sample('output_y', dist_y, obs = y_extracted)
            else:
                # When testing, just insert y_guess as a new y and deliver the
                # full datapoint (including x) for purposes of easier visualization.
                pyro.deterministic('output_y', xy_guess)
                
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
            # When testing, pass x and an estimate for y to the recognition_net to
            # generate a sample of z
            y_hat = torch.hstack((mask_with*torch.ones([x.shape[0], dim_data_x]),self.baseline_net(x[:,0:dim_data_x])))
            z_loc, z_scale = self.recognition_net(x,y_hat)
            z_dist = dist.Normal(z_loc, z_scale).to_event(1)
        
        # Sample z from the distribution
        with pyro.plate('batch_plate', size = batch_size, dim = -1):
            z = pyro.sample('latent_z', z_dist)
            
        return z


    # iv) Define some support and illustration functions

    def reconstruct_point(self, x):
        # encode datapoint x
        y_hat = torch.hstack((mask_with*torch.ones([x.shape[0], dim_data_x]),self.baseline_net(x[:,0:dim_data_x])))
        z_loc, z_scale = self.recognition_net(x,y_hat)
        z_dist = dist.Normal(z_loc, z_scale).to_event(1)
        # sample in latent space
        z = pyro.sample('latent_z', z_dist)
        # decode the latent code
        xy_guess = self.generation_net(z)
        return xy_guess
    

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

cvae_dataloader = torch.utils.data.DataLoader(cvae_dataset, batch_size=10, shuffle=True, num_workers=0) 
pyro.clear_param_store()


# ii) Training function

def train(svi, train_loader, use_cuda = False):
    # Initialize loss and cycle through batches
    loss_accumulator = 0
    
    for k, data in enumerate(train_loader):
        # Extract batches
        x_batch = data['input']
        y_batch = data['output']
        # pass minibatch to cuda
        if use_cuda == True:
            x_batch = x_batch.cuda()
        # Perform svi gradient step
        temp_loss = svi.step(x_batch, y_batch)
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
    
    for k, data in enumerate(test_loader):
        # Extract batches
        x_batch = data['input']
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
    epoch_loss = train(svi, train_loader)
    train_elbo.append(-epoch_loss)
    if epoch % 50 == 0:
        print("Epoch : {} cvae train loss : {}".format(epoch, epoch_loss))



"""
    6. Plots and illustrations
"""


# i) Illustrate baseline results via scatterplots

y_pred = baseline_net(x_data[:, 0:dim_data_x]).detach()

# Marginal covariances and means
mu_x = mu_true[0:dim_data_x]
mu_y = mu_true[dim_data_x:]
sigma_xx = sigma_true[0:dim_data_x, 0 :dim_data_x]
sigma_yy = sigma_true[dim_data_x:, dim_data_x:]
sigma_xy = sigma_true[0:dim_data_x, dim_data_x:]
sigma_yx = sigma_xy.T

# Conditional expectation
def conditional_y_given_x(x):
    batch_size = x.shape[0]
    y_hat = np.zeros(x.shape)
    sigma_y_hat = np.zeros([x.shape[0], dim_data_y, dim_data_y])
    for k in range(batch_size):    
        x_temp = x[k,:].squeeze()
        y_hat_temp =  mu_y + sigma_yx@np.linalg.pinv(sigma_xx)@(x_temp.detach().numpy() - mu_x)
        sigma_y_hat_temp = sigma_yy - sigma_yx@np.linalg.pinv(sigma_xx)@sigma_xy
        
        y_hat[k,:] = y_hat_temp.squeeze()
        sigma_y_hat[k, :,:] = sigma_y_hat_temp
    return y_hat, sigma_y_hat
y_hat_cond, _ = conditional_y_given_x(x_data[:,0:dim_data_x])

fig, axs = plt.subplots(3, figsize=(10, 10)) 

# Create scatterplots
axs[0].scatter(y_data[:,2], y_data[:,3])
axs[0].set_title('Original data')
axs[1].scatter(y_hat_cond[:,0], y_hat_cond[:,1])
axs[1].set_title('Conditional_expectation')
axs[2].scatter(y_pred[:,0], y_pred[:,1])
axs[2].set_title('ANN prediction')

plt.tight_layout()
plt.show()


# ii) Illustrate baseline results via lineplots
n_datapoints = 100
index_datapoint = np.linspace(0,n_datapoints-1, n_datapoints)
x_data_synthetic = torch.vstack((torch.zeros(n_datapoints), torch.linspace(-1,1,n_datapoints))).T
y_data_synthetic, _ = conditional_y_given_x(x_data_synthetic)
y_pred_synthetic = baseline_net(x_data_synthetic).detach()

# Create lineplots
fig, axs = plt.subplots(3, figsize=(10, 10)) 

axs[0].plot(index_datapoint, x_data_synthetic[:,0], label = 'x_1')
axs[0].plot(index_datapoint, x_data_synthetic[:,1], label = 'x_2')
axs[0].set_title('Synthetic x data')
axs[0].legend()
axs[1].plot(index_datapoint, y_data_synthetic[:,0], label = 'y_1')
axs[1].plot(index_datapoint, y_data_synthetic[:,1], label = 'y_2')
axs[1].set_title('Conditional y data')
axs[1].legend()
axs[2].plot(index_datapoint, y_pred_synthetic[:,0], label = 'y_1')
axs[2].plot(index_datapoint, y_pred_synthetic[:,1], label = 'y_2')
axs[2].set_title('ANN y predictions')
axs[2].legend()

plt.tight_layout()
plt.show()


# iii) Illustrate results of baseline net by showcasing distributions

# on old data
x_data_cut = x_data[:, 0:dim_data_x].detach()
y_data_cut = y_data[:, dim_data_x:].detach()
y_pred_cut = baseline_net(x_data_cut).detach()

# on unseen data
x_data_cut_unseen = x_data_unseen[:, 0:dim_data_x].detach()
y_data_cut_unseen = y_data_unseen[:, dim_data_x:].detach()
y_pred_cut_unseen = baseline_net(x_data_cut_unseen).detach()

# plot scatterplots
fig, axs = plt.subplots(2, 2, figsize=(10, 10)) # Create a 2x2 grid of Axes objects

# Create scatterplots
axs[0, 0].scatter(y_data_cut[:,0], y_data_cut[:,1])
axs[0, 0].set_title('Original data y')
axs[0, 1].scatter(y_pred_cut[:,0], y_pred_cut[:,1])
axs[0, 1].set_title('Predicted data y')
axs[1, 0].scatter(y_data_cut_unseen[:,0], y_data_cut_unseen[:,1])
axs[1, 0].set_title('Unseen data y')
axs[1, 1].scatter(y_pred_cut_unseen[:,0], y_pred_cut_unseen[:,1])
axs[1, 1].set_title('Predicted y on unseen data')
plt.tight_layout()
plt.show()


# iv) Compare conditional distributions
# Fix a value for x, then plot the true conditional distribution y|x and the 
# distribution of y|x encoded in the cvae
x_example = torch.tensor([0,1]).reshape([-1,2]).float()
y_example_cond, sigma_example_cond = conditional_y_given_x(x_example)
y_example_pred = cvae.baseline_net(x_example)

# Simulate from conditional distribution
n_simu = 1000
conditional_samples_y = np.zeros([n_simu,dim_data_y])
for k in range(n_simu):
    conditional_samples_y[k,:] = np.random.multivariate_normal(y_example_cond.squeeze(), sigma_example_cond.squeeze())

# Simulate from model
x_example_completed = torch.hstack((x_example, mask_with*torch.ones([1,dim_data_y])))
predictive = pyro.infer.Predictive(cvae.model, posterior_samples = None, guide = cvae.guide, num_samples = n_simu)
cvae_predictions = predictive(x_example_completed)["output_y"].squeeze().detach().numpy()[:,dim_data_x:]

cvae_samples_y = np.zeros([n_simu,dim_data_y])
for k in range(n_simu):
    cvae_samples_y[k,:] = cvae.model(x_example_completed).detach().numpy()[:,dim_data_x:]


# Plot marginal and joint empirical distributions of true conditionals and model outputs
fig, ax = plt.subplots(2, 2, figsize=(10, 10))

# 1D histograms for the first dimension
ax[0, 0].hist(conditional_samples_y[:, 0], bins=20, color='blue', alpha=0.5, label='y_conditional')
# ax[0, 0].hist(cvae_predictions[:, 0], bins=20, color='red', alpha=0.5, label='y_estimated')
ax[0, 0].hist(cvae_samples_y[:, 0], bins=20, color='green', alpha=0.5, label='y_model')
ax[0, 0].set_title('1D Histograms for dimension 1')
ax[0, 0].legend()

# 2D histogram for the first dimension
h = ax[0, 1].hist2d(conditional_samples_y[:, 0], conditional_samples_y[:, 1], bins=20, cmap='viridis')
ax[0, 1].set_title('2D Histogram for conditional distribution')
fig.colorbar(h[3], ax=ax[0, 1])

# 1D histograms for the second dimension
ax[1, 0].hist(conditional_samples_y[:, 1], bins=20, color='blue', alpha=0.5, label='y_conditional')
# ax[0, 0].hist(cvae_predictions[:, 1], bins=20, color='red', alpha=0.5, label='y_estimated')
ax[1, 0].hist(cvae_samples_y[:, 1], bins=20, color='green', alpha=0.5, label='y_model')
ax[1, 0].set_title('1D Histograms for dimension 2')
ax[1, 0].legend()

# 2D histogram for the second dimension
h = ax[1, 1].hist2d(cvae_samples_y[:, 1], cvae_samples_y[:, 1], bins=50, cmap='viridis')
ax[1, 1].set_title('2D Histogram for cvae samples')
fig.colorbar(h[3], ax=ax[1, 1])

plt.tight_layout()
plt.show()




# # i) Reconstruct data

# xy_data_rec = cvae.model(x_data).detach()


# # ii) Simulate new data 
# # Do it by exploring predictive methods - sampling new data x

# # Define new model for data prediction (without obs keyword)
# def predictive_model(x):
#     xy_guess = cvae.model(x)
    
#     return xy_guess

# x_data_new = predictive_model(x_data).detach().numpy()




# # iii) Illustrate via scatterplot

# fig, ax = plt.subplots(3, figsize=[10, 10], dpi=300)

# ax[0].scatter(x_data[:,0], x_data[:,1])
# ax[0].set_title('Original dataset')
# ax[0].set_xlabel('x-axis')
# ax[0].set_ylabel('y-axis')

# ax[1].scatter(xy_data_rec[:,0], xy_data_rec[:,1])
# ax[1].set_title('Reconstructed dataset')
# ax[1].set_xlabel('x-axis')
# ax[1].set_ylabel('y-axis')

# ax[2].scatter(x_data_new[:,0], x_data_new[:,1])
# ax[2].set_title('New dataset')
# ax[2].set_xlabel('x-axis')
# ax[2].set_ylabel('y-axis')

# plt.tight_layout()
# plt.show()


# # iv) Explore predictive methods - sampling from the posterior on z

# predictive = pyro.infer.Predictive(model = cvae.model, posterior_samples = None, guide = cvae.guide, num_samples = 1)
# # samples = predictive()
# samples = predictive(x_data)
# # samples = predictive.get_samples(x_data)
# samples_z = samples['latent_z'].squeeze()
# samples_x = samples['output_y'].squeeze()
# x_data_new = samples_x


# # Plot the predictive distributions
# fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# # Scatter plot of samples_z
# axes[0].scatter(samples_z[:, 0], samples_z[:, 1])
# axes[0].set_title('samples_z')
# axes[0].set_xlabel('Dimension 1')
# axes[0].set_ylabel('Dimension 2')

# # Scatter plot of samples_x
# axes[1].scatter(samples_x[:, 0], samples_x[:, 1])
# axes[1].set_title('samples_x')
# axes[1].set_xlabel('Dimension 1')
# axes[1].set_ylabel('Dimension 2')

# plt.tight_layout()
# plt.show()



