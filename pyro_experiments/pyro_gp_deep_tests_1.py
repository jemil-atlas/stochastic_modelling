#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The goal of this script is to show how a deep gp can be trained to learn 
nonnormally distributed data. The data does not yet contain samples from a 
stochastic process, they are independent realizations of a distribution whose
parameters depend on an index set.
For this, do the following:
    1. Imports and definitions
    2. Generate Data
    3. Set up deep GP
    4. Training
    5. Plots and illustrations
--> It seems my custom model architecture does not really work. At least the 
fitting does not produce something that looks significantly non-gaussian.
"""


"""
    1. Imports and definitions
"""


# i) Imports

import pyro
import pyro.distributions as dist
import pyro.optim as optim
import pyro.infer
import torch
import matplotlib.pyplot as plt


# ii) Definitions

dim_data = 1
n_samples = 1000

dim_input = 1
dim_hidden = 1
dim_index = 1
dim_cov_features = 1
dim_output = 1


# Clear param store
pyro.clear_param_store()




"""
    2. Generate Data
"""


# i) Generate synthetic data

data = torch.rand(n_samples).reshape([-1,1])
t = torch.linspace(0,0,n_samples)



"""
    3. Set up deep GP
"""


# i) Generic ANN mapping class

class ANN(pyro.nn.PyroModule):
    # Initialize module
    def __init__(self, dim_input, dim_hidden, dim_output):
        # Initialize instance using init method from base class
        super().__init__()
        
        # dimensions
        self.dim_input = dim_input
        self.dim_hidden = dim_hidden
        self.dim_output = dim_output
        
        # linear transforms
        self.fc_1 = torch.nn.Linear(dim_input, dim_hidden)
        self.fc_2 = torch.nn.Linear(dim_hidden, dim_hidden)
        self.fc_3 = torch.nn.Linear(dim_hidden, dim_output)
        # nonlinear transforms
        self.nonlinear = torch.nn.ReLU()

    def forward(self, x):
        # Define forward computation on the input data x
        # Shape the minibatch so that batch_dims are on left, argument_dims on right
        x = x.reshape([-1, self.dim_input])
        
        # Then compute hidden units and output of nonlinear pass
        hidden_units_1 = self.nonlinear(self.fc_1(x))
        hidden_units_2 = self.nonlinear(self.fc_2(hidden_units_1))
        output = self.fc_3(hidden_units_2)

        return output


# ii) Generic GPlayer class

class GPLayer(pyro.nn.PyroModule):
    # Initialize module
    def __init__(self, layer_name, dim_input, dim_hidden, dim_index, dim_cov_features, dim_output):
        # Initialize instance using init method from base class
        super().__init__()
        
        # dimensions
        self.dim_input = dim_input
        self.dim_hidden = dim_hidden
        self.dim_index = dim_index
        self.dim_cov_features = dim_cov_features
        self.dim_output = dim_output
        self.layer_name = layer_name
        
        # create & register distributional maps
        # dimensions are worthy of explanation. 
        # index map psi: 
        # Result is of shape [n_simu(batch), dim_output, dim_index] with dim_index the
        # dimension of the internal index passed to the fortet map. The whole set
        # of internal indices corresponding to an input is generated at once, 
        # they are therefore folly attributed to the ann output and not to the batch
        # dimension. Ann output dims needs to be reshaped since input/output dims
        # must be integers.
        # fortet map phi:
        # Result is of shape [n_simu(batch), dim_output, dim_cov_features] with dim_cov_features the
        # dimension of the covariace features passed to the cone map. The covariance
        # features generated via phi are are generated for each index separately and
        # there exists only a single map phi that takes an index of dim dim_index
        # and maps it to a feature vector of dim dim_cov_features. Ann output dims 
        # needs to be reshaped since input/output dims must be integers.
        self.cov_psi = ANN(self.dim_input, self.dim_hidden, self.dim_output*self.dim_index)   # index_map
        self.cov_phi = ANN(self.dim_index, self.dim_hidden, self.dim_cov_features)   # fortet_map
        self.ann_mean = ANN(self.dim_input, self.dim_hidden, self.dim_output)
        self.map_to_mean = self.map_to_mean
        self.map_to_cov = self.map_to_cov
        # pyro.module("map_to_mean_{}".format(layer_name), self.map_to_mean)   
        # pyro.module("map_to_cov_{}".format(layer_name), self.map_to_cov)  
        
    def map_to_mean(self, layer_input):
        pred_mean = self.ann_mean(layer_input)
        return pred_mean
    
    def map_to_cov(self, layer_input):
        # Build psi and reshape such that phi acts on dim_index as event_dim
        psi = self.cov_psi(layer_input)
        psi = psi.reshape([-1, self.dim_output, self.dim_index])
        psi = psi.reshape([-1, self.dim_index])
        # Build phi and reshape such that cone_map can produce covariance matrices
        phi = self.cov_phi(psi)
        phi = phi.reshape([-1, self.dim_output, self.dim_cov_features])
        phi_T = phi.permute(0,2,1)
        # Cone map to build covariance matrices
        pred_cov = torch.bmm(phi, phi_T)
        return pred_cov
        

# iii) Functions for generating observations
# Function for a single GP layer
def gp_layer(x, layer_name, dim_input, dim_output):
    # Evoke gp object and calculate mans, covariances
    gp = GPLayer(layer_name, dim_input, dim_hidden, dim_index, dim_cov_features, dim_output)
    mean_tensor = gp.map_to_mean(x)
    cov_tensor = gp.map_to_cov(x)
    cov_regularizer = 1e-3*(torch.eye(cov_tensor.shape[1]).repeat(cov_tensor.shape[0], 1, 1))

    # Build distribution and sample from it
    layer_dist = dist.MultivariateNormal(mean_tensor, cov_tensor + cov_regularizer)
    observation = pyro.sample("obs_{}".format(layer_name), layer_dist)
    return observation

# Pyro model for Deep GP
def deep_gp_model(input_data, observations = None):

    with pyro.plate('batch_plate', size = input_data.shape[0], dim = -1):
        z1 = gp_layer(input_data, "layer_1", 1, 2)
        z2 = gp_layer(z1, "layer_2", 2, 2)
        z3 = gp_layer(z2, "layer_3", 2, 1)
        obs_dist = dist.Normal(z3, 0.01).to_event(1)
        obs = pyro.sample("obs", obs_dist, obs=observations)
    return obs

# # Trivial Guide
# def guide(input_data, observations = None):
#     pass

# Autoguide
guide = pyro.infer.autoguide.AutoDiagonalNormal(deep_gp_model)

# Pre-train data generation
pre_train_data = deep_gp_model(t)



"""
    4. Training
"""



# i) Setup SVI

optimizer = optim.ClippedAdam({"lr": 0.01})
svi = pyro.infer.SVI(model=deep_gp_model, guide=guide, optim=optimizer, loss=pyro.infer.Trace_ELBO())


# ii) Train the model

losses = []
num_epochs = 300
for epoch in range(num_epochs):
    loss = svi.step(t, data)
    losses.append(loss)
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")


# iii) Generate new data

post_train_data = deep_gp_model(t)



"""
    5. Plots and illustrations
"""




# Plotting the loss
plt.figure(figsize=(10, 6))
plt.plot(losses)
plt.title("ELBO Loss During Training")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()


# # Plotting
# plt.figure(figsize=(10, 5))
# plt.title("Deep Gaussian Process pre-train")
# plt.hist(data.detach().numpy())
# plt.legend()
# plt.show()





# Create a figure with 5 vertically aligned subplots
fig, axes = plt.subplots(3, 1, figsize=(10, 15))

# Plot the line plots
axes[0].hist(data.detach().numpy())
axes[0].set_title("Original_data")

axes[1].hist(pre_train_data.detach().numpy())
axes[1].set_title("GP_pretraining")

axes[2].hist(post_train_data.detach().numpy())
axes[2].set_title("GP_posttraining")

# Make layout tight
plt.tight_layout()
plt.show()



# # ii) Plot distributional_parameters

# mu_gp = pyro.get_param_store()['mu_gp']
# sigma_gp = pyro.get_param_store()['sigma_gp']

# fig, axes = plt.subplots(5, 1, figsize=(10, 15))

# # Plot the line plots
# axes[0].plot(t, mu_gp.detach())
# axes[0].set_title("GP_mean")

# axes[1].imshow(sigma_gp.detach())
# axes[1].set_title("GP_covariance")


























