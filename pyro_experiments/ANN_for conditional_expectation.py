#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The goal of this script is to train a neural network to recreate the conditional
expectation of a multivariate Gaussian. 
For this, do the following:
    1. Definitions and imports
    2. Simulate data and package it
    3. Define Baseline Net and train
    4. Compare to conditional expectation
"""

"""
    1. Definitions and imports
"""


# i) Imports

import numpy as np
import pyro
import torch
import matplotlib.pyplot as plt


# ii) Definitions

n_data = 100
dim_data_x = 2
dim_data_y = 2
dim_data = dim_data_x + dim_data_y
dim_z = 2
dim_hidden = 20
use_cuda = False


"""
    2. Simulate data and package it
"""


# i) Draw from multivariate gaussian

mu_true = np.zeros([dim_data])
sigma_true = np.eye(dim_data)
np.fill_diagonal(sigma_true[1:], 0.5)  
np.fill_diagonal(sigma_true[:, 1:], 0.5)  

xy_data_prel = np.random.multivariate_normal(mu_true,sigma_true, size = [n_data])
xy_data = torch.tensor(xy_data_prel).float()
# nonlinear_fun = lambda x : x
nonlinear_fun = lambda x : torch.tensor([x[:,0],x[:,1],x[:,1],x[:,1]**2]).T
# nonlinear_fun = lambda x : torch.tensor([x[:,0], x[:,1], x[:,0]**2, x[:,1]**2]).T
xy_data = torch.tensor(nonlinear_fun(xy_data_prel)).float()

# iii) Create new dataset

x_data = xy_data[:, 0:dim_data_x]
y_data = xy_data[:, dim_data_x:]

tensor_data = torch.utils.data.TensorDataset(x_data, y_data)
train_loader = torch.utils.data.DataLoader(tensor_data, batch_size = 32)



"""
    3. Define Baseline Net and train
"""


# i) Build baseline_net

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


# ii) Initialize the model and optimizer
baseline_net = BaselineNet(dim_data_x, dim_hidden, dim_data_y)
optimizer = torch.optim.Adam(baseline_net.parameters(), lr=0.001)

# Define a loss function
criterion = torch.nn.MSELoss()

# iii) Train the model
epochs = 1000
for epoch in range(epochs):
    for inputs, targets in train_loader:
        # Forward pass
        outputs = baseline_net(inputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Print the loss for this epoch
    if epoch % 50 == 0:
        print("Epoch {} / {} Loss : {}".format(epoch, epochs, loss.item()))

y_pred = baseline_net(x_data).detach()



"""
    4. Compare to conditional expectation
"""


# i) Inference via conditional expectation

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
    for k in range(batch_size):    
        x_temp = x[k,:].squeeze()
        y_hat_temp =  mu_y + sigma_yx@np.linalg.pinv(sigma_xx)@(x_temp.detach().numpy() - mu_x)
        y_hat[k,:] = y_hat_temp.squeeze()
    return y_hat
y_hat_cond = conditional_y_given_x(x_data)


# ii) Illustrate conditional expectation and baseline on scatterplots
fig, axs = plt.subplots(3, figsize=(10, 10)) 

# Create scatterplots
axs[0].scatter(y_data[:,0], y_data[:,1])
axs[0].set_title('Original data')
axs[1].scatter(y_hat_cond[:,0], y_hat_cond[:,1])
axs[1].set_title('Conditional_expectation')
axs[2].scatter(y_pred[:,0], y_pred[:,1])
axs[2].set_title('ANN prediction')

plt.tight_layout()
plt.show()


# iii) Illustrate conditional expectation and baseline on lineplots
n_datapoints = 100
index_datapoint = np.linspace(0,n_datapoints-1, n_datapoints)
x_data_synthetic = torch.vstack((torch.zeros(n_datapoints), torch.linspace(-1,1,n_datapoints))).T
y_data_synthetic = conditional_y_given_x(x_data_synthetic)
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



























