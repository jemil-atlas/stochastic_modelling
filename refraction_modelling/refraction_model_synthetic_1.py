#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The goal of this script is to showcase inference of random fields based on refraction
measurements. We aim to model the conversion from temperature, pressure, humidity
random fields to the refractivity random field and the subsequent conversion to
measured 3d point coordinates. Inference and modelling is done in pyro.
For this, do the following:
    1. Imports and definitions
    2. Generate synthetic data
    3. Build model
    4. Build guide
    5. Inference
    6. Plots and illustrations
"""

"""
    1. Imports and definitions
"""


# i) Imports

import pyro
import torch
import copy
import matplotlib.pyplot as plt
# torch.set_default_dtype(torch.float64)


# ii) Definitions

field_dim = 2

# gradbox dimension
len_x = 1   # in m
len_z = 0.2 # in m
n_x = 30
n_z = 10
n_pts = n_z*n_x
n_endpoints = 25

# x,z coordinates of gradbox
x = torch.linspace(0,len_x,n_x)
z = torch.linspace(0.5*len_z, -0.5*len_z, n_z)
xx,zz = torch.meshgrid(x,z, indexing = 'xy')
coords = torch.cat((zz.unsqueeze(-1),xx.unsqueeze(-1)), 2) # n_z, n_x, 2


"""
    2. Generate synthetic data
"""



# i) Synthetic ground truth

# Temperature in gradbox assumed linear [0,10] deg C [bottom, top], given in deg K
t_0 = 273.15
t = t_0 + 10 * (zz + 0.5*len_z)/len_z

# Relative humidity in gradbox assumed constant at 10%
# can be converted to water vapor pressure in hPa via Tetens equation
h = 10*torch.ones([n_z,n_x])

# Pressure in gradbox assumed constant at standard value 1013.25 hPa
p = 1013.25*torch.ones([n_z, n_x])


# ii) Refractive index

# equation for radiowaves
def n_Smith_and_Weintraub(t, p, h):
    # t, p, h are temperature, pressure, water vaport pressure fields and will 
    # be converted to a field of refractive index values n
    saturation_pressure = 6.11*torch.exp((17.27*(t-t_0)) /(t-t_0 + 237.3))  # Tetens equation
    e = h*saturation_pressure
    n_field = 1+ (1e-6)*(77.7/t)*(p + 4.81*1e3*e/t)
    return n_field

# equation for optical waves
def n_Ciddor(t, p, h):
    # t, p, h are temperature, pressure, humidity fields and will be converted
    # to a field of refractive index values n
    n_field = 0
    return n_field


# iii)  Line integration

# conversion of n_field to accumulated_refraction
n_disc = 10

def get_index(coords):
    # convert coordinate vector [z,x] to index for accessing n_field
    d_z = coords[0] - zz
    d_x = coords[1] - xx
    d_vec = torch.cat((d_z.unsqueeze(-1),d_x.unsqueeze(-1)),2)
    d_mat = torch.linalg.norm(d_vec, ord = 2, dim = 2)
    index = torch.tensor(torch.unravel_index(torch.argmin(d_mat), d_mat.shape)).unsqueeze(-1)
    return index

def build_grad_field(n_field):
    # Build the field of refractive index gradients for each point in the grad box

    # Derivatives in z direction
    dn_dz = torch.zeros_like(n_field)
    dn_dz[0:-1,:] = n_field[1:,:] - n_field[:-1,:]
    dn_dz[-1, :] = dn_dz[-2,:]
    
    # Derivatives in x direction
    dn_dx = torch.zeros_like(n_field)
    dn_dx[:,:-1] = n_field[:, 1:] - n_field[:,:-1]
    dn_dx[:, -1] = dn_dx[:, -2]
    
    
    # Concatenating to vector field
    nabla_n = torch.cat((dn_dz.unsqueeze(-1), dn_dx.unsqueeze(-1)), 2)
    
    return nabla_n
    
def accumulate_refraction(x_start, x_end, n_field):
    # Go along path x_start -> x_end and aggregate the angular refraction effects
    # by weighting dn/dq by step length and accumulating these values.
    
    # get indices
    n_endpoints = x_end.shape[1]
    delta_u = (x_end-x_start)/n_disc
    delta_u_norm = torch.linalg.norm(delta_u,2,0)
    u = torch.cat([(x_start + delta_u * k).unsqueeze(-1) for k in range(n_disc)],2)
    u = u.permute([0,2,1])      # dims : [zx, steps, endpoints]
    
    u_indices = torch.zeros([2,n_disc,n_endpoints])
    for k in range(n_disc):
        for l in range(n_endpoints):
            u_indices[:,k,l] = get_index(u[:, k, l]).flatten()
    u_indices = u_indices.long()
    
    # evaluate grads
    n_grads = build_grad_field(n_field)
    grad_vals = torch.zeros([n_disc, n_endpoints, 2])
    for k in range(n_disc):
        for l in range(n_endpoints):
            grad_vals[k,l,:] = n_grads[u_indices[0,k,l], u_indices[1,k,l],:]
    
    # accumulate
    delta_u_ortho = torch.vstack([-delta_u[1,:],delta_u[0,:]])/delta_u_norm
    beta_diff = torch.einsum('ve,dev->de', delta_u_ortho, grad_vals)
    refraction_angles = torch.cumsum(beta_diff, 0)
    beta = refraction_angles[-1,:]
    return refraction_angles, beta
    
# iv) Apply to geometry

x_start = torch.zeros([2,1])
# x_end = torch.tensor([[0,],[1]])
x_end = torch.vstack((torch.linspace(-0.5*len_z, 0.5*len_z, n_endpoints), torch.ones([1,n_endpoints])))
n_field = n_Smith_and_Weintraub(t, p, h)

_, beta_obs = accumulate_refraction(x_start, x_end, n_field)
t_obs = 1
p_obs = 1
h_obs = 1


"""
    3. Build model
"""


# i) Support functions

def build_cov_mat(coords, sigma, bandwidth):

    n_z, n_x, _ = coords.shape
    n_pts = n_z * n_x

    # Flatten coords into shape (N, 2)
    coords_flat = coords.reshape(n_pts, 2)

    # Compute pairwise distances:
    # delta shape: (N, N, 2)
    delta = coords_flat.unsqueeze(1) - coords_flat.unsqueeze(0)
    # dist shape: (N, N)
    dist = torch.linalg.norm(delta, dim=2)

    # Compute covariance matrix in flattened form
    cov_mat_flat = (sigma**2) * torch.exp(- (dist / bandwidth)**2)

    # # Reshape back to (n_z, n_x, n_z, n_x)
    # cov_mat = cov_mat_flat.reshape(n_z, n_x, n_z, n_x)
    return cov_mat_flat 

# ii) Set up the priors

# temperature
def t_prior():
    t_mu_prior = 273.15*torch.ones([1,]) + 10
    t_sigma_prior = 10*torch.ones([1,])
    t_bw_prior = torch.ones([1,])
    # t_mu = pyro.param('t_mu', 273.15*torch.ones([1,]))
    # t_sigma = pyro.param('t_sigma', torch.ones([1,]), pyro.distributions.constraints.positive)
    # t_bw = pyro.param('t_bw', torch.ones([1,]), pyro.distributions.constraints.positive)
    
    t_cov_mat = build_cov_mat(coords, t_sigma_prior, t_bw_prior) + 1e-3*torch.eye(n_pts)
    t_dist = pyro.distributions.MultivariateNormal(loc = t_mu_prior*torch.ones([n_pts]), covariance_matrix = t_cov_mat)
    sample = pyro.sample('t_sample', t_dist)
    sample_reshaped = sample.reshape([n_z,n_x])
    return sample_reshaped
    
# pressure
def p_prior():
    p_mu_prior = 1013.25*torch.ones([1,])
    p_sigma_prior = 100*torch.ones([1,])
    p_bw_prior = torch.ones([1,])
    # p_mu = pyro.param('p_mu', 1013.25*torch.ones([1,]))
    # p_sigma = pyro.param('p_sigma', torch.ones([1,]), pyro.distributions.constraints.positive)
    # p_bw = pyro.param('p_bw', torch.ones([1,]), pyro.distributions.constraints.positive)
    
    p_cov_mat = build_cov_mat(coords, p_sigma_prior, p_bw_prior) + 1e1*torch.eye(n_pts)
    p_dist = pyro.distributions.MultivariateNormal(loc = p_mu_prior*torch.ones([n_pts]), covariance_matrix = p_cov_mat)
    sample = pyro.sample('p_sample', p_dist)
    sample_reshaped = sample.reshape([n_z,n_x])
    return sample_reshaped

# humidity
def h_prior():
    h_mu_prior = 10*torch.ones([1,])
    h_sigma_prior = 30*torch.ones([1,])
    h_bw_prior = torch.ones([1,])
    # h_mu = pyro.param('h_mu', 10*torch.ones([1,]))
    # h_sigma = pyro.param('h_sigma', torch.ones([1,]), pyro.distributions.constraints.positive)
    # h_bw = pyro.param('h_bw', torch.ones([1,]), pyro.distributions.constraints.positive)
    
    h_cov_mat = build_cov_mat(coords, h_sigma_prior, h_bw_prior) + 1e-1*torch.eye(n_pts)
    h_dist = pyro.distributions.MultivariateNormal(loc = h_mu_prior*torch.ones([n_pts]), covariance_matrix = h_cov_mat)
    sample = pyro.sample('h_sample', h_dist)
    sample_reshaped = sample.reshape([n_z,n_x])
    return sample_reshaped



# iii) Build the pyro model

def model(observations = None):    
    
    t_field = t_prior()
    p_field = p_prior()
    h_field = h_prior()
    
    n_field = n_Smith_and_Weintraub(t_field, p_field, h_field)

    sigma_meas = 1e-6
    _, refraction_integrals = accumulate_refraction(x_start, x_end, n_field)
    ref_dist = pyro.distributions.Normal(loc = refraction_integrals, scale = sigma_meas)
    
    with pyro.plate('endpoint_plate', n_endpoints, dim = -1):
        ref_sample = pyro.sample('ref_sample', ref_dist, obs = observations)
    return ref_sample

def conditioned_model(t_sample, p_sample, h_sample):
    cmodel = pyro.condition(model, {'t_sample': t_sample, 'p_sample' : p_sample, 'h_sample': h_sample})
    return_val = cmodel()
    return return_val


"""
    4. Build guide
"""

# i) Set up the posteriors

# temperature
def t_posterior():
    t_mu = pyro.param('t_mu', 273.15*torch.ones([n_pts]))
    # t_sigma = pyro.param('t_sigma', torch.eye(n_pts), pyro.distributions.constraints.positive_definite)
    # t_dist = pyro.distributions.MultivariateNormal(loc = t_mu, covariance_matrix = t_sigma)
    
    t_sigma = pyro.param('t_sigma', 10*torch.ones([1,]), pyro.distributions.constraints.positive)
    t_bw = pyro.param('t_bw', 0.3*torch.ones([1,]), pyro.distributions.constraints.positive)
    t_cov_mat = build_cov_mat(coords, t_sigma, t_bw) + 1e-3*torch.eye(n_pts)
    t_dist = pyro.distributions.MultivariateNormal(loc = t_mu, covariance_matrix = t_cov_mat)
    sample = pyro.sample('t_sample', t_dist)
    sample_reshaped = sample.reshape([n_z,n_x])
    return sample_reshaped, t_mu.reshape([n_z,n_x]), t_cov_mat
    
# pressure
def p_posterior():
    p_mu = pyro.param('p_mu', 1013.25*torch.ones([n_pts]))
    # p_sigma = pyro.param('p_sigma', torch.eye(n_pts), pyro.distributions.constraints.positive_definite)
    # p_dist = pyro.distributions.MultivariateNormal(loc = p_mu*torch.ones([n_pts]), covariance_matrix = p_sigma)
    
    p_sigma = pyro.param('p_sigma', 100*torch.ones([1,]), pyro.distributions.constraints.positive)
    p_bw = pyro.param('p_bw', 0.3*torch.ones([1,]), pyro.distributions.constraints.positive)
    p_cov_mat = build_cov_mat(coords, p_sigma, p_bw) + 1e1*torch.eye(n_pts)
    p_dist = pyro.distributions.MultivariateNormal(loc = p_mu*torch.ones([n_pts]), covariance_matrix = p_cov_mat)
    sample = pyro.sample('p_sample', p_dist)
    sample_reshaped = sample.reshape([n_z,n_x])
    return sample_reshaped, p_mu.reshape([n_z,n_x]), p_cov_mat

# humidity
def h_posterior():
    h_mu = pyro.param('h_mu', 10*torch.ones([n_pts]))
    # h_sigma = pyro.param('h_sigma', torch.eye(n_pts), pyro.distributions.constraints.positive_definite)
    # h_dist = pyro.distributions.MultivariateNormal(loc = h_mu*torch.ones([n_pts]), covariance_matrix = h_sigma)
    
    h_sigma = pyro.param('h_sigma', 10*torch.ones([1,]), pyro.distributions.constraints.positive)
    h_bw = pyro.param('h_bw', 0.3*torch.ones([1,]), pyro.distributions.constraints.positive)
    h_cov_mat = build_cov_mat(coords, h_sigma, h_bw)  + 1e-1*torch.eye(n_pts)
    h_dist = pyro.distributions.MultivariateNormal(loc = h_mu*torch.ones([n_pts]), covariance_matrix = h_cov_mat)
    sample = pyro.sample('h_sample', h_dist)
    sample_reshaped = sample.reshape([n_z,n_x])
    return sample_reshaped, h_mu.reshape([n_z,n_x]), h_cov_mat



# ii) Build the pyro guide

def guide(observations = None):
    t_result = t_posterior() # t_result = (t_field, mu_t, sigma_t)
    p_result = p_posterior()
    h_result = h_posterior()
    
    return t_result, p_result, h_result


# iii) Evaluate ingredients

t_prior_sample = t_prior()
p_prior_sample = p_prior()
h_prior_sample = h_prior()
n_prior_sample = n_Smith_and_Weintraub(t_prior_sample, p_prior_sample, h_prior_sample)

t_pretrain, p_pretrain, h_pretrain = copy.deepcopy(guide())
n_pretrain = n_Smith_and_Weintraub(t_pretrain[0], p_pretrain[0], h_pretrain[0])
ref_pretrain = conditioned_model(t_pretrain[0], p_pretrain[0], h_pretrain[0])


"""
    5. Inference
"""


# i) Set up inference


adam = pyro.optim.Adam({"lr": 1e-1})
elbo = pyro.infer.Trace_ELBO(num_particles = 1)
svi = pyro.infer.SVI(model, guide, adam, elbo)


# ii) Perform svi

losses = []
n_steps = 50
data_svi = beta_obs
for step in range(n_steps):
    loss = svi.step(data_svi)
    losses.append(loss)
    if step % 10 == 0:
        print('Step: {}/{} Loss : {} '.format(step, n_steps, loss))

t_posttrain, p_posttrain, h_posttrain = copy.deepcopy(guide())
n_posttrain = n_Smith_and_Weintraub(t_posttrain[0], p_posttrain[0], h_posttrain[0])
ref_posttrain = conditioned_model(t_posttrain[0], p_posttrain[0], h_posttrain[0])




"""
    6. Plots and illustrations
"""


# i) Plot of ground truth

fig, axes = plt.subplots(2, 2, figsize=(10,8))

im0 = axes[0, 0].imshow(t.numpy(), cmap='viridis', aspect='auto')
axes[0, 0].set_title("Temperature in deg C")
fig.colorbar(im0, ax=axes[0, 0])

im1 = axes[0, 1].imshow(p.numpy(), cmap='viridis', aspect='auto')
axes[0, 1].set_title("Pressure in hPa")
fig.colorbar(im1, ax=axes[0, 1])

im2 = axes[1, 0].imshow(h.numpy(), cmap='viridis', aspect='auto')
axes[1, 0].set_title("Relative Humidity in %")
fig.colorbar(im2, ax=axes[1, 0])

im3 = axes[1, 1].imshow(n_field.numpy(), cmap='viridis', aspect='auto')
axes[1, 1].set_title("Refraction Coefficient")
fig.colorbar(im3, ax=axes[1, 1])

plt.tight_layout()
plt.show()


# ii) Optimization progress

plt.figure(1, dpi = 300)
plt.plot(losses)
plt.xlabel('n_step')
plt.ylabel('ELBO')


# iii) Plots of priors, posteriors

fig, axes = plt.subplots(4, 3, figsize=(15,20))

im0 = axes[0, 0].imshow(t.numpy(), cmap='viridis', aspect='auto')
axes[0, 0].set_title("Ground truth: t")
# fig.colorbar(im0, ax=axes[0, 0])
im1 = axes[1, 0].imshow(p.numpy(), cmap='viridis', aspect='auto')
axes[1, 0].set_title("Ground truth: p")
# fig.colorbar(im0, ax=axes[1, 0])
im2 = axes[2, 0].imshow(h.numpy(), cmap='viridis', aspect='auto')
axes[2, 0].set_title("Ground truth: h")
# fig.colorbar(im0, ax=axes[2, 0])
im3 = axes[3, 0].imshow(n_field.numpy(), cmap='viridis', aspect='auto')
axes[3, 0].set_title("Ground truth: n")
# fig.colorbar(im0, ax=axes[3, 0])

# Pretrain
im0 = axes[0, 1].imshow(t_pretrain[0].detach().numpy(), cmap='viridis', aspect='auto')
axes[0, 1].set_title("Pretrain samples: t")
# fig.colorbar(im0, ax=axes[0, 0])
im1 = axes[1, 1].imshow(p_pretrain[0].detach().numpy(), cmap='viridis', aspect='auto')
axes[1, 1].set_title("Pretrain samples: p")
# fig.colorbar(im0, ax=axes[1, 0])
im2 = axes[2, 1].imshow(h_pretrain[0].detach().numpy(), cmap='viridis', aspect='auto')
axes[2, 1].set_title("Pretrain samples: h")
# fig.colorbar(im0, ax=axes[2, 0])
im3 = axes[3, 1].imshow(n_pretrain.detach().numpy(), cmap='viridis', aspect='auto')
axes[3, 1].set_title("Pretrain samples: n")
# fig.colorbar(im0, ax=axes[3, 0])

# Posttrain
im0 = axes[0, 2].imshow(t_posttrain[0].detach().numpy(), cmap='viridis', aspect='auto')
axes[0, 1].set_title("Posttrain samples: t")
# fig.colorbar(im0, ax=axes[0, 0])
im1 = axes[1, 2].imshow(p_posttrain[0].detach().numpy(), cmap='viridis', aspect='auto')
axes[1, 1].set_title("Posttrain samples: p")
# fig.colorbar(im0, ax=axes[1, 0])
im2 = axes[2, 2].imshow(h_posttrain[0].detach().numpy(), cmap='viridis', aspect='auto')
axes[2, 1].set_title("Posttrain samples: h")
# fig.colorbar(im0, ax=axes[2, 0])
im3 = axes[3, 2].imshow(n_posttrain.detach().numpy(), cmap='viridis', aspect='auto')
axes[3, 1].set_title("Posttrain samples: n")
# fig.colorbar(im0, ax=axes[3, 0])

plt.tight_layout()
plt.show()


# iv) Plots of params

fig, axes = plt.subplots(3, 4, figsize=(15,20))

# Pretrain
im0 = axes[0, 0].imshow(t_pretrain[1].detach().numpy(), cmap='viridis', aspect='auto')
axes[0, 0].set_title("PRETRAIN mu: t ")
im1 = axes[1, 0].imshow(p_pretrain[1].detach().numpy(), cmap='viridis', aspect='auto')
axes[1, 0].set_title("mu: p")
im2 = axes[2, 0].imshow(h_pretrain[1].detach().numpy(), cmap='viridis', aspect='auto')
axes[2, 0].set_title("mu: h")

im0 = axes[0, 1].imshow(t_pretrain[2].detach().numpy(), cmap='viridis', aspect='auto')
axes[0, 1].set_title("PRETRAIN sigma: t")
im1 = axes[1, 1].imshow(p_pretrain[2].detach().numpy(), cmap='viridis', aspect='auto')
axes[1, 1].set_title("sigma: p")
im2 = axes[2, 1].imshow(h_pretrain[2].detach().numpy(), cmap='viridis', aspect='auto')
axes[2, 1].set_title("sigma: h")

# Posttrain
im0 = axes[0, 2].imshow(t_posttrain[1].detach().numpy(), cmap='viridis', aspect='auto')
axes[0, 2].set_title("POSTTRAIN mu: t")
im1 = axes[1, 2].imshow(p_posttrain[1].detach().numpy(), cmap='viridis', aspect='auto')
axes[1, 2].set_title("mu: p")
im2 = axes[2, 2].imshow(h_posttrain[1].detach().numpy(), cmap='viridis', aspect='auto')
axes[2, 2].set_title("mu: h")

im0 = axes[0, 3].imshow(t_posttrain[2].detach().numpy(), cmap='viridis', aspect='auto')
axes[0, 3].set_title("POSTTRAIN sigma: t")
im1 = axes[1, 3].imshow(p_posttrain[2].detach().numpy(), cmap='viridis', aspect='auto')
axes[1, 3].set_title("sigma: p")
im2 = axes[2, 3].imshow(h_posttrain[2].detach().numpy(), cmap='viridis', aspect='auto')
axes[2, 3].set_title("sigma: h")

plt.tight_layout()
plt.show()


# v) Plot of observation fit

print('ref_true', beta_obs)
print('ref_pretrain', ref_pretrain)
print('ref_posttrain', ref_posttrain)

fig, axes = plt.subplots(3, 1, figsize=(10,10))
plt0 = axes[0].plot(beta_obs.detach())
axes[0].set_title("observed refraction ")
plt1 = axes[1].plot(ref_pretrain.detach())
axes[1].set_title("explained refraction pretrain")
plt2 = axes[2].plot(ref_posttrain.detach())
axes[2].set_title("explained refraction posttrain")
axes[2].set_xlabel('endpoint nr')


plt.tight_layout()
plt.show()