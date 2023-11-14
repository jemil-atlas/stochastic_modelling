#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 10:24:08 2023

@author: jemil
"""

import os
import matplotlib.pyplot as plt
import torch
import numpy as np


import pyro
import pyro.contrib.gp as gp
import pyro.distributions as dist

from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable

import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


smoke_test = "CI" in os.environ  # ignore; used to check code integrity in the Pyro repo
# assert pyro.__version__.startswith('1.8.6')
pyro.set_rng_seed(0)
torch.set_default_tensor_type(torch.DoubleTensor)


# note that this helper function does three different things:
# (i) plots the observed data;
# (ii) plots the predictions from the learned GP after conditioning on data;
# (iii) plots samples from the GP prior (with no conditioning on observed data)


def plot(
    plot_observed_data=False,
    plot_predictions=False,
    n_prior_samples=0,
    model=None,
    kernel=None,
    n_test=500,
    ax=None,
):

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    if plot_observed_data:
        ax.plot(X.numpy(), y.numpy(), "kx")
    if plot_predictions:
        Xtest = torch.linspace(-0.5, 5.5, n_test)  # test inputs
        # compute predictive mean and variance
        with torch.no_grad():
            if type(model) == gp.models.VariationalSparseGP:
                mean, cov = model(Xtest, full_cov=True)
            else:
                mean, cov = model(Xtest, full_cov=True, noiseless=False)
        sd = cov.diag().sqrt()  # standard deviation at each input point x
        ax.plot(Xtest.numpy(), mean.numpy(), "r", lw=2)  # plot the mean
        ax.fill_between(
            Xtest.numpy(),  # plot the two-sigma uncertainty about the mean
            (mean - 2.0 * sd).numpy(),
            (mean + 2.0 * sd).numpy(),
            color="C0",
            alpha=0.3,
        )
    if n_prior_samples > 0:  # plot samples from the GP prior
        Xtest = torch.linspace(-0.5, 5.5, n_test)  # test inputs
        noise = (
            model.noise
            if type(model) != gp.models.VariationalSparseGP
            else model.likelihood.variance
        )
        cov = kernel.forward(Xtest) + noise.expand(n_test).diag()
        samples = dist.MultivariateNormal(
            torch.zeros(n_test), covariance_matrix=cov
        ).sample(sample_shape=(n_prior_samples,))
        ax.plot(Xtest.numpy(), samples.numpy().T, lw=2, alpha=0.4)

    ax.set_xlim(-0.5, 5.5)


N = 20
X = dist.Uniform(0.0, 5.0).sample(sample_shape=(N,))
y = 0.5 * torch.sin(3 * X) + dist.Normal(0.0, 0.2).sample(sample_shape=(N,))

plot(plot_observed_data=True)  # let's plot the observed data



kernel = gp.kernels.RBF(
    input_dim=1, variance=torch.tensor(6.0), lengthscale=torch.tensor(0.05)
)
gpr = gp.models.GPRegression(X, y, kernel, noise=torch.tensor(0.2))

plot(model=gpr, kernel=kernel, n_prior_samples=2)
_ = plt.ylim((-8, 8))

kernel2 = gp.kernels.RBF(
    input_dim=1, variance=torch.tensor(6.0), lengthscale=torch.tensor(1)
)
gpr2 = gp.models.GPRegression(X, y, kernel2, noise=torch.tensor(0.2))
plot(model=gpr2, kernel=kernel2, n_prior_samples=2)
_ = plt.ylim((-8, 8))


kernel3 = gp.kernels.RBF(
    input_dim=1, variance=torch.tensor(1.0), lengthscale=torch.tensor(1)
)
gpr3 = gp.models.GPRegression(X, y, kernel3, noise=torch.tensor(0.01))
plot(model=gpr3, kernel=kernel3, n_prior_samples=2)
_ = plt.ylim((-8, 8))

optimizer = torch.optim.Adam(gpr.parameters(), lr=0.005)
loss_fn = pyro.infer.Trace_ELBO().differentiable_loss
losses = []
variances = []
lengthscales = []
noises = []
num_steps = 2000 if not smoke_test else 2
for i in range(num_steps):
    variances.append(gpr.kernel.variance.item())
    noises.append(gpr.noise.item())
    lengthscales.append(gpr.kernel.lengthscale.item())
    optimizer.zero_grad()
    loss = loss_fn(gpr.model, gpr.guide)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    
    
    # let's plot the loss curve after 2000 steps of training
def plot_loss(loss):
    plt.plot(loss)
    plt.xlabel("Iterations")
    _ = plt.ylabel("Loss")  # supress output text


plot_loss(losses)
plot(model=gpr, plot_observed_data=True, plot_predictions=True)

gpr.kernel.variance.item()
gpr.kernel.lengthscale.item()
gpr.noise.item()

fig, ax = plt.subplots(figsize=(12, 6))


def update(iteration):
    pyro.clear_param_store()
    ax.cla()
    kernel_iter = gp.kernels.RBF(
        input_dim=1,
        variance=torch.tensor(variances[iteration]),
        lengthscale=torch.tensor(lengthscales[iteration]),
    )
    gpr_iter = gp.models.GPRegression(
        X, y, kernel_iter, noise=torch.tensor(noises[iteration])
    )
    plot(model=gpr_iter, plot_observed_data=True, plot_predictions=True, ax=ax)
    ax.set_title(f"Iteration: {iteration}, Loss: {losses[iteration]:0.2f}")


# anim = FuncAnimation(fig, update, frames=np.arange(0, num_steps, 30), interval=100)
# plt.close()

# anim.save("../source/_static/img/gpr-fit.gif", fps=60)

# Define the same model as before.
pyro.clear_param_store()
kernel = gp.kernels.RBF(
    input_dim=1, variance=torch.tensor(5.0), lengthscale=torch.tensor(10.0)
)
gpr = gp.models.GPRegression(X, y, kernel, noise=torch.tensor(1.0))

# note that our priors have support on the positive reals
gpr.kernel.lengthscale = pyro.nn.PyroSample(dist.LogNormal(0.0, 1.0))
gpr.kernel.variance = pyro.nn.PyroSample(dist.LogNormal(0.0, 1.0))

optimizer = torch.optim.Adam(gpr.parameters(), lr=0.005)
loss_fn = pyro.infer.Trace_ELBO().differentiable_loss
losses = []
num_steps = 2000 if not smoke_test else 2
for i in range(num_steps):
    optimizer.zero_grad()
    loss = loss_fn(gpr.model, gpr.guide)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

plot_loss(losses)

plot(model=gpr, plot_observed_data=True, plot_predictions=True)
# tell gpr that we want to get samples from guides
gpr.set_mode("guide")
print("variance = {}".format(gpr.kernel.variance))
print("lengthscale = {}".format(gpr.kernel.lengthscale))
print("noise = {}".format(gpr.noise))


# Sparse GP

N = 1000
X = dist.Uniform(0.0, 5.0).sample(sample_shape=(N,))
y = 0.5 * torch.sin(3 * X) + dist.Normal(0.0, 0.2).sample(sample_shape=(N,))
plot(plot_observed_data=True)

N = 1000
X = dist.Uniform(0.0, 5.0).sample(sample_shape=(N,))
y = 0.5 * torch.sin(3 * X) + dist.Normal(0.0, 0.2).sample(sample_shape=(N,))
plot(plot_observed_data=True)

# initialize the inducing inputs
Xu = torch.arange(20.0) / 4.0


def plot_inducing_points(Xu, ax=None):
    for xu in Xu:
        g = ax.axvline(xu, color="red", linestyle="-.", alpha=0.5)
    ax.legend(
        handles=[g],
        labels=["Inducing Point Locations"],
        bbox_to_anchor=(0.5, 1.15),
        loc="upper center",
    )


plot_inducing_points(Xu, plt.gca())

# initialize the kernel and model
pyro.clear_param_store()
kernel = gp.kernels.RBF(input_dim=1)
# we increase the jitter for better numerical stability
sgpr = gp.models.SparseGPRegression(X, y, kernel, Xu=Xu, jitter=1.0e-5)

# the way we setup inference is similar to above
optimizer = torch.optim.Adam(sgpr.parameters(), lr=0.005)
loss_fn = pyro.infer.Trace_ELBO().differentiable_loss
losses = []
locations = []
variances = []
lengthscales = []
noises = []
num_steps = 2000 if not smoke_test else 2
for i in range(num_steps):
    optimizer.zero_grad()
    loss = loss_fn(sgpr.model, sgpr.guide)
    locations.append(sgpr.Xu.data.numpy().copy())
    variances.append(sgpr.kernel.variance.item())
    noises.append(sgpr.noise.item())
    lengthscales.append(sgpr.kernel.lengthscale.item())
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    
plot_loss(losses)


plot(model=sgpr, plot_observed_data=True, plot_predictions=True)
plot_inducing_points(sgpr.Xu.data.numpy(), plt.gca())

fig, ax = plt.subplots(figsize=(12, 6))


def update(iteration):
    pyro.clear_param_store()
    ax.cla()
    kernel_iter = gp.kernels.RBF(
        input_dim=1,
        variance=torch.tensor(variances[iteration]),
        lengthscale=torch.tensor(lengthscales[iteration]),
    )
    sgpr_iter = gp.models.SparseGPRegression(
        X,
        y,
        kernel_iter,
        Xu=torch.tensor(locations[iteration]),
        noise=torch.tensor(noises[iteration]),
        jitter=1.0e-5,
    )
    plot(model=sgpr_iter, plot_observed_data=True, plot_predictions=True, ax=ax)
    plot_inducing_points(sgpr_iter.Xu.data.numpy(), ax=ax)
    ax.set_title(f"Iteration: {iteration}, Loss: {losses[iteration]:0.2f}")
    fig.tight_layout()


# anim = FuncAnimation(fig, update, frames=np.arange(0, num_steps, 30), interval=100)
# plt.close()
# anim.save("../source/_static/img/svgpr-fit.gif", fps=60)

# initialize the inducing inputs
Xu = torch.arange(10.0) / 2.0

# initialize the kernel, likelihood, and model
pyro.clear_param_store()
kernel = gp.kernels.RBF(input_dim=1)
likelihood = gp.likelihoods.Gaussian()
# turn on "whiten" flag for more stable optimization
vsgp = gp.models.VariationalSparseGP(
    X, y, kernel, Xu=Xu, likelihood=likelihood, whiten=True
)

# instead of defining our own training loop, we will
# use the built-in support provided by the GP module
num_steps = 1500 if not smoke_test else 2
losses = gp.util.train(vsgp, num_steps=num_steps)
plot_loss(losses)

plot(model=vsgp, plot_observed_data=True, plot_predictions=True)

df = sns.load_dataset("iris")
df.head()

# only take petal length and petal width
X = torch.from_numpy(
    df[df.columns[2:4]].values.astype("float64"),
)
df["species"] = df["species"].astype("category")
# encode the species as 0, 1, 2
y = torch.from_numpy(df["species"].cat.codes.values.copy())

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors=(0, 0, 0))
plt.xlabel("Feature 1 (Petal length)")
_ = plt.ylabel("Feature 2 (Petal width)")


kernel = gp.kernels.RBF(input_dim=2)
pyro.clear_param_store()
likelihood = gp.likelihoods.MultiClass(num_classes=3)
# Important -- we need to add latent_shape argument here to the number of classes we have in the data
model = gp.models.VariationalGP(
    X,
    y,
    kernel,
    likelihood=likelihood,
    whiten=True,
    jitter=1e-03,
    latent_shape=torch.Size([3]),
)
num_steps = 1000
loss = gp.util.train(model, num_steps=num_steps)

mean, var = model(X)
y_hat = model.likelihood(mean, var)

print(f"Accuracy: {(y_hat==y).sum()*100/(len(y)) :0.2f}%")

cm = confusion_matrix(y, y_hat, labels=[0, 1, 2])
ConfusionMatrixDisplay(cm).plot()


xs = torch.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, steps=100)
ys = torch.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, steps=100)
xx, yy = torch.meshgrid(xs, ys, indexing="xy")

with torch.no_grad():
    mean, var = model(torch.vstack((xx.ravel(), yy.ravel())).t())
    Z = model.likelihood(mean, var)



def plot_pred_2d(arr, xx, yy, contour=False, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    image = ax.imshow(
        arr,
        interpolation="nearest",
        extent=(xx.min(), xx.max(), yy.min(), yy.max()),
        aspect="equal",
        origin="lower",
        cmap=plt.cm.PuOr_r,
    )
    if contour:
        contours = ax.contour(
            xx,
            yy,
            torch.sigmoid(mean).reshape(xx.shape),
            levels=[0.5],
            linewidths=2,
            colors=["k"],
        )

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)

    ax.get_figure().colorbar(image, cax=cax)
    if title:
        ax.set_title(title)

fig, ax = plt.subplots(ncols=3, figsize=(16, 4))
for cl in [0, 1, 2]:
    plot_pred_2d(
        mean[cl, :].reshape(xx.shape), xx, yy, ax=ax[cl], title=f"f (class {cl})"
        )


p_class = torch.nn.functional.softmax(mean, dim=0)
fig, ax = plt.subplots(ncols=3, figsize=(16, 4))
for cl in [0, 1, 2]:
    plot_pred_2d(
        p_class[cl, :].reshape(xx.shape), xx, yy, ax=ax[cl], title=f" p(class {cl})"
        )

plot_pred_2d(Z.reshape(xx.shape), xx, yy, title="Prediction")

# Combining Kernels

X = torch.linspace(-5, 5, 100)
y = torch.sin(X * 8) + 2 * X + 4 + 0.2 * torch.rand_like(X)
plt.scatter(X, y)
plt.show()


pyro.clear_param_store()
linear = gp.kernels.Linear(
    input_dim=1,
)
periodic = gp.kernels.Periodic(
    input_dim=1, period=torch.tensor(0.5), lengthscale=torch.tensor(4.0)
)
rbf = gp.kernels.RBF(
    input_dim=1, lengthscale=torch.tensor(0.5), variance=torch.tensor(0.5)
)
k1 = gp.kernels.Product(kern0=rbf, kern1=periodic)

k = gp.kernels.Sum(linear, k1)
model = gp.models.GPRegression(
    X=X,
    y=y,
    kernel=k,
    jitter=2e-3,
)

loss = gp.util.train(model)
plot_loss(loss)

plt.scatter(X, y, s=50, alpha=0.5)

with torch.no_grad():
    mean, var = model(X)
_ = plt.plot(X, mean, color="C3", lw=2)

