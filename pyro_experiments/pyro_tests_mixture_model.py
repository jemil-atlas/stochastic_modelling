import torch
import pyro
import matplotlib.pyplot as plt
from pyro.distributions import Normal, MixtureSameFamily, Categorical
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

# Step 1: Generate some multimodal data
locs = torch.tensor([-3., 3.])
scales = torch.tensor([1., 1.])
mix = torch.tensor([0.5, 0.5])

# create a mixture of two normal distributions
mixture = MixtureSameFamily(Categorical(mix), Normal(locs, scales))
data = mixture.sample(sample_shape=(1000,))

# Step 2: Define a mixture model
def model(data):
    locs = pyro.param("locs", torch.tensor([-10., 10.]))
    scales = pyro.param("scales", torch.tensor([1., 1.]), constraint=pyro.distributions.constraints.positive)
    mix = pyro.param("mix", torch.tensor([0.5, 0.5]), constraint=pyro.distributions.constraints.simplex)
    with pyro.plate("data", len(data)):
        pyro.sample("obs", MixtureSameFamily(Categorical(mix), Normal(locs, scales)), obs=data)

# Step 3: Define a guide (here we use an empty guide because all variables are observed)
def guide(data):
    pass

# Step 4: Set up inference
svi = SVI(model, guide, Adam({"lr": 0.005}), Trace_ELBO())

# Step 5: Perform inference
num_epochs = 2000
for epoch in range(num_epochs):
    loss = svi.step(data)
    if epoch % 100 == 0:
        print(f"Epoch {epoch} Loss {loss}")

# Step 6: Extract learned parameters
learned_locs = pyro.param("locs").detach()
learned_scales = pyro.param("scales").detach()
learned_mix = pyro.param("mix").detach()

# Step 7: Visualize
plt.figure(figsize=(10, 6))
plt.hist(data.numpy(), bins=50, density=True, label='Original Data', alpha=0.5)
x = torch.linspace(-10, 10, 1000)
learned_mixture = MixtureSameFamily(Categorical(learned_mix), Normal(learned_locs, learned_scales))
plt.plot(x, torch.exp(learned_mixture.log_prob(x)), label='Learned Mixture')
plt.legend()
plt.show()

