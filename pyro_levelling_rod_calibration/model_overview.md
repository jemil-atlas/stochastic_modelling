# Model Line-Up: from Global Pooling to Full Hierarchy

This document reviews the Gaussian models that underpin our levelling-rod calibration workflow.  Each model shares the same *building blocks*—multivariate normals for **offset** and **scale**—but differs in **where** randomness is introduced and **how** information is pooled.

| Model | Levels | Measurement error (`Σ_cal`) | Latent per-rod vars | Typical use-case |
|-------|--------|-----------------------------|---------------------|------------------|
| **0** | *population → rod* | — (ignored) | — | smoke test, global ball-park |
| **1** | *class → rod* | — (ignored) | — | average correction per make |
| **2** | *class → rod → obs* | **learned** | `α_j` | full certificates, trend analysis |

---

## 0️⃣ Model 0 — *Global Gaussian population*

```python
# μ ∈ ℝ²  (offset, scale)
# Σ ∈ ℝ²ˣ²  positive-definite
μ  = pyro.param("mu",    torch.zeros(2))
Σ  = pyro.param("Sigma", torch.eye(2),
                constraint=constraints.positive_definite)

with pyro.plate("rod", n_rods):
    pyro.sample("y", dist.MultivariateNormal(μ, Σ), obs=data)
````

### Generative story

1. **Population parameters** `μ, Σ` are fixed but unknown.
2. Every *rod* is manufactured by *drawing its true parameters*
   $(\text{offset}, \text{scale})$ **once** from
   $\mathcal N(μ, Σ)$.
3. We observe those parameters **without additional measurement noise**
   (a conscious simplification).

Thus the likelihood is

$$
\mathcal L(μ, Σ \mid \mathbf y)
  =\prod_{i=1}^{n_{\text{rods}}}
   \mathcal N(\mathbf y_i; μ, Σ).
$$

No latent variables appear; optimising `μ, Σ` via SVI collapses to
**maximum-likelihood estimation**.

**Strengths ✅**

* **Tiny**: only 5 parameters (2 means + 3 cov entries).
* Ideal smoke test for data loading and unit handling.

**Limitations ⚠️**

* Attributes *all* spread to manufacturing variation—bench repeatability is
  ignored.
* Ignores rod classes—mixing Leica 2 m and Trimble 3 m rods will smear their
  distinct behaviours.
* Provides no posterior uncertainty for `μ, Σ` themselves (point estimates).

---

## 1️⃣ Model 1 — *Per-class Gaussian*

```python
# One (μ_c, Σ_c) pair for each class c
μ_c = pyro.param("mu",    torch.zeros([C, 2]))
Σ_c = pyro.param("Sigma", torch.eye(2).expand(C,2,2),
                 constraint=constraints.positive_definite)

with pyro.plate("rod", n_rods):
    c = class_idx[i]                     # lookup vector
    pyro.sample("y",
        dist.MultivariateNormal(μ_c[c], Σ_c[c]),
        obs=data)
```

### What changes?

* Likelihood factorises by class:

  $$
  \mathcal L
  =\prod_{c=1}^{C}
    \prod_{i\in c}
    \mathcal N(\mathbf y_i;\, μ_c, Σ_c).
  $$

* **Partial pooling** now happens at the class level; rods of the same make
  share parameters.

* Still assumes perfect measurement (Σ\_cal absent).

**When to use**: ≥ 20 rods per make, you trust the bench, and only need the
*average* correction per product line.

---

## 2️⃣ Model 2 — *Full hierarchical Gaussian*

```python
# ----- Class level -----
μ_c , Σ_c  = pyro.param(...)

# ----- Calibration level -----
Σ_cal      = pyro.param("Sigma_cal", torch.eye(2),
                        constraint=constraints.positive_definite)

# ----- Rod level -----
with pyro.plate("rod", n_rods):
    α_j = pyro.sample("alpha",
            dist.MultivariateNormal(μ_c[class_of[j]], Σ_c[class_of[j]]))

# ----- Observation level -----
with pyro.plate("obs", n_obs):
    pyro.sample("y",
        dist.MultivariateNormal(α_j[rod_of[i]], Σ_cal),
        obs=data)
```

### Key benefits

* **Latent rod truths** `α_j` yield rod-specific posteriors and enable
  **shrinkage** for sparsely observed rods.
* **Uncertainty decomposition**: manufacturing spread (`Σ_c`) is separated from
  bench repeatability (`Σ_cal`).
* Provides a natural base for extensions—e.g., time-varying drift or Student-t
  tails.

**Trade-off**: more parameters and a heavier guide, but the only choice once
you issue certificates per rod or track long-term stability.

---

## At-a-glance rating

| Feature                      | Model 0 | Model 1 | Model 2 |
| ---------------------------- | ------- | ------- | ------- |
| Handles measurement error    | ❌       | ❌       | **✔**   |
| Distinguishes makes          | ❌       | **✔**   | **✔**   |
| Gives posterior for each rod | ❌       | ❌       | **✔**   |
| Setup complexity             | ★☆☆     | ★★☆     | ★★★     |

Pick the *simplest* model that answers your question—upgrade only when the data
(or the metrologist) demands more nuance.




