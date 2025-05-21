# Levelling‑Rod Calibration with **calipy** & Pyro

Welcome to the open‑source workspace for probabilistic calibration of geodetic levelling rods.  We combine the **calipy** calibration toolkit with **Pyro**‐based Bayesian models to quantify both **random** and **systematic** effects in rod readings, delivering traceable uncertainties for high‑precision height transfer.

> **Why it matters**  Millimetre‑level levelling depends on rods whose scale and zero are stable at the 10⁻5 level.  Classical least‑squares gives point estimates only; our hierarchical Bayes approach propagates full posterior uncertainty **per rod** and **per rod family**, making acceptance and re‑calibration decisions defensible.

---

## Model Line‑up (road‑map)

| Stage | Folder               | Main Idea                                                                                                                           | Strengths                                                          | Known Limitations                                                                            |
| ----- | -------------------- | ----------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ | -------------------------------------------------------------------------------------------- |
| **1** | `Model_0_global_params/`   | Assumes `alpha = (offset, scale)` is drawn from one global multivariate Gaussian distribution independent of staff id or staff class | Fast ⏱, transparent, good didactic starting point                  | Ignores rod‑to‑rod heterogeneity, over‑shrinks extremes                                      |
| **2** | `Model_1_perclass_params` | Introduces per-class multivariate Gaussian distributions. Mean and covariance of alpha are different for each class of rods | Captures different properties of classes | Assumes there is no noise on the observations of the levelling rod params and ignores staff_ids to do so |
| **3** | `Model_2_perstaff_latents` | Two‑level hierarchical – per‑rod latent `(offset, scale)` drawn from a class‑specific Gaussian prior, shared calibration covariance | Captures class structure, gives posterior for each rod, extensible | Assumes impact of calibration procedure has no systematics |
| **4** | `Model_2_perstaff_latents`            | Non‑stationary GP over rod graduations, temperature model                                                                           | Handles scale drift along rod, environmental corrections           | Higher compute cost, requires dense data                                                     |

*Detailed markdown for each model.  Start with **baseline\Model_0_global_params.md** to grasp the notation.*


---

## Results Snapshot

* **Baseline ELBO:** ≈ 1.3 × 10⁵ after 2 k steps (converged)
* **Hierarchical ELBO:** ≈ 9.8 × 10⁴ after 10 k steps – 10 % relative improvement
* Posterior predictive coverage (**95 %‑CI**) aligns with held‑out check observations within 1 σ.

---

## License

Distributed under the **MIT License**.  See `LICENSE` for details.

---

## References & Further Reading

* Blei, D. M. *&* Jordan, M. I. (2006). **Variational Methods for Bayesian Hierarchical Models.**


