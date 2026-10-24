import numpy as np
import pymc as pm
def model(data):
    with pm.Model() as null_model:
        y = pm.Data("target_data", np.asarray(data["log_reaction_time"], dtype=float))
        intercept = pm.Normal("intercept", mu=0.0, sigma=10.0)
        sigma = pm.HalfNormal("sigma", sigma=5.0)
        mu = intercept
        pm.Normal("log_reaction_time", mu=mu, sigma=sigma, observed=y)
    return null_model
