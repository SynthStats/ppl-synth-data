import numpy as np
import pymc as pm
def model(data):
    x = np.asarray(data["stimulus_difficulty"], dtype=float)
    y = np.asarray(data["log_reaction_time"], dtype=float)
    with pm.Model() as fitted_model:
        x_data = pm.Data("Input", x, dims="obs_id")
        intercept = pm.Normal("intercept", mu=0.0, sigma=10.0)
        slope = pm.Normal("slope", mu=0.0, sigma=10.0)
        sigma = pm.HalfNormal("sigma", sigma=5.0)
        pm.Normal("y_obs", mu=intercept+slope*x_data, sigma=sigma, observed=y, dims="obs_id")
    return fitted_model
