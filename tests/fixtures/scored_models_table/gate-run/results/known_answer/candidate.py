import numpy as np
import pymc as pm
def model(data):
    y = np.asarray(data["y"], dtype=float)
    with pm.Model() as known_answer_model:
        mu = pm.Normal("mu", mu=0.0, sigma=1.0)
        pm.Normal("y", mu=mu, sigma=1.0, observed=y)
    return known_answer_model
