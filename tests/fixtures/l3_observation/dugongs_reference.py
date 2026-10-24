import numpy as np
import pymc as pm


def model(data):
    """Dugong length against age: the growth curve and priors of BoxingGym's
    dugongs environment, with a Normal likelihood and a fixed noise scale.
    """
    age = np.asarray(data["age"], dtype=float)
    y = np.asarray(data["length"], dtype=float)
    with pm.Model() as dugongs_model:
        age_data = pm.Data("age_data", age)
        alpha = pm.Normal("alpha", mu=2.0, sigma=0.2)
        beta = pm.Normal("beta", mu=1.5, sigma=0.5)
        lam = pm.Normal("lam", mu=0.4, sigma=0.5)
        mu = alpha - beta * pm.math.abs(lam) ** age_data
        pm.Normal("length", mu=mu, sigma=0.25, observed=y)
    return dugongs_model
