import pymc as pm
def model(data):
    with pm.Model() as result:
        t = pm.Data("t", data["Time"])
        y = pm.Data("y", data["Infected_Count"])
        theta = pm.TruncatedNormal("theta", mu=1, sigma=1, lower=0, upper=2)
        pm.Binomial("Infected_Count", n=50, p=1-pm.math.exp(-theta*t), observed=y)
    return result
