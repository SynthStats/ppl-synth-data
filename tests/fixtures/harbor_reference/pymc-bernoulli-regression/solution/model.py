def model(data):
    y = np.asarray(data['y'])
    x_0 = np.asarray(data['dot_coherence'])
    x_1 = np.asarray(data['trial_number'])
    x_2 = np.asarray(data['mask_applied'])
    with pm.Model() as result:
        intercept_centered = pm.Normal('intercept_centered', mu=0.0, sigma=2.5)
        coef_0 = pm.Normal('coef_0', mu=0.0, sigma=0.11600531164046893)
        coef_1 = pm.Normal('coef_1', mu=0.0, sigma=0.01721812922114998)
        coef_2 = pm.Normal('coef_2', mu=0.0, sigma=5.096720759468778)
        intercept = pm.Deterministic('intercept', intercept_centered - coef_0 * 41.5825 - coef_1 * 247.7 - coef_2 * 0.4)
        linear = intercept + coef_0 * x_0 + coef_1 * x_1 + coef_2 * x_2
        pm.Bernoulli('y', logit_p=linear, observed=y)
    return result
