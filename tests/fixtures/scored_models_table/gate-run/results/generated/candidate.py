import numpy as np
import pymc as pm


def model(data):
    """
    PyMC model for latent strategy reaction time mixture.
    
    This model accounts for:
    1. A mixture of latent behavioral regimes (fast, moderate, slow)
    2. Participant-level random effects (repeated measures)
    3. Stimulus difficulty effects on reaction time
    """
    # Extract data
    log_reaction_time = data['log_reaction_time']
    stimulus_difficulty = data['stimulus_difficulty']
    participant_id = data['participant_id']
    n = len(log_reaction_time)
    
    # Get unique participants
    unique_participants = np.unique(participant_id)
    n_participants = len(unique_participants)
    
    # Map participant IDs to indices
    participant_to_idx = {pid: idx for idx, pid in enumerate(unique_participants)}
    participant_indices = np.array([participant_to_idx[pid] for pid in participant_id])
    
    with pm.Model() as model:
        # Number of mixture components
        n_components = 3
        
        # Mixture weights (Dirichlet prior)
        weights = pm.Dirichlet('weights', a=np.ones(n_components))
        
        # Component-specific intercepts (baseline log RT for each regime)
        # Use ordered constraints to ensure fast < moderate < slow
        mu_raw = pm.Normal('mu_raw', mu=0, sigma=5, shape=n_components)
        # Order the mu values: fast < moderate < slow
        mu = pm.math.sort(mu_raw)
        
        # Shared standard deviation for all components
        sigma = pm.HalfNormal('sigma', sigma=5)
        
        # Stimulus difficulty effect (positive means harder = slower)
        beta_difficulty = pm.Normal('beta_difficulty', mu=0, sigma=2)
        
        # Participant-level random effects (shared across components)
        # This captures participant-specific baseline speed
        participant_effects = pm.Normal('participant_effects', mu=0, sigma=3, 
                                       shape=n_participants)
        
        # Get participant effect for each observation
        p_effects = participant_effects[participant_indices]
        
        # Log-mixture density for each observation
        # For each observation, compute the probability under each component
        # and then mix them
        
        # Linear predictor for difficulty and participant effects
        # The difficulty effect is shared across components
        # But each component has its own baseline
        
        # Compute log probabilities for each component
        log_probs = []
        for k in range(n_components):
            # Mean for component k for observation i
            mu_k = mu[k] + beta_difficulty * stimulus_difficulty + p_effects
            
            # Log probability under component k
            # Using log of normal density
            log_prob_k = pm.math.log(weights[k]) + pm.math.log(
                pm.math.exp(-0.5 * ((log_reaction_time - mu_k) / sigma) ** 2) / 
                (sigma * np.sqrt(2 * np.pi))
            )
            log_probs.append(log_prob_k)
        
        # Stack log probabilities
        log_probs_stack = pm.math.stack(log_probs, axis=0)
        
        # Log-sum-exp for mixture probability
        log_mix = pm.math.logsumexp(log_probs_stack, axis=0)
        
        # Observed data
        pm.Normal('obs', mu=0, sigma=1, observed=log_mix)
        
        # Wait, this approach with observed=log_mix is not correct for observed data.
        # Let me use pm.Mixture instead.
        
    # Let me restart with a cleaner approach using pm.Mixture
    with pm.Model() as model:
        # Number of mixture components
        n_components = 3
        
        # Mixture weights (Dirichlet prior)
        weights = pm.Dirichlet('weights', a=np.ones(n_components))
        
        # Component-specific intercepts (baseline log RT for each regime)
        # Use ordered constraints to ensure fast < moderate < slow
        mu_raw = pm.Normal('mu_raw', mu=0, sigma=5, shape=n_components)
        # Order the mu values: fast < moderate < slow
        mu = pm.math.sort(mu_raw)
        
        # Shared standard deviation for all components
        sigma = pm.HalfNormal('sigma', sigma=5)
        
        # Stimulus difficulty effect (positive means harder = slower)
        beta_difficulty = pm.Normal('beta_difficulty', mu=0, sigma=2)
        
        # Participant-level random effects (shared across components)
        participant_effects = pm.Normal('participant_effects', mu=0, sigma=3, 
                                       shape=n_participants)
        
        # Get participant effect for each observation
        p_effects = participant_effects[participant_indices]
        
        # For each component, define the mean
        # The mean depends on component-specific baseline, difficulty, and participant effect
        mu_components = []
        for k in range(n_components):
            mu_k = mu[k] + beta_difficulty * stimulus_difficulty + p_effects
            mu_components.append(mu_k)
        
        # Use pm.Mixture to define the observed distribution
        # pm.Mixture expects a list of component distributions or a way to specify them
        # We'll use pm.Normal for each component
        
        # Define component distributions
        comp_dists = []
        for k in range(n_components):
            comp_dists.append(pm.Normal.dist(mu=mu_components[k], sigma=sigma))
        
        # Create mixture distribution
        mixture_dist = pm.Mixture('mixture', w=weights, comp_dists=comp_dists)
        
        # Observed data
        pm.DensityDist('obs', logp=lambda value: pm.logp(mixture_dist, value),
                       observed=log_reaction_time)
    
    return model