import pymc as pm
import numpy as np

def model(data):
    y = np.asarray(data["log_reaction_time"], dtype=float).reshape(-1)
    difficulty = np.asarray(data["stimulus_difficulty"], dtype=float).reshape(-1)
    participant_raw = np.asarray(data["participant_id"]).reshape(-1).astype(str)

    participant_labels, participant_idx = np.unique(participant_raw, return_inverse=True)
    participant_idx = participant_idx.astype("int64")

    n = y.shape[0]
    K = 3

    y_mean = float(np.mean(y))
    y_sd = float(np.std(y, ddof=1)) if n > 1 else 1.0
    if not np.isfinite(y_sd) or y_sd <= 1e-6:
        y_sd = 1.0

    coords = {
        "trial": np.arange(n),
        "mode": ["fast", "deliberative", "lapse"],
        "mode_gap": ["fast_to_deliberative", "deliberative_to_lapse"],
        "mode_nonreference": ["deliberative", "lapse"],
        "participant": participant_labels,
    }

    with pm.Model(coords=coords) as model:
        difficulty_data = pm.Data("stimulus_difficulty", difficulty, dims="trial")
        participant_idx_data = pm.Data("participant_idx", participant_idx, dims="trial")

        grand_mean = pm.Normal("grand_mean", mu=y_mean, sigma=2.5 * y_sd)

        mode_gaps = pm.HalfNormal(
            "mode_mean_gaps",
            sigma=2.0 * y_sd,
            dims="mode_gap",
        )
        uncentered_offsets = pm.math.concatenate(
            [
                np.zeros(1),
                mode_gaps[0:1],
                mode_gaps[0:1] + mode_gaps[1:2],
            ],
            axis=0,
        )
        centered_offsets = uncentered_offsets - pm.math.mean(uncentered_offsets)

        base_mode_mean = pm.Deterministic(
            "base_mode_mean_easy",
            grand_mean + centered_offsets,
            dims="mode",
        )

        difficulty_effect = pm.Normal(
            "difficulty_effect_by_mode",
            mu=0.0,
            sigma=y_sd,
            dims="mode",
        )

        hard_mode_mean = pm.Deterministic(
            "base_mode_mean_hard",
            base_mode_mean + difficulty_effect,
            dims="mode",
        )

        mode_sigma = pm.HalfNormal(
            "mode_sigma",
            sigma=1.5 * y_sd,
            dims="mode",
        )

        participant_sd = pm.HalfNormal(
            "participant_sd",
            sigma=y_sd,
        )
        participant_z = pm.Normal(
            "participant_z",
            mu=0.0,
            sigma=1.0,
            dims="participant",
        )
        participant_offset = pm.Deterministic(
            "participant_offset",
            participant_sd * (participant_z - pm.math.mean(participant_z)),
            dims="participant",
        )

        weight_logit_intercept_free = pm.Normal(
            "mode_weight_logit_intercept_free",
            mu=0.0,
            sigma=1.5,
            dims="mode_nonreference",
        )
        weight_logit_difficulty_free = pm.Normal(
            "mode_weight_logit_difficulty_free",
            mu=0.0,
            sigma=1.0,
            dims="mode_nonreference",
        )

        weight_logit_intercept = pm.Deterministic(
            "mode_weight_logit_intercept",
            pm.math.concatenate([np.zeros(1), weight_logit_intercept_free], axis=0),
            dims="mode",
        )
        weight_logit_difficulty = pm.Deterministic(
            "mode_weight_logit_difficulty",
            pm.math.concatenate([np.zeros(1), weight_logit_difficulty_free], axis=0),
            dims="mode",
        )

        logits_trial = (
            weight_logit_intercept[None, :]
            + difficulty_data[:, None] * weight_logit_difficulty[None, :]
        )

        mode_probability = pm.Deterministic(
            "mode_probability",
            pm.math.softmax(logits_trial, axis=1),
            dims=("trial", "mode"),
        )

        pm.Deterministic(
            "mode_probability_easy",
            pm.math.softmax(weight_logit_intercept[None, :], axis=1)[0],
            dims="mode",
        )
        pm.Deterministic(
            "mode_probability_hard",
            pm.math.softmax(
                (weight_logit_intercept + weight_logit_difficulty)[None, :],
                axis=1,
            )[0],
            dims="mode",
        )

        mu_components = pm.Deterministic(
            "mean_log_reaction_time_by_trial_mode",
            base_mode_mean[None, :]
            + difficulty_data[:, None] * difficulty_effect[None, :]
            + participant_offset[participant_idx_data][:, None],
            dims=("trial", "mode"),
        )

        component_distributions = pm.Normal.dist(
            mu=mu_components,
            sigma=mode_sigma,
        )

        pm.Mixture(
            "log_reaction_time",
            w=mode_probability,
            comp_dists=component_distributions,
            observed=y,
            dims="trial",
        )

        return model