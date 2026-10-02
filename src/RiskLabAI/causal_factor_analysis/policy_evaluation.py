"""One-step discrete-action policy evaluation with explicit overlap checks."""

import numpy as np

from RiskLabAI.utils._validation import probability_rows, real_array

__all__ = ["evaluate_policy", "select_policy"]


def evaluate_policy(
    actions, rewards, behavior_probabilities, policy_probabilities, outcome_predictions
):
    """Return inverse-propensity and doubly robust policy-value estimates.

    Probability matrices and outcome_predictions have shape (rows, actions).
    Outcome predictions must be out of fold, or from an independent sample.
    The caller must justify consistency and conditional exchangeability.
    Every policy-supported action needs behavior support. No clipping,
    self-normalization or sequential reinforcement-learning interpretation
    is used. Diagnostics include weights and their effective sample size.
    """
    rewards = real_array(rewards, "rewards", 1)
    behavior = probability_rows(behavior_probabilities, "behavior_probabilities")
    policy = probability_rows(policy_probabilities, "policy_probabilities")
    predictions = real_array(outcome_predictions, "outcome_predictions", 2)
    actions = np.asarray(actions)
    if (
        behavior.shape != policy.shape
        or predictions.shape != policy.shape
        or len(behavior) != len(rewards)
        or actions.shape != rewards.shape
        or actions.dtype.kind not in "iu"
        or np.any(actions < 0)
        or np.any(actions >= behavior.shape[1])
    ):
        raise ValueError("Actions, outcomes and probability shapes are incompatible.")
    row = np.arange(len(actions))
    observed = behavior[row, actions]
    if np.any((policy > 0) & (behavior == 0)) or np.any(observed == 0):
        raise ValueError("Policy evaluation requires behavior-policy overlap.")
    weights = policy[row, actions] / observed
    ipw = weights * rewards
    dr = (policy * predictions).sum(axis=1) + weights * (
        rewards - predictions[row, actions]
    )
    if (
        not np.all(np.isfinite(ipw))
        or not np.all(np.isfinite(dr))
        or not np.all(np.isfinite(weights))
    ):
        raise ValueError("Policy scores exceed floating-point range.")
    normalized = weights / weights.max() if weights.max() else weights
    effective = (
        0.0
        if not normalized.any()
        else float(normalized.sum() ** 2 / (normalized @ normalized))
    )
    return {
        "ipw_value": float(ipw.mean()),
        "dr_value": float(dr.mean()),
        "ipw_scores": ipw,
        "dr_scores": dr,
        "importance_weights": weights,
        "effective_sample_size": effective,
        "max_weight": float(weights.max()),
        "mean_weight": float(weights.mean()),
    }


def select_policy(
    actions,
    rewards,
    behavior_probabilities,
    candidate_probabilities,
    outcome_predictions,
    *,
    training_indices,
    evaluation_indices,
):
    """Select the highest training DR value, then evaluate on later held-out rows.

    Candidates have shape (policies, rows, actions), fixed before evaluation.
    The first candidate wins ties. All training indices must precede all
    evaluation indices, with no duplicates or overlap. Nuisance predictions
    must independently obey the same information boundary; this function
    cannot verify their training history. Evaluation outcomes never select
    the policy. No value estimate for a reused selection sample is reported
    as held-out performance.
    """
    rewards = real_array(rewards, "rewards", 1)
    candidates = real_array(candidate_probabilities, "candidate_probabilities", 3)
    actions = np.asarray(actions)
    behavior = np.asarray(behavior_probabilities)
    outcomes = np.asarray(outcome_predictions)
    if (
        candidates.shape[1] != len(rewards)
        or actions.shape != rewards.shape
        or behavior.shape != candidates.shape[1:]
        or outcomes.shape != behavior.shape
    ):
        raise ValueError("Candidate and observation arrays must match.")
    indices = []
    for raw in (training_indices, evaluation_indices):
        index = np.asarray(raw)
        if (
            index.ndim != 1
            or index.dtype.kind not in "iu"
            or index.size == 0
            or np.any(index < 0)
            or np.any(index >= len(rewards))
            or len(np.unique(index)) != len(index)
        ):
            raise ValueError("Split indices must be nonempty, distinct and valid.")
        indices.append(index)
    train, evaluation = indices
    if train.max() >= evaluation.min():
        raise ValueError("Training must precede held-out evaluation.")
    values = [
        evaluate_policy(
            actions[train],
            rewards[train],
            behavior[train],
            candidate[train],
            outcomes[train],
        )["dr_value"]
        for candidate in candidates
    ]
    selected = int(np.argmax(values))
    result = evaluate_policy(
        actions[evaluation],
        rewards[evaluation],
        behavior[evaluation],
        candidates[selected, evaluation],
        outcomes[evaluation],
    )
    return {
        "selected_policy": selected,
        "training_values": np.array(values),
        "evaluation": result,
    }
