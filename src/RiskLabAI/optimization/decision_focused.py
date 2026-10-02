"""SPO+ loss over the convex hull of explicitly supplied feasible vertices."""

import numpy as np

from RiskLabAI.utils._validation import real_array

__all__ = ["spo_plus_loss", "spo_plus_torch"]


def spo_plus_loss(predicted_cost, true_cost, vertices):
    """Return SPO+ loss and a subgradient for linear cost minimization.

    The fixed feasible region is the convex hull of rows of vertices.
    Complete enumeration is an exact optimization oracle for this region.
    Ties choose the first supplied vertex. This does not represent a general
    nonlinear optimization oracle or enumerate vertices automatically.
    """
    predicted = real_array(predicted_cost, "predicted_cost", 1)
    truth = real_array(true_cost, "true_cost", 1)
    vertices = real_array(vertices, "vertices", 2)
    if predicted.shape != truth.shape or vertices.shape[1] != len(truth):
        raise ValueError("Costs and vertices must have the same decision dimension.")
    true_objectives = vertices @ truth
    transformed_objectives = vertices @ (2 * predicted - truth)
    if (
        not np.isfinite(true_objectives).all()
        or not np.isfinite(transformed_objectives).all()
    ):
        raise ValueError("Oracle objectives exceed floating-point range.")
    true_index = int(np.argmin(true_objectives))
    transformed_index = int(np.argmin(transformed_objectives))
    true_decision, transformed = vertices[true_index], vertices[transformed_index]
    loss = float(
        (truth - 2 * predicted) @ transformed + (2 * predicted - truth) @ true_decision
    )
    if (
        not np.isfinite(loss)
        or not np.isfinite(2 * (true_decision - transformed)).all()
    ):
        raise ValueError("Loss/subgradient exceeds floating-point range.")
    return {
        "loss": loss,
        "subgradient": 2 * (true_decision - transformed),
        "true_vertex": true_index,
        "transformed_vertex": transformed_index,
    }


def spo_plus_torch(predicted_cost, true_cost, vertices):
    """Return differentiable SPO+ losses for one vector or a batch of costs.

    All inputs are finite floating tensors on the same device, with matching
    dtype. true_cost and vertices are constants for differentiation. The
    gradient with respect to predicted_cost is a valid selected subgradient,
    using first-index tie breaking. No batch reduction is performed.
    """
    import torch

    if not all(
        isinstance(x, torch.Tensor)
        and x.is_floating_point()
        and torch.isfinite(x).all()
        for x in (predicted_cost, true_cost, vertices)
    ):
        raise ValueError("All inputs must be finite floating tensors.")
    if (
        predicted_cost.ndim not in (1, 2)
        or predicted_cost.shape != true_cost.shape
        or vertices.ndim != 2
        or vertices.shape[0] == 0
        or vertices.shape[1] != predicted_cost.shape[-1]
        or predicted_cost.numel() == 0
    ):
        raise ValueError("Costs and vertex shapes are incompatible.")
    if any(
        x.device != predicted_cost.device or x.dtype != predicted_cost.dtype
        for x in (true_cost, vertices)
    ):
        raise ValueError("All tensors must use the same device and dtype.")
    truth, choices = true_cost.detach(), vertices.detach()
    true_objectives = truth @ choices.T
    transformed_objectives = (2 * predicted_cost.detach() - truth) @ choices.T
    if (
        not torch.isfinite(true_objectives).all()
        or not torch.isfinite(transformed_objectives).all()
    ):
        raise ValueError("Oracle objectives exceed floating-point range.")
    true_decision = choices[torch.argmin(true_objectives, dim=-1)]
    transformed = choices[torch.argmin(transformed_objectives, dim=-1)]
    loss = (
        (truth - 2 * predicted_cost) * transformed
        + (2 * predicted_cost - truth) * true_decision
    ).sum(dim=-1)
    if (
        not torch.isfinite(loss).all()
        or not torch.isfinite(2 * (true_decision - transformed)).all()
    ):
        raise ValueError("Loss/subgradient exceeds floating-point range.")
    return loss
