"""Prequential prediction with an optional ADWIN-triggered rolling refit."""

import numpy as np
from sklearn.base import clone

from RiskLabAI.utils._validation import positive_integer, real_array, real_scalar

__all__ = ["prequential_adaptation"]


def prequential_adaptation(
    features, targets, estimator, *, loss_function, window_size, min_samples, delta
):
    """Predict each row before observing its target, then update ADWIN.

    loss_function(target, prediction) must return a value in [0, 1]. The
    initial fit uses the first min_samples observed rows. A drift alarm
    triggers a fit using at most window_size most recent observed rows,
    including the current row, so that refit can affect only later forecasts.
    Warm-up predictions/losses are NaN. The supplied estimator is cloned;
    callers must set its own random_state for deterministic behavior.
    Requires optional River; returns predictions, losses and refit indices.
    """
    from river.drift import ADWIN

    x = real_array(features, "features", 2)
    y = real_array(targets, "targets", 1)
    window_size = positive_integer(window_size, "window_size")
    min_samples = positive_integer(min_samples, "min_samples")
    delta = real_scalar(delta, "delta")
    if (
        len(x) != len(y)
        or min_samples > window_size
        or min_samples >= len(y)
        or not 0 < delta < 1
        or not callable(loss_function)
    ):
        raise ValueError(
            "Require matching rows, min_samples <= window_size < infinity, enough data, and delta in (0,1)."
        )
    detector = ADWIN(delta=delta)
    predictions = np.full(len(y), np.nan)
    losses = np.full(len(y), np.nan)
    alarms, refits = [], [min_samples - 1]
    model = clone(estimator).fit(x[:min_samples], y[:min_samples])
    for index in range(min_samples, len(y)):
        raw_prediction = np.asarray(model.predict(x[index : index + 1]))
        if raw_prediction.shape != (1,):
            raise ValueError("Estimator must return one scalar prediction per row.")
        prediction = real_scalar(raw_prediction[0], "prediction")
        loss = real_scalar(loss_function(y[index], prediction), "loss", minimum=0)
        if loss > 1:
            raise ValueError("ADWIN losses must be in [0,1].")
        predictions[index], losses[index] = prediction, loss
        detector.update(loss)
        if detector.drift_detected:
            alarms.append(index)
            begin = max(0, index + 1 - window_size)
            model = clone(estimator).fit(x[begin : index + 1], y[begin : index + 1])
            refits.append(index)
    return {
        "predictions": predictions,
        "losses": losses,
        "alarm_indices": np.array(alarms, dtype=int),
        "refit_indices": np.array(refits, dtype=int),
    }
