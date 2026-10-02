"""Fixed-length temporal TimeGAN using the paper's reconstruction and GAN losses.

Implements the temporal-only setting of equations 7--11, with an autoregressive
generator used both freely and with teacher forcing. It does not reproduce the
reference repository's separate supervisor or moment-matching extensions.
Requires optional PyTorch. Synthetic windows are not an unlimited market path.
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from RiskLabAI.utils._validation import positive_integer, real_array, real_scalar

__all__ = [
    "chronological_timegan_data",
    "TimeGAN",
    "fit_timegan",
    "timegan_quality_report",
]


def chronological_timegan_data(series, *, training_end, window_length):
    """Split before windowing and scale using only the training segment.

    Return overlapping windows inside each disjoint segment and the feature
    minimum/scale needed for inversion. Validation values are never clipped.
    Constant training features have scale one. No window crosses the split.
    """
    series = real_array(series, "series", 2)
    training_end = positive_integer(training_end, "training_end")
    length = positive_integer(window_length, "window_length")
    if length < 2 or training_end < length or len(series) - training_end < length:
        raise ValueError(
            "Both chronological segments need a complete window of length >= 2."
        )
    minimum = series[:training_end].min(axis=0)
    scale = np.ptp(series[:training_end], axis=0)
    scale[scale == 0] = 1
    scaled = (series - minimum) / scale

    def windows(part):
        return np.stack([part[i : i + length] for i in range(len(part) - length + 1)])

    return {
        "training": windows(scaled[:training_end]),
        "validation": windows(scaled[training_end:]),
        "minimum": minimum,
        "scale": scale,
        "training_end": training_end,
    }


class _SequenceMap(nn.Module):
    def __init__(self, inputs, hidden, outputs, *, sigmoid=True, bidirectional=False):
        super().__init__()
        self.recurrent = nn.GRU(
            inputs, hidden, batch_first=True, bidirectional=bidirectional
        )
        self.output = nn.Linear(hidden * (2 if bidirectional else 1), outputs)
        self.sigmoid = sigmoid

    def forward(self, inputs):
        output = self.output(self.recurrent(inputs)[0])
        return torch.sigmoid(output) if self.sigmoid else output


class TimeGAN(nn.Module):
    """Temporal-only GRU TimeGAN with an explicitly autoregressive generator.

    The embedding and recovery use causal GRUs; the discriminator uses a
    bidirectional GRU. Independent Gaussian innovations enter every step.
    Supplied training data must be scaled to [0,1]. CPU float64 is the default.
    """

    def __init__(self, features, hidden_size=12, *, seed=0):
        super().__init__()
        self.features = positive_integer(features, "features")
        self.hidden_size = positive_integer(hidden_size, "hidden_size")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.embedding = _SequenceMap(features, hidden_size, hidden_size)
            self.recovery = _SequenceMap(hidden_size, hidden_size, features)
            self.generator_cell = nn.GRUCell(2 * hidden_size, hidden_size)
            self.generator_output = nn.Linear(hidden_size, hidden_size)
            self.discriminator = _SequenceMap(
                hidden_size, hidden_size, 1, sigmoid=False, bidirectional=True
            )
        self.double()

    def _check(self, value, name, features):
        parameter = next(self.parameters())
        if (
            not isinstance(value, torch.Tensor)
            or value.ndim != 3
            or value.shape[0] == 0
            or value.shape[1] < 2
            or value.shape[2] != features
            or not value.is_floating_point()
            or value.dtype != parameter.dtype
            or value.device != parameter.device
            or not torch.isfinite(value).all()
        ):
            raise ValueError(
                f"{name} must be finite (windows, time>=2, {features}) and match the model dtype/device."
            )

    def latent_sequence(self, noise, teacher=None):
        """Generate latents freely, or condition step t on observed latent t-1."""
        self._check(noise, "noise", self.hidden_size)
        if teacher is not None:
            self._check(teacher, "teacher", self.hidden_size)
            if teacher.shape != noise.shape:
                raise ValueError("Teacher and noise shapes must match.")
        state = torch.zeros_like(noise[:, 0])
        previous = torch.zeros_like(state)
        outputs = []
        for index in range(noise.shape[1]):
            if teacher is not None and index > 0:
                previous = teacher[:, index - 1]
            state = self.generator_cell(
                torch.cat([previous, noise[:, index]], dim=1), state
            )
            previous = torch.sigmoid(self.generator_output(state))
            outputs.append(previous)
        return torch.stack(outputs, dim=1)

    def losses(self, observations, noise):
        """Return equations 7--9, averaged over windows and summed over time.

        Discriminator loss is -L_U. Generator adversarial loss is the minimax
        fake-data term log(1-D(G(z))); the real-data term has zero G gradient.
        Supervision includes the initial latent with zero initial history.
        """
        self._check(observations, "observations", self.features)
        if torch.any((observations < 0) | (observations > 1)):
            raise ValueError("Training observations must be scaled to [0,1].")
        if observations.shape[:2] != noise.shape[:2]:
            raise ValueError("Observations and noise need matching windows/times.")
        latent = self.embedding(observations)
        reconstructed = self.recovery(latent)
        synthetic = self.latent_sequence(noise)
        teacher = self.latent_sequence(noise, latent)
        real_logits = self.discriminator(latent)
        fake_logits = self.discriminator(synthetic)
        reconstruction = (observations - reconstructed).square().sum(dim=(1, 2)).mean()
        supervised = (latent - teacher).square().sum(dim=(1, 2)).mean()
        real_log = F.logsigmoid(real_logits).sum(dim=(1, 2)).mean()
        fake_log = F.logsigmoid(-fake_logits).sum(dim=(1, 2)).mean()
        return {
            "reconstruction": reconstruction,
            "supervised": supervised,
            "discriminator": -real_log - fake_log,
            "generator_adversarial": fake_log,
        }

    def sample(self, windows, length, *, seed):
        """Return detached scaled windows from independent local seeded noise."""
        windows, length = positive_integer(windows, "windows"), positive_integer(
            length, "length"
        )
        parameter = next(self.parameters())
        generator = torch.Generator(device=parameter.device).manual_seed(seed)
        noise = torch.randn(
            windows,
            length,
            self.hidden_size,
            generator=generator,
            dtype=parameter.dtype,
            device=parameter.device,
        )
        with torch.no_grad():
            return self.recovery(self.latent_sequence(noise))


def fit_timegan(
    model,
    training_windows,
    *,
    embedding_steps,
    supervised_steps,
    joint_steps,
    batch_size,
    learning_rate=0.001,
    embedding_weight=1.0,
    generator_weight=10.0,
    seed=0,
):
    """Train reconstruction, teacher-forced generation, then all objectives.

    Step counts are fixed in advance; only training windows enter optimization.
    Joint training separately updates E/R on L_R+lambda*L_S, G on L_U+eta*L_S,
    and D on -L_U. Returns losses, which do not establish generation quality.
    """
    counts = [
        positive_integer(value, name)
        for value, name in zip(
            [embedding_steps, supervised_steps, joint_steps],
            ["embedding_steps", "supervised_steps", "joint_steps"],
        )
    ]
    batch_size = positive_integer(batch_size, "batch_size")
    learning_rate = real_scalar(learning_rate, "learning_rate", minimum=0, strict=True)
    embedding_weight = real_scalar(embedding_weight, "embedding_weight", minimum=0)
    generator_weight = real_scalar(generator_weight, "generator_weight", minimum=0)
    parameter = next(model.parameters())
    data = torch.as_tensor(
        real_array(training_windows, "training_windows", 3),
        dtype=parameter.dtype,
        device=parameter.device,
    )
    model._check(data, "training_windows", model.features)
    if torch.any((data < 0) | (data > 1)):
        raise ValueError("Training observations must be scaled to [0,1].")
    generator = torch.Generator(device=parameter.device).manual_seed(seed)
    groups = [
        list(model.embedding.parameters()) + list(model.recovery.parameters()),
        list(model.generator_cell.parameters())
        + list(model.generator_output.parameters()),
        list(model.discriminator.parameters()),
    ]
    optimizers = [torch.optim.Adam(group, lr=learning_rate) for group in groups]
    history = []
    for phase, count in enumerate(counts):
        for _ in range(count):
            index = torch.randint(
                len(data), (batch_size,), generator=generator, device=data.device
            )
            observed = data[index]
            noise = torch.randn(
                batch_size,
                data.shape[1],
                model.hidden_size,
                generator=generator,
                dtype=data.dtype,
                device=data.device,
            )
            updates = [0] if phase == 0 else [1] if phase == 1 else [0, 1, 2]
            for group in updates:
                model.zero_grad(set_to_none=True)
                losses = model.losses(observed, noise)
                if group == 0:
                    objective = losses["reconstruction"] + (
                        embedding_weight * losses["supervised"] if phase == 2 else 0
                    )
                elif group == 1:
                    objective = generator_weight * losses["supervised"] + (
                        losses["generator_adversarial"] if phase == 2 else 0
                    )
                else:
                    objective = losses["discriminator"]
                objective.backward()
                if not torch.isfinite(objective) or any(
                    p.grad is not None and not torch.isfinite(p.grad).all()
                    for p in groups[group]
                ):
                    raise RuntimeError(
                        "TimeGAN objective or gradients became nonfinite."
                    )
                optimizers[group].step()
                if any(not torch.isfinite(p).all() for p in groups[group]):
                    raise RuntimeError("TimeGAN parameters became nonfinite.")
            history.append(
                {
                    "phase": ("embedding", "supervised", "joint")[phase],
                    **{key: value.detach().item() for key, value in losses.items()},
                }
            )
    return history


def timegan_quality_report(
    real_training, real_validation, synthetic_training, synthetic_validation
):
    """Evaluate held-out temporal, ridge-predictive and logistic-discrimination scores.

    All arrays are windows in the same units. Real validation must be from a
    disjoint chronological segment; synthetic validation uses fresh noise.
    Report descriptive lag-one correlation gaps, train-synthetic/test-real
    next-step MSE, real-trained MSE, and held-out balanced real/fake accuracy.
    These light diagnostics are not the paper's learned GRU evaluation or a
    certification of realistic financial dynamics. No uncertainty is asserted
    for overlapping windows, and no metric selects model parameters here.
    """
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    arrays = [
        real_array(value, "windows", 3)
        for value in (
            real_training,
            real_validation,
            synthetic_training,
            synthetic_validation,
        )
    ]
    if (
        any(
            array.shape[1:] != arrays[0].shape[1:] or len(array) < 2 for array in arrays
        )
        or arrays[0].shape[1] < 2
    ):
        raise ValueError(
            "All window shapes must agree and need at least two samples/times."
        )
    rt, rv, st, sv = arrays

    def lag(array):
        left, right = array[:, :-1].reshape(-1, array.shape[-1]), array[:, 1:].reshape(
            -1, array.shape[-1]
        )
        left, right = left - left.mean(axis=0), right - right.mean(axis=0)
        denominator = np.sqrt((left**2).sum(axis=0) * (right**2).sum(axis=0))
        return np.divide(
            (left * right).sum(axis=0),
            denominator,
            out=np.full(array.shape[-1], np.nan),
            where=denominator > 0,
        )

    def predictive(train):
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(
            train[:, :-1].reshape(len(train), -1), train[:, -1]
        )
        prediction = np.asarray(model.predict(rv[:, :-1].reshape(len(rv), -1))).reshape(
            rv[:, -1].shape
        )
        return float(np.mean((prediction - rv[:, -1]) ** 2))

    train_x = np.concatenate([rt.reshape(len(rt), -1), st.reshape(len(st), -1)])
    train_y = np.r_[np.ones(len(rt)), np.zeros(len(st))]
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=1000, random_state=0),
    ).fit(train_x, train_y)
    test_x = np.concatenate([rv.reshape(len(rv), -1), sv.reshape(len(sv), -1)])
    test_y = np.r_[np.ones(len(rv)), np.zeros(len(sv))]
    return {
        "real_lag_one_correlation": lag(rv),
        "synthetic_lag_one_correlation": lag(sv),
        "lag_one_absolute_gap": np.abs(lag(rv) - lag(sv)),
        "synthetic_trained_prediction_mse": predictive(st),
        "real_trained_prediction_mse": predictive(rt),
        "balanced_discriminator_accuracy": float(
            balanced_accuracy_score(test_y, classifier.predict(test_x))
        ),
    }
