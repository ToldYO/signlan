"""Confidence calibration and robustness metrics for ASL classifiers."""

import numpy as np
import torch
import torch.nn.functional as F


def expected_calibration_error(probabilities: np.ndarray, targets: np.ndarray, bins: int = 10) -> float:
    """Computes equal-width expected calibration error (ECE)."""
    if bins < 1:
        raise ValueError("bins must be positive")
    probabilities = np.asarray(probabilities)
    targets = np.asarray(targets)
    if probabilities.ndim != 2 or len(probabilities) != len(targets):
        raise ValueError("probabilities must be (N, C) and match targets")
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == targets
    ece = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        selected = (confidence > edges[index]) & (confidence <= edges[index + 1])
        if selected.any():
            ece += selected.mean() * abs(correct[selected].mean() - confidence[selected].mean())
    return float(ece)


class TemperatureScaler:
    """Fits one positive temperature on held-out logits."""

    def __init__(self, temperature: float = 1.0):
        self.temperature = float(temperature)

    def fit(self, logits: torch.Tensor, targets: torch.Tensor, max_iter: int = 50) -> "TemperatureScaler":
        if logits.ndim != 2 or targets.ndim != 1 or len(logits) != len(targets):
            raise ValueError("logits must be (N, C) and match one-dimensional targets")
        log_temperature = torch.zeros(1, device=logits.device, requires_grad=True)
        optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            loss = F.cross_entropy(logits / log_temperature.exp().clamp_min(1e-4), targets)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.temperature = float(log_temperature.detach().exp().clamp(0.05, 20.0).item())
        return self

    def probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.softmax(logits / self.temperature, dim=1)


def corrupt_keypoints(x: torch.Tensor, kind: str, severity: float) -> torch.Tensor:
    """Applies deterministic evaluation-time coordinate corruptions."""
    if severity < 0:
        raise ValueError("severity must be non-negative")
    if kind == "gaussian_noise":
        generator = torch.Generator(device=x.device).manual_seed(0)
        noise = torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)
        return x + noise * severity
    if kind == "frame_dropout":
        result = x.clone()
        stride = max(1, round(1.0 / max(severity, 1e-6)))
        result[:, ::stride] = 0
        return result
    raise ValueError(f"Unknown corruption: {kind}")
