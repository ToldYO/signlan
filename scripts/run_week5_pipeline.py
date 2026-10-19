"""Week 5: adaptive motion graph modeling, calibration, and robustness evaluation."""
import argparse
import json
import os
import sys
from typing import Dict, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.calibration import TemperatureScaler, corrupt_keypoints, expected_calibration_error
from src.data_ingestion import load_vocabulary
from src.dataset import create_dataloaders
from src.models import AdaptiveMotionSTGCNClassifier, count_parameters
from src.preprocessor import preprocess_sequence
from src.profiler import profile_model_latency
from src.trainer import ASLModelTrainer
from scripts.run_week4_pipeline import load_cached_sequences, _split


@torch.no_grad()
def collect_logits(model, loader, device) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    logits, targets = [], []
    for batch_x, batch_y in loader:
        logits.append(model(batch_x.to(device)).cpu())
        targets.append(batch_y.cpu())
    return torch.cat(logits), torch.cat(targets)


@torch.no_grad()
def robustness_accuracy(model, loader, device, kind: str, severity: float) -> float:
    model.eval()
    correct = total = 0
    for batch_x, batch_y in loader:
        prediction = model(corrupt_keypoints(batch_x.to(device), kind, severity)).argmax(1).cpu()
        correct += int((prediction == batch_y).sum())
        total += len(batch_y)
    return correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser(description="Run Week 5 reliability experiments.")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    vocab = load_vocabulary("config/vocabulary.json")
    raw, labels, splits = load_cached_sequences()
    sequences = [preprocess_sequence(x, target_frames=30, num_hands=1) for x in raw]
    train = _split(sequences, labels, splits, "train")
    validation = _split(sequences, labels, splits, "val")
    test = _split(sequences, labels, splits, "test")
    loaders = create_dataloaders(train, validation, test, batch_size=16)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AdaptiveMotionSTGCNClassifier(input_dim=63, num_classes=len(vocab))
    trainer = ASLModelTrainer(model, num_classes=len(vocab), device=device, lr=1.5e-3)
    trainer.fit(loaders["train"], loaders["val"], epochs=args.epochs, patience=18,
                save_path="models/adaptive_motion_stgcn_best.pth", target_names=vocab)
    clean = trainer.evaluate(loaders["test"], target_names=vocab)
    val_logits, val_targets = collect_logits(model, loaders["val"], device)
    test_logits, test_targets = collect_logits(model, loaders["test"], device)
    scaler = TemperatureScaler().fit(val_logits, val_targets)
    uncalibrated = torch.softmax(test_logits, dim=1).numpy()
    calibrated = scaler.probabilities(test_logits).numpy()
    results: Dict = {
        "model": "adaptive_motion_stgcn", "parameter_count": count_parameters(model),
        "clean": {key: clean[key] for key in ("top_1_accuracy", "top_5_accuracy", "macro_f1")},
        "calibration": {"temperature": scaler.temperature,
            "ece_before": expected_calibration_error(uncalibrated, test_targets.numpy()),
            "ece_after": expected_calibration_error(calibrated, test_targets.numpy())},
        "robustness_top_1": {
            "gaussian_noise_0.02": robustness_accuracy(model, loaders["test"], device, "gaussian_noise", 0.02),
            "gaussian_noise_0.05": robustness_accuracy(model, loaders["test"], device, "gaussian_noise", 0.05),
            "frame_dropout_0.10": robustness_accuracy(model, loaders["test"], device, "frame_dropout", 0.10),
            "frame_dropout_0.20": robustness_accuracy(model, loaders["test"], device, "frame_dropout", 0.20)},
        "latency": profile_model_latency(model, test[0][0], is_pytorch=True, iterations=200),
    }
    os.makedirs("data/processed", exist_ok=True)
    with open("data/processed/benchmark_results_week5.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
