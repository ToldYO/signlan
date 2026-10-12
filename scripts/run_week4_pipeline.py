"""Week 4: train ST-GCN/Transformer models and export deep classifiers to ONNX."""
import argparse
import json
import os
import sys
import logging
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data_ingestion import load_vocabulary, generate_benchmark_dataset
from src.dataset import create_dataloaders
from src.preprocessor import preprocess_sequence
from src.models import BiGRUClassifier, TemporalConvNet, STGCNClassifier, TransformerClassifier, count_parameters
from src.trainer import ASLModelTrainer
from src.profiler import profile_model_latency
from src.export import export_onnx_model, quantize_onnx_dynamic

logger = logging.getLogger(__name__)
CACHE_FILE = "data/processed/cached_trajectories.npz"


def load_cached_sequences(cache_path=CACHE_FILE):
    """Loads the Week 3 cache; extraction must be performed by Week 3 first."""
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"{cache_path} was not found. Run scripts/run_week3_pipeline.py first.")
    data = np.load(cache_path, allow_pickle=True)
    sequences = [data[f"arr_{i}"] for i in range(len(data.files) - 2)]
    return sequences, data["labels"].tolist(), data["splits"].tolist()


def _split(sequences, labels, splits, partition):
    return [x for x, p in zip(sequences, splits) if p == partition], [y for y, p in zip(labels, splits) if p == partition]


def main():
    parser = argparse.ArgumentParser(description="Run Week 4 ASL experiments.")
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--quantize", action="store_true", help="Also create optional ONNX Runtime INT8 files.")
    args = parser.parse_args()
    vocab = load_vocabulary("config/vocabulary.json")
    raw, labels, splits = load_cached_sequences()
    sequences = [preprocess_sequence(seq, target_frames=30, num_hands=1) for seq in raw]
    train, train_y = _split(sequences, labels, splits, "train")
    val, val_y = _split(sequences, labels, splits, "val")
    test, test_y = _split(sequences, labels, splits, "test")
    loaders = create_dataloaders((train, train_y), (val, val_y), (test, test_y), batch_size=16)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    experiments = {
        "stgcn": STGCNClassifier(input_dim=63, num_classes=len(vocab)),
        "transformer": TransformerClassifier(input_dim=63, num_classes=len(vocab)),
    }
    results = {}
    for name, model in experiments.items():
        print(f"Training {name} ({count_parameters(model):,} parameters)")
        trainer = ASLModelTrainer(model, num_classes=len(vocab), device=device, lr=1.5e-3)
        checkpoint = f"models/{name}_best.pth"
        trainer.fit(loaders["train"], loaders["val"], epochs=args.epochs, patience=15, save_path=checkpoint, target_names=vocab)
        metrics = trainer.evaluate(loaders["test"], target_names=vocab)
        latency = profile_model_latency(model, test[0], is_pytorch=True, iterations=200)
        results[name] = {
            "top_1_accuracy": metrics["top_1_accuracy"], "top_5_accuracy": metrics["top_5_accuracy"],
            "macro_f1": metrics["macro_f1"], "parameter_count": count_parameters(model),
            "latency_p50_ms": latency["median_latency_ms"], "latency_p95_ms": latency["p95_latency_ms"],
            "throughput_fps": latency["throughput_fps"],
        }
        if not args.skip_export:
            onnx_path = export_onnx_model(model, f"models/{name}.onnx", 63)
            results[name]["onnx_path"] = onnx_path
            if args.quantize:
                try:
                    results[name].update(quantize_onnx_dynamic(onnx_path, f"models/{name}_int8.onnx"))
                except RuntimeError as error:
                    logger.warning("%s", error)
    with open("data/processed/benchmark_results_week4.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print("Week 4 complete. Results: data/processed/benchmark_results_week4.json")


if __name__ == "__main__":
    main()
