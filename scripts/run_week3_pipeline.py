"""
Week 3 End-to-End Pipeline: Deep Temporal Sequence Modeling and Ablation Study.

Executes:
1. Data Ingestion & Preprocessing: Loads/caches keypoint trajectories for 30 dynamic ASL words.
2. PyTorch DataLoader Pipeline: Serves (batch_size, 30, 63) tensors with kinematic augmentations.
3. Baseline Training & Evaluation: Random Forest and Support Vector Classifier (SVC).
4. Deep Temporal Modeling:
   - Bidirectional GRU (BiGRU) with Temporal Attention Pooling.
   - 1D Temporal Convolutional Network (1D-TCN) with Dilated Residual Blocks (d in [1, 2, 4, 8]).
5. Inference Latency & Profiling: Measures mean, p50, p95, p99 latency and CPU throughput for all models.
6. Comparative Ablation Study: Benchmarks Top-1, Top-5, Macro F1, params, and latency.
7. Persists checkpoints to models/ and exported metrics to data/processed/benchmark_results_week3.json.
"""

import os
import sys
import json
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_ingestion import load_vocabulary, generate_benchmark_dataset
from src.keypoint_extractor import MediaPipeKeypointExtractor
from src.preprocessor import preprocess_sequence
from src.dataset import ASLKeypointDataset, create_dataloaders
from src.feature_extraction import extract_trajectory_features_batch
from src.baselines import ASLBaselineBenchmarks
from src.models import BiGRUClassifier, TemporalConvNet, count_parameters
from src.trainer import ASLModelTrainer
from src.profiler import profile_model_latency

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CACHE_FILE = "data/processed/cached_trajectories.npz"


def load_or_extract_dataset(
    manifest: List[Dict[str, Any]],
    cache_path: str = CACHE_FILE,
    num_hands: int = 1,
) -> Tuple[List[np.ndarray], List[int], List[str]]:
    """
    Loads trajectories from disk cache if present, otherwise extracts with MediaPipe
    and caches the results for fast subsequent runs.
    """
    if os.path.exists(cache_path):
        logger.info(f"Loading cached keypoint trajectories from {cache_path}...")
        data = np.load(cache_path, allow_pickle=True)
        raw_sequences = [data[f"arr_{i}"] for i in range(len(data.files) - 2)]
        labels = data["labels"].tolist()
        splits = data["splits"].tolist()
        logger.info(f"Loaded {len(raw_sequences)} sequences from cache.")
        return raw_sequences, labels, splits

    logger.info("Extracting MediaPipe HandLandmarker keypoints from video clips...")
    extractor = MediaPipeKeypointExtractor(num_hands=num_hands)

    raw_sequences = []
    labels = []
    splits = []

    for item in manifest:
        try:
            ext = extractor.extract_from_video(item["video_path"], interpolate_missing=True)
            raw_sequences.append(ext["trajectory"])
            labels.append(item["label_id"])
            splits.append(item["split"])
        except Exception as e:
            logger.warning(f"Error extracting {item['video_path']}: {e}")

    extractor.close()

    # Cache to disk
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    save_dict = {f"arr_{i}": seq for i, seq in enumerate(raw_sequences)}
    save_dict["labels"] = np.array(labels)
    save_dict["splits"] = np.array(splits)
    np.savez_compressed(cache_path, **save_dict)
    logger.info(f"Cached {len(raw_sequences)} trajectories to {cache_path}.")

    return raw_sequences, labels, splits


def main():
    print("=" * 75)
    print("   WEEK 3 PIPELINE: DEEP TEMPORAL SEQUENCE MODELING & ABLATION STUDY")
    print("=" * 75)

    # 1. Environment & Vocabulary
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Step 1/6] Initializing Environment...")
    print(f"  - Device: {device}")
    vocab = load_vocabulary("config/vocabulary.json")
    num_classes = len(vocab)
    print(f"  - Vocabulary: {num_classes} dynamic ASL words")

    # 2. Ingestion & Data Preparation
    print(f"\n[Step 2/6] Preparing Dataset & Preprocessing Sequences...")
    manifest = generate_benchmark_dataset(vocab=vocab, samples_per_word=5, output_dir="data/raw")
    raw_sequences, labels, splits = load_or_extract_dataset(manifest, num_hands=1)

    # Spatial normalization + temporal resampling to T=30
    processed_sequences = [
        preprocess_sequence(seq, target_frames=30, num_hands=1) for seq in raw_sequences
    ]

    train_trajs = [s for s, sp in zip(processed_sequences, splits) if sp == "train"]
    train_labels = [l for l, sp in zip(labels, splits) if sp == "train"]

    val_trajs = [s for s, sp in zip(processed_sequences, splits) if sp == "val"]
    val_labels = [l for l, sp in zip(labels, splits) if sp == "val"]

    test_trajs = [s for s, sp in zip(processed_sequences, splits) if sp == "test"]
    test_labels = [l for l, sp in zip(labels, splits) if sp == "test"]

    print(f"  - Partition Sizes: Train={len(train_labels)}, Val={len(val_labels)}, Test={len(test_labels)}")

    # PyTorch DataLoaders
    batch_size = 16
    dataloaders = create_dataloaders(
        train_data=(train_trajs, train_labels),
        val_data=(val_trajs, val_labels),
        test_data=(test_trajs, test_labels),
        batch_size=batch_size,
        num_hands=1,
        target_frames=30,
    )

    # Engineered Trajectory Features for ML Baselines
    print("\n[Step 3/6] Extracting Kinematic Features for ML Baselines...")
    X_train = extract_trajectory_features_batch(train_trajs)
    y_train = np.array(train_labels, dtype=np.int64)

    X_test = extract_trajectory_features_batch(test_trajs)
    y_test = np.array(test_labels, dtype=np.int64)
    print(f"  - Extracted {X_train.shape[1]} engineered trajectory features per instance.")

    # 3. Train & Evaluate Baselines
    print("\n[Step 4/6] Training Non-Deep-Learning Baselines (Random Forest & SVC)...")
    benchmarks = ASLBaselineBenchmarks(vocab=vocab)
    rf_model = benchmarks.train_random_forest(X_train, y_train, n_estimators=150, max_depth=16)
    svc_model = benchmarks.train_svc(X_train, y_train, C=3.0)

    baseline_results = benchmarks.evaluate_all(X_test, y_test)
    rf_results = baseline_results["random_forest"]
    svc_results = baseline_results["svc"]

    # 4. Train Deep Learning Architectures
    print("\n[Step 5/6] Training Deep Temporal Sequence Models...")

    # A. Bidirectional GRU with Temporal Attention Pooling
    print("\n  --- [Model 1/2] Training Bidirectional GRU (BiGRU + Temporal Attention) ---")
    bigru_model = BiGRUClassifier(
        input_dim=63,
        num_classes=num_classes,
        hidden_dim=128,
        num_layers=2,
        dropout=0.3,
        attention_dim=64,
        bidirectional=True,
    )
    bigru_params = count_parameters(bigru_model)
    print(f"  - BiGRU Trainable Parameters: {bigru_params:,}")

    bigru_trainer = ASLModelTrainer(
        model=bigru_model,
        num_classes=num_classes,
        device=device,
        lr=1.5e-3,
        weight_decay=1e-4,
        label_smoothing=0.05,
    )

    bigru_save_path = "models/bigru_best.pth"
    bigru_trainer.fit(
        train_loader=dataloaders["train"],
        val_loader=dataloaders["val"],
        epochs=45,
        patience=15,
        save_path=bigru_save_path,
        target_names=vocab,
        verbose=True,
    )

    bigru_eval = bigru_trainer.evaluate(dataloaders["test"], target_names=vocab)
    print(f"  - BiGRU Test Top-1 Acc: {bigru_eval['top_1_accuracy']*100:.2f}% | "
          f"Top-5 Acc: {bigru_eval['top_5_accuracy']*100:.2f}% | "
          f"Macro F1: {bigru_eval['macro_f1']*100:.2f}%")

    # B. Dilated 1D Temporal Convolutional Network (1D-TCN)
    print("\n  --- [Model 2/2] Training 1D Temporal Convolutional Network (1D-TCN) ---")
    tcn_model = TemporalConvNet(
        input_dim=63,
        num_classes=num_classes,
        num_channels=[64, 128, 128, 256],
        dilations=[1, 2, 4, 8],
        kernel_size=3,
        dropout=0.25,
    )
    tcn_params = count_parameters(tcn_model)
    print(f"  - 1D-TCN Trainable Parameters: {tcn_params:,} (Receptive Field: {tcn_model.receptive_field} frames)")

    tcn_trainer = ASLModelTrainer(
        model=tcn_model,
        num_classes=num_classes,
        device=device,
        lr=1.5e-3,
        weight_decay=1e-4,
        label_smoothing=0.05,
    )

    tcn_save_path = "models/tcn_best.pth"
    tcn_trainer.fit(
        train_loader=dataloaders["train"],
        val_loader=dataloaders["val"],
        epochs=45,
        patience=15,
        save_path=tcn_save_path,
        target_names=vocab,
        verbose=True,
    )

    tcn_eval = tcn_trainer.evaluate(dataloaders["test"], target_names=vocab)
    print(f"  - 1D-TCN Test Top-1 Acc: {tcn_eval['top_1_accuracy']*100:.2f}% | "
          f"Top-5 Acc: {tcn_eval['top_5_accuracy']*100:.2f}% | "
          f"Macro F1: {tcn_eval['macro_f1']*100:.2f}%")

    # 5. Latency and Profiling Benchmarking
    print("\n[Step 6/6] Profiling CPU Latency Percentiles & Throughput (200 Iterations)...")
    sample_seq = test_trajs[0]  # (30, 63)
    sample_feat = X_test[0]     # (714,)

    rf_profile = profile_model_latency(rf_model, sample_feat, is_pytorch=False, iterations=200)
    svc_profile = profile_model_latency(svc_model, sample_feat, is_pytorch=False, iterations=200)
    bigru_profile = profile_model_latency(bigru_model, sample_seq, is_pytorch=True, iterations=200)
    tcn_profile = profile_model_latency(tcn_model, sample_seq, is_pytorch=True, iterations=200)

    # 6. Consolidated Comparative Ablation Results
    ablation_summary = {
        "random_forest": {
            "top_1_accuracy": rf_results["top_1_accuracy"],
            "top_5_accuracy": rf_results["top_5_accuracy"],
            "macro_f1": rf_results["macro_f1"],
            "weighted_f1": rf_results["weighted_f1"],
            "parameter_count": None,
            "latency_p50_ms": rf_profile["median_latency_ms"],
            "latency_p95_ms": rf_profile["p95_latency_ms"],
            "latency_p99_ms": rf_profile["p99_latency_ms"],
            "mean_latency_ms": rf_profile["mean_latency_ms"],
            "throughput_fps": rf_profile["throughput_fps"],
            "model_type": "Ensemble (150 Trees)",
        },
        "svc": {
            "top_1_accuracy": svc_results["top_1_accuracy"],
            "top_5_accuracy": svc_results["top_5_accuracy"],
            "macro_f1": svc_results["macro_f1"],
            "weighted_f1": svc_results["weighted_f1"],
            "parameter_count": None,
            "latency_p50_ms": svc_profile["median_latency_ms"],
            "latency_p95_ms": svc_profile["p95_latency_ms"],
            "latency_p99_ms": svc_profile["p99_latency_ms"],
            "mean_latency_ms": svc_profile["mean_latency_ms"],
            "throughput_fps": svc_profile["throughput_fps"],
            "model_type": "Kernel SVM (RBF)",
        },
        "bigru_attention": {
            "top_1_accuracy": bigru_eval["top_1_accuracy"],
            "top_5_accuracy": bigru_eval["top_5_accuracy"],
            "macro_f1": bigru_eval["macro_f1"],
            "weighted_f1": bigru_eval["weighted_f1"],
            "parameter_count": bigru_params,
            "latency_p50_ms": bigru_profile["median_latency_ms"],
            "latency_p95_ms": bigru_profile["p95_latency_ms"],
            "latency_p99_ms": bigru_profile["p99_latency_ms"],
            "mean_latency_ms": bigru_profile["mean_latency_ms"],
            "throughput_fps": bigru_profile["throughput_fps"],
            "model_type": "Recurrent + Attention",
        },
        "tcn_1d": {
            "top_1_accuracy": tcn_eval["top_1_accuracy"],
            "top_5_accuracy": tcn_eval["top_5_accuracy"],
            "macro_f1": tcn_eval["macro_f1"],
            "weighted_f1": tcn_eval["weighted_f1"],
            "parameter_count": tcn_params,
            "latency_p50_ms": tcn_profile["median_latency_ms"],
            "latency_p95_ms": tcn_profile["p95_latency_ms"],
            "latency_p99_ms": tcn_profile["p99_latency_ms"],
            "mean_latency_ms": tcn_profile["mean_latency_ms"],
            "throughput_fps": tcn_profile["throughput_fps"],
            "model_type": "Dilated 1D-CNN (Residual)",
        },
    }

    # Print Clean Ablation Table
    print("\n" + "=" * 90)
    print("                    WEEK 3 COMPREHENSIVE ABLATION BENCHMARKS")
    print("=" * 90)
    header = f"{'Architecture':<22} | {'Top-1':<9} | {'Top-5':<9} | {'Macro F1':<9} | {'Params':<10} | {'p50 (ms)':<9} | {'Throughput':<11}"
    print(header)
    print("-" * 90)

    for name, m in ablation_summary.items():
        p_str = f"{m['parameter_count']:,}" if m['parameter_count'] else "N/A"
        name_display = name.replace("_", " ").title()
        print(
            f"{name_display:<22} | "
            f"{m['top_1_accuracy']*100:>7.2f}% | "
            f"{m['top_5_accuracy']*100:>7.2f}% | "
            f"{m['macro_f1']*100:>7.2f}% | "
            f"{p_str:>10} | "
            f"{m['latency_p50_ms']:>7.2f}ms | "
            f"{m['throughput_fps']:>8.1f} FPS"
        )
    print("=" * 90)

    # Persist exported results
    results_export_path = "data/processed/benchmark_results_week3.json"
    with open(results_export_path, "w", encoding="utf-8") as f:
        json.dump(ablation_summary, f, indent=2)

    # Also update master benchmark_results.json
    master_results_path = "data/processed/benchmark_results.json"
    with open(master_results_path, "w", encoding="utf-8") as f:
        json.dump(ablation_summary, f, indent=2)

    print(f"\nSuccessfully persisted model weights to 'models/' and metrics to '{results_export_path}'.")
    print("=" * 75)
    print("WEEK 3 PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 75)


if __name__ == "__main__":
    main()
