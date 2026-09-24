"""
Week 2 End-to-End Execution Script:
1. Spatial Normalization: Zero-centers at wrist (Landmark 0) and scales by Euclidean
   distance to middle MCP joint (Landmark 9) to achieve scale and position invariance.
2. Formats and validates PyTorch Dataset & DataLoader serving (batch_size, 30, 63) tensors.
3. Feature Engineering: Extracts kinematic trajectory statistics (mean, variance, velocity,
   acceleration, displacement, path lengths).
4. Trains Non-Deep-Learning Baselines: Random Forest and Support Vector Classifier (SVC).
5. Evaluates Top-1 Categorical Accuracy, Top-5 Accuracy, and Macro-Averaged F1-Score.
6. Persists trained baseline models and benchmark metrics to disk.
"""

import os
import sys
import json
import logging
import numpy as np
import torch

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_ingestion import load_vocabulary, generate_benchmark_dataset
from src.keypoint_extractor import MediaPipeKeypointExtractor
from src.preprocessor import (
    normalize_spatial_coordinates,
    resample_temporal_trajectory,
    preprocess_sequence,
    WRIST_IDX,
    MIDDLE_FINGER_MCP_IDX,
)
from src.dataset import ASLKeypointDataset, create_dataloaders
from src.feature_extraction import extract_trajectory_features_batch
from src.baselines import ASLBaselineBenchmarks

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def verify_spatial_invariance():
    """
    Formally verifies translation and scale invariance of the spatial normalization.
    """
    print("\n--- Verifying Translation and Scale Invariance ---")
    np.random.seed(42)
    # Generate 1 hand with 21 3D points
    base_pts = np.random.uniform(0.2, 0.8, (1, 21, 3)).astype(np.float32)

    # Apply arbitrary translation (+0.4, -0.3, +0.1) and scaling (x2.5)
    translated_scaled = (base_pts + np.array([0.4, -0.3, 0.1])) * 2.5

    norm_base = normalize_spatial_coordinates(base_pts, num_hands=1)
    norm_transformed = normalize_spatial_coordinates(translated_scaled, num_hands=1)

    diff = np.max(np.abs(norm_base - norm_transformed))
    print(f"  - Max absolute difference after translation + scale shift: {diff:.2e}")
    assert diff < 1e-4, f"Spatial normalization failed invariance check! Diff: {diff}"
    print("  - [PASS] Spatial Normalization is strictly scale- and position-invariant.")


def main():
    print("=" * 70)
    print("   WEEK 2 PIPELINE: PREPROCESSING, PYTORCH DATASET & ML BASELINES")
    print("=" * 70)

    # -------------------------------------------------------------
    # 1. Spatial Normalization & Invariance Proof
    # -------------------------------------------------------------
    print("\n[Step 1/5] Testing Spatial Normalization (Wrist zero-center + MCP scale)...")
    verify_spatial_invariance()

    # -------------------------------------------------------------
    # 2. Benchmark Dataset Preparation
    # -------------------------------------------------------------
    print("\n[Step 2/5] Preparing Sign Dataset across 30 Target Vocabulary Words...")
    vocab = load_vocabulary("config/vocabulary.json")
    print(f"  - Vocabulary Size: {len(vocab)} classes")

    # Generate benchmark dataset: 5 samples per word = 150 video clips
    manifest = generate_benchmark_dataset(vocab=vocab, samples_per_word=5, output_dir="data/raw")
    print(f"  - Ingested {len(manifest)} video samples across {len(vocab)} classes.")

    # Extract keypoints for all samples
    print("  - Extracting MediaPipe HandLandmarker keypoints...")
    extractor = MediaPipeKeypointExtractor(num_hands=1)

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
    print(f"  - Successfully extracted keypoint trajectories for {len(raw_sequences)} clips.")

    # -------------------------------------------------------------
    # 3. Preprocessing & PyTorch Dataset / DataLoader Validation
    # -------------------------------------------------------------
    print("\n[Step 3/5] Building & Validating PyTorch Dataset and DataLoader...")
    # Preprocess all sequences: spatial norm + temporal resample to T=30
    processed_sequences = [
        preprocess_sequence(seq, target_frames=30, num_hands=1) for seq in raw_sequences
    ]

    # Split into train, val, and test partitions
    train_data = (
        [s for s, sp in zip(processed_sequences, splits) if sp == "train"],
        [l for l, sp in zip(labels, splits) if sp == "train"],
    )
    val_data = (
        [s for s, sp in zip(processed_sequences, splits) if sp == "val"],
        [l for l, sp in zip(labels, splits) if sp == "val"],
    )
    test_data = (
        [s for s, sp in zip(processed_sequences, splits) if sp == "test"],
        [l for l, sp in zip(labels, splits) if sp == "test"],
    )

    print(f"  - Train Partition: {len(train_data[1])} samples")
    print(f"  - Val Partition:   {len(val_data[1])} samples")
    print(f"  - Test Partition:  {len(test_data[1])} samples")

    # Construct DataLoaders
    batch_size = 16
    dataloaders = create_dataloaders(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        batch_size=batch_size,
        num_hands=1,
        target_frames=30,
    )

    # Validate tensor batch shapes from PyTorch DataLoader
    sample_batch_x, sample_batch_y = next(iter(dataloaders["train"]))
    print(f"  - Sample PyTorch Batch Shape X: {tuple(sample_batch_x.shape)} (batch_size, T=30, D=63)")
    print(f"  - Sample PyTorch Batch Shape Y: {tuple(sample_batch_y.shape)} (batch_size,)")
    assert sample_batch_x.shape == (batch_size, 30, 63), "PyTorch DataLoader shape mismatch!"
    print("  - [PASS] PyTorch DataLoader serves valid batches matching exact specification.")

    # -------------------------------------------------------------
    # 4. Trajectory Feature Engineering
    # -------------------------------------------------------------
    print("\n[Step 4/5] Extracting Aggregated Trajectory Statistics (Mean, Var, Velocity)...")
    X_train = extract_trajectory_features_batch(train_data[0])
    y_train = np.array(train_data[1], dtype=np.int64)

    X_val = extract_trajectory_features_batch(val_data[0])
    y_val = np.array(val_data[1], dtype=np.int64)

    X_test = extract_trajectory_features_batch(test_data[0])
    y_test = np.array(test_data[1], dtype=np.int64)

    print(f"  - Extracted feature matrix shape: X_train={X_train.shape}, X_test={X_test.shape}")
    print(f"  - Total engineered trajectory features per instance: {X_train.shape[1]}")

    # -------------------------------------------------------------
    # 5. Non-Deep-Learning Baselines: Random Forest & SVC
    # -------------------------------------------------------------
    print("\n[Step 5/5] Training and Benchmarking Baseline Classifiers...")
    benchmarks = ASLBaselineBenchmarks(vocab=vocab)

    # Train Random Forest
    rf_model = benchmarks.train_random_forest(X_train, y_train, n_estimators=150, max_depth=16)

    # Train Support Vector Classifier
    svc_model = benchmarks.train_svc(X_train, y_train, C=3.0)

    # Evaluate on Test Split
    print("\n" + "-" * 70)
    print("                    BENCHMARK EVALUATION RESULTS")
    print("-" * 70)
    results = benchmarks.evaluate_all(X_test, y_test)

    # Display clear comparison table
    print("\n" + "=" * 70)
    print(f"{'Model':<25} | {'Top-1 Acc':<12} | {'Top-5 Acc':<12} | {'Macro F1':<12}")
    print("-" * 70)
    for model_name, res in results.items():
        print(
            f"{model_name.replace('_', ' ').title():<25} | "
            f"{res['top_1_accuracy']*100:>10.2f}% | "
            f"{res['top_5_accuracy']*100:>10.2f}% | "
            f"{res['macro_f1']*100:>10.2f}%"
        )
    print("=" * 70)

    # Persist models and metrics
    benchmarks.save_model("random_forest", "models/random_forest_baseline.joblib")
    benchmarks.save_model("svc", "models/svc_baseline.joblib")

    metrics_export_path = "data/processed/benchmark_results.json"
    os.makedirs(os.path.dirname(metrics_export_path), exist_ok=True)
    with open(metrics_export_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nPersisted baseline models to 'models/' and metrics to '{metrics_export_path}'.")
    print("\n" + "=" * 70)
    print("WEEK 2 MILESTONES COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
