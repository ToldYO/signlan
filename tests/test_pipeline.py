"""
Comprehensive Unit Test Suite for ASL Recognition Pipeline.
Tests spatial normalization, missing-frame interpolation, temporal resampling,
PyTorch Dataset/DataLoader shapes, feature extraction, and ML baselines.
"""

import os
import sys
import unittest
import numpy as np
import torch

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessor import (
    normalize_spatial_coordinates,
    resample_temporal_trajectory,
    preprocess_sequence,
)
from src.keypoint_extractor import MediaPipeKeypointExtractor
from src.dataset import ASLKeypointDataset, ASLKeypointAugmentation, create_dataloaders
from src.feature_extraction import extract_trajectory_features_single, extract_trajectory_features_batch
from src.baselines import compute_top_k_accuracy, ASLBaselineBenchmarks


class TestASLPipeline(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        torch.manual_seed(42)

    def test_spatial_translation_invariance(self):
        """Zero-centering at wrist ensures translation invariance."""
        pts = np.random.uniform(0.1, 0.9, (10, 21, 3)).astype(np.float32)
        shift = np.array([0.5, -0.7, 1.2], dtype=np.float32)
        shifted_pts = pts + shift

        norm1 = normalize_spatial_coordinates(pts.reshape(10, 63))
        norm2 = normalize_spatial_coordinates(shifted_pts.reshape(10, 63))

        max_err = np.max(np.abs(norm1 - norm2))
        self.assertLess(max_err, 1e-4, f"Translation invariance error too high: {max_err}")

    def test_spatial_scale_invariance(self):
        """MCP distance division ensures scale invariance."""
        pts = np.random.uniform(0.1, 0.9, (5, 21, 3)).astype(np.float32)
        scaled_pts = pts * 3.75

        norm1 = normalize_spatial_coordinates(pts.reshape(5, 63))
        norm2 = normalize_spatial_coordinates(scaled_pts.reshape(5, 63))

        max_err = np.max(np.abs(norm1 - norm2))
        self.assertLess(max_err, 1e-4, f"Scale invariance error too high: {max_err}")

    def test_temporal_resampling(self):
        """Arbitrary sequence lengths (15 to 75) correctly resample to T=30."""
        for length in [15, 23, 30, 48, 75]:
            raw_seq = np.random.randn(length, 63).astype(np.float32)
            resampled = resample_temporal_trajectory(raw_seq, target_frames=30)
            self.assertEqual(resampled.shape, (30, 63))
            self.assertFalse(np.isnan(resampled).any())

    def test_missing_frame_interpolation(self):
        """Linear interpolation correctly reconstructs interior, leading, and trailing missing frames."""
        T = 30
        sim_data = [np.random.randn(21, 3).astype(np.float32) for _ in range(T)]
        valid_mask = np.ones(T, dtype=bool)

        # Missing leading frames (0, 1), middle frames (10..15), trailing frames (28, 29)
        missing_indices = [0, 1, 10, 11, 12, 13, 14, 15, 28, 29]
        for idx in missing_indices:
            sim_data[idx] = None
            valid_mask[idx] = False

        interpolated = MediaPipeKeypointExtractor.interpolate_missing_frames(
            sim_data, valid_mask, num_hands=1
        )
        self.assertEqual(interpolated.shape, (T, 63))
        self.assertFalse(np.isnan(interpolated).any())
        self.assertFalse(np.isinf(interpolated).any())

    def test_pytorch_dataset_and_loader(self):
        """PyTorch DataLoader yields batches of shape (batch_size, 30, 63)."""
        trajectories = [np.random.randn(25, 63).astype(np.float32) for _ in range(20)]
        labels = [i % 5 for i in range(20)]

        loaders = create_dataloaders(
            train_data=(trajectories, labels),
            batch_size=4,
            num_hands=1,
            target_frames=30,
        )

        train_loader = loaders["train"]
        batch_x, batch_y = next(iter(train_loader))

        self.assertEqual(batch_x.shape, (4, 30, 63))
        self.assertEqual(batch_y.shape, (4,))
        self.assertEqual(batch_x.dtype, torch.float32)
        self.assertEqual(batch_y.dtype, torch.int64)

    def test_augmentation(self):
        """Data augmentation produces valid perturbed trajectories of same shape."""
        seq = np.random.randn(30, 63).astype(np.float32)
        augmenter = ASLKeypointAugmentation()
        aug_seq = augmenter(seq)

        self.assertEqual(aug_seq.shape, (30, 63))
        self.assertFalse(np.allclose(seq, aug_seq))  # Must have perturbed
        self.assertFalse(np.isnan(aug_seq).any())

    def test_feature_extraction(self):
        """Trajectory statistics extraction yields non-empty 1D feature vectors."""
        seq = np.random.randn(30, 63).astype(np.float32)
        feat = extract_trajectory_features_single(seq)

        self.assertEqual(len(feat.shape), 1)
        self.assertGreater(feat.shape[0], 100)
        self.assertFalse(np.isnan(feat).any())

    def test_top_k_accuracy(self):
        """Top-K accuracy computation works correctly."""
        y_true = np.array([0, 1, 2])
        # Probabilities where class 0 is top-1, class 1 is rank-2, class 2 is rank-4
        probs = np.array([
            [0.9, 0.05, 0.05],
            [0.6, 0.4, 0.0],
            [0.5, 0.3, 0.2],
        ])
        top_1 = compute_top_k_accuracy(y_true, probs, k=1)
        top_2 = compute_top_k_accuracy(y_true, probs, k=2)

        self.assertAlmostEqual(top_1, 1.0 / 3.0)
        self.assertAlmostEqual(top_2, 2.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
