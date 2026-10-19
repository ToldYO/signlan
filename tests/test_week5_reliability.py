"""Tests for the Week 5 adaptive model, calibration, and robustness tools."""
import os
import sys
import unittest
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models import AdaptiveMotionSTGCNClassifier
from src.calibration import TemperatureScaler, corrupt_keypoints, expected_calibration_error


class TestWeek5Reliability(unittest.TestCase):
    def test_adaptive_motion_stgcn_single_and_dual_hand(self):
        for features in (63, 126):
            model = AdaptiveMotionSTGCNClassifier(features, 7, hidden_dim=24)
            logits = model(torch.randn(2, 30, features))
            self.assertEqual(logits.shape, (2, 7))
            logits.sum().backward()
            self.assertIsNotNone(model.adjacency_residual.grad)
            torch.testing.assert_close(model.normalized_adjacency().sum(1), torch.ones(features // 3))

    def test_temperature_scaling_and_ece(self):
        logits = torch.tensor([[6., 0.], [5., 0.], [0., 4.], [3., 0.]])
        targets = torch.tensor([0, 1, 1, 1])
        before = torch.softmax(logits, 1).numpy()
        scaler = TemperatureScaler().fit(logits, targets)
        after = scaler.probabilities(logits).numpy()
        self.assertGreater(scaler.temperature, 1.0)
        self.assertLess(expected_calibration_error(after, targets.numpy()), expected_calibration_error(before, targets.numpy()))

    def test_corruptions_preserve_shape_and_are_deterministic(self):
        x = torch.ones(2, 30, 63)
        a = corrupt_keypoints(x, "gaussian_noise", 0.05)
        b = corrupt_keypoints(x, "gaussian_noise", 0.05)
        self.assertEqual(a.shape, x.shape)
        torch.testing.assert_close(a, b)
        dropped = corrupt_keypoints(x, "frame_dropout", 0.2)
        self.assertTrue(torch.all(dropped[:, ::5] == 0))


if __name__ == "__main__":
    unittest.main()
