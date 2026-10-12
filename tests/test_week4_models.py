"""Tests for Week 4 anatomical graph, Transformer, and ONNX export utilities."""
import os
import sys
import unittest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models import STGCNClassifier, TransformerClassifier, build_hand_adjacency


class TestWeek4Models(unittest.TestCase):
    def test_hand_adjacency_is_normalized_and_dual_hand_safe(self):
        adjacency = build_hand_adjacency(2)
        self.assertEqual(adjacency.shape, (42, 42))
        torch.testing.assert_close(adjacency.sum(dim=1), torch.ones(42))
        self.assertEqual(adjacency[:21, 21:].sum().item(), 0.0)

    def test_stgcn_single_and_dual_hand_shapes_and_gradients(self):
        for features in (63, 126):
            model = STGCNClassifier(input_dim=features, num_classes=7, hidden_dim=32)
            x = torch.randn(3, 30, features)
            logits = model(x)
            self.assertEqual(logits.shape, (3, 7))
            logits.sum().backward()
            self.assertIsNotNone(model.gcn1.projection.weight.grad)

    def test_transformer_mask_and_gradient(self):
        model = TransformerClassifier(input_dim=63, num_classes=7, hidden_dim=32, num_heads=4)
        x = torch.randn(3, 30, 63)
        mask = torch.ones(3, 30, dtype=torch.bool)
        mask[:, -4:] = False
        logits = model(x, mask)
        self.assertEqual(logits.shape, (3, 7))
        logits.sum().backward()
        self.assertIsNotNone(model.input_proj[0].weight.grad)


if __name__ == "__main__":
    unittest.main()
