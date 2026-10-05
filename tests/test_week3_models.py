"""
Unit Tests for Week 3 Deep Learning Sequence Models:
- BiGRU with Temporal Attention Pooling
- Dilated 1D Temporal Convolutional Network (1D-TCN)
- Temporal Attention Pooling Layer & Weight Normalization
- Backward pass and gradient flow verification
- Dual-hand (126-D) and Single-hand (63-D) tensor shapes
"""

import os
import sys
import unittest
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models import (
    TemporalAttentionPooling,
    BiGRUClassifier,
    TemporalBlock,
    TemporalConvNet,
    count_parameters,
)
from src.trainer import ASLModelTrainer


class TestWeek3Models(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        self.batch_size = 4
        self.seq_len = 30
        self.num_classes = 30

    def test_temporal_attention_pooling(self):
        """Attention weights sum to 1.0 along sequence dimension."""
        in_dim = 256
        att_layer = TemporalAttentionPooling(in_features=in_dim, attention_dim=64)

        x = torch.randn(self.batch_size, self.seq_len, in_dim)
        context, weights = att_layer(x)

        self.assertEqual(context.shape, (self.batch_size, in_dim))
        self.assertEqual(weights.shape, (self.batch_size, self.seq_len))

        # Check weights sum to 1.0 along time dimension
        weight_sums = weights.sum(dim=1)
        torch.testing.assert_close(weight_sums, torch.ones_like(weight_sums), atol=1e-5, rtol=1e-5)

    def test_temporal_attention_masking(self):
        """Padded frames receive near-zero attention weight when masked."""
        in_dim = 128
        att_layer = TemporalAttentionPooling(in_features=in_dim, attention_dim=32)

        x = torch.randn(2, 10, in_dim)
        mask = torch.ones(2, 10, dtype=torch.bool)
        mask[:, 7:] = False  # Mask frames 7..9

        context, weights = att_layer(x, mask=mask)
        self.assertEqual(context.shape, (2, in_dim))

        # Masked weights should be 0.0
        self.assertTrue(torch.all(weights[:, 7:] < 1e-4))

    def test_bigru_single_hand_forward(self):
        """BiGRU forward pass on single-hand (63-d) tensor."""
        input_dim = 63
        model = BiGRUClassifier(input_dim=input_dim, num_classes=self.num_classes, hidden_dim=64)

        x = torch.randn(self.batch_size, self.seq_len, input_dim)
        logits, att_weights = model(x, return_attention=True)

        self.assertEqual(logits.shape, (self.batch_size, self.num_classes))
        self.assertEqual(att_weights.shape, (self.batch_size, self.seq_len))
        self.assertFalse(torch.isnan(logits).any())

    def test_bigru_dual_hand_forward(self):
        """BiGRU forward pass on dual-hand (126-d) tensor."""
        input_dim = 126
        model = BiGRUClassifier(input_dim=input_dim, num_classes=self.num_classes, hidden_dim=64)

        x = torch.randn(self.batch_size, self.seq_len, input_dim)
        logits = model(x)

        self.assertEqual(logits.shape, (self.batch_size, self.num_classes))

    def test_bigru_gradient_flow(self):
        """BiGRU backward pass generates non-zero gradients across all layers."""
        model = BiGRUClassifier(input_dim=63, num_classes=self.num_classes, hidden_dim=64)
        x = torch.randn(self.batch_size, self.seq_len, 63)
        y = torch.randint(0, self.num_classes, (self.batch_size,))

        logits = model(x)
        loss = nn.CrossEntropyLoss()(logits, y)
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Gradient missing for {name}")
                self.assertFalse(torch.isnan(param.grad).any(), f"NaN gradient in {name}")

    def test_tcn_receptive_field_and_forward(self):
        """1D-TCN receptive field covers sequence and forward pass preserves output shape."""
        input_dim = 63
        channels = [32, 64, 64, 128]
        dilations = [1, 2, 4, 8]
        model = TemporalConvNet(
            input_dim=input_dim,
            num_classes=self.num_classes,
            num_channels=channels,
            dilations=dilations,
            kernel_size=3,
        )

        # Receptive field should be 1 + 2 * (3-1) * (1+2+4+8) = 61 frames
        self.assertEqual(model.receptive_field, 61)
        self.assertGreater(model.receptive_field, self.seq_len)

        x = torch.randn(self.batch_size, self.seq_len, input_dim)
        logits = model(x)

        self.assertEqual(logits.shape, (self.batch_size, self.num_classes))
        self.assertFalse(torch.isnan(logits).any())

    def test_tcn_dual_hand_forward(self):
        """1D-TCN operates on dual-hand (126-d) tensors."""
        input_dim = 126
        model = TemporalConvNet(input_dim=input_dim, num_classes=self.num_classes)

        x = torch.randn(self.batch_size, self.seq_len, input_dim)
        logits = model(x)

        self.assertEqual(logits.shape, (self.batch_size, self.num_classes))

    def test_tcn_gradient_flow(self):
        """1D-TCN backward pass propagates gradients through all dilated blocks."""
        model = TemporalConvNet(input_dim=63, num_classes=self.num_classes)
        x = torch.randn(self.batch_size, self.seq_len, 63)
        y = torch.randint(0, self.num_classes, (self.batch_size,))

        logits = model(x)
        loss = nn.CrossEntropyLoss()(logits, y)
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Gradient missing for {name}")

    def test_parameter_counting(self):
        """Parameter counting correctly totals module weights."""
        model = BiGRUClassifier(input_dim=63, num_classes=30, hidden_dim=64)
        params = count_parameters(model)
        self.assertGreater(params, 10000)
        self.assertLess(params, 2000000)

    def test_trainer_fit_mini_epoch(self):
        """Trainer completes an epoch and evaluation without error."""
        model = BiGRUClassifier(input_dim=63, num_classes=self.num_classes, hidden_dim=32)
        trainer = ASLModelTrainer(model=model, num_classes=self.num_classes, lr=1e-3)

        x = torch.randn(12, self.seq_len, 63)
        y = torch.tensor([i % self.num_classes for i in range(12)], dtype=torch.long)
        ds = TensorDataset(x, y)
        loader = DataLoader(ds, batch_size=4, shuffle=False)

        metrics = trainer.train_epoch(loader)
        self.assertIn("loss", metrics)
        self.assertIn("top_1_accuracy", metrics)

        eval_res = trainer.evaluate(loader)
        self.assertIn("top_1_accuracy", eval_res)
        self.assertIn("top_5_accuracy", eval_res)
        self.assertIn("macro_f1", eval_res)


if __name__ == "__main__":
    unittest.main()
