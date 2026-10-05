"""
Deep Temporal Sequence Architectures for Dynamic ASL Recognition.
Implements:
1. Bidirectional GRU (BiGRU) with Temporal Attention Pooling.
2. 1D Temporal Convolutional Network (1D-TCN) with Dilated Residual Blocks.
3. Temporal Attention Pooling mechanism for dynamic sequence aggregation.
4. Dual-hand (126-D) and single-hand (63-D) tensor compatibility.
"""

import math
from typing import Dict, List, Optional, Tuple, Union, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttentionPooling(nn.Module):
    """
    Self-attentive temporal pooling layer.
    Computes normalized attention scores over temporal sequence states:
        alpha_t = Softmax(w^T * tanh(W * h_t + b))
        c = sum_{t=1}^T (alpha_t * h_t)
    """

    def __init__(self, in_features: int, attention_dim: int = 64):
        super().__init__()
        self.in_features = in_features
        self.attention_dim = attention_dim

        self.projection = nn.Sequential(
            nn.Linear(in_features, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1, bias=False),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Hidden representations of shape (batch_size, seq_len, in_features).
            mask: Optional boolean mask of shape (batch_size, seq_len) where True = valid, False = pad.

        Returns:
            context: Pooled representation of shape (batch_size, in_features).
            weights: Normalized attention weights of shape (batch_size, seq_len).
        """
        # (batch_size, seq_len, 1)
        scores = self.projection(x)

        if mask is not None:
            mask_expanded = mask.unsqueeze(-1)  # (batch_size, seq_len, 1)
            scores = scores.masked_fill(~mask_expanded, -1e9)

        weights = F.softmax(scores, dim=1)  # (batch_size, seq_len, 1)
        context = torch.sum(weights * x, dim=1)  # (batch_size, in_features)

        return context, weights.squeeze(-1)


class BiGRUClassifier(nn.Module):
    """
    Bidirectional Gated Recurrent Unit (BiGRU) with Temporal Attention Pooling.
    
    Architecture:
    1. Input Linear Projection + LayerNorm + Dropout
    2. Multi-layer Bidirectional GRU (default: 2 layers, hidden_dim=128)
    3. Temporal Attention Pooling over hidden sequence states (2 * hidden_dim)
    4. Residual feature fusion (pooled attention context + temporal mean/max pooling)
    5. Fully connected classification head with Dropout & BatchNorm/LayerNorm
    """

    def __init__(
        self,
        input_dim: int = 63,
        num_classes: int = 30,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        attention_dim: int = 64,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # Coordinate embedding projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Recurrent backbone
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        gru_out_dim = hidden_dim * self.num_directions  # 256 for bidirectional

        # Temporal attention mechanism
        self.attention = TemporalAttentionPooling(in_features=gru_out_dim, attention_dim=attention_dim)

        # Classification Head (concatenating attention context with mean-pooled context)
        combined_dim = gru_out_dim * 2  # attention pooled + temporal average pooled
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: Input coordinate tensor of shape (batch_size, seq_len, input_dim).
            mask: Optional validity mask.
            return_attention: If True, also returns attention weights.

        Returns:
            logits: Class logits of shape (batch_size, num_classes).
            attention_weights: (Optional) Tensor of shape (batch_size, seq_len).
        """
        # Embed coordinates
        embedded = self.input_proj(x)  # (batch_size, seq_len, hidden_dim)

        # GRU forward
        gru_out, _ = self.gru(embedded)  # (batch_size, seq_len, 2 * hidden_dim)

        # Temporal pooling
        att_context, att_weights = self.attention(gru_out, mask=mask)  # (batch_size, 2 * hidden_dim)
        mean_context = torch.mean(gru_out, dim=1)  # (batch_size, 2 * hidden_dim)

        # Multi-representation context
        fused = torch.cat([att_context, mean_context], dim=-1)  # (batch_size, 4 * hidden_dim)

        logits = self.classifier(fused)  # (batch_size, num_classes)

        if return_attention:
            return logits, att_weights
        return logits


class Chomp1d(nn.Module):
    """
    Strips trailing padding to maintain causality in temporal convolutions if needed.
    """

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size <= 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """
    Dilated 1D Residual Convolutional Block for Temporal Convolutional Networks (TCN).
    Consists of two dilated convolution layers with batch normalization, GELU activation,
    spatial dropout, and residual identity/projection skip connections.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.act1 = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.act2 = nn.GELU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.bn1,
            self.act1,
            self.dropout1,
            self.conv2,
            self.bn2,
            self.act2,
            self.dropout2,
        )

        # 1x1 Conv shortcut if channel dimensions change
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )
        self.final_act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, in_channels, seq_len)
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.final_act(out + res)


class TemporalConvNet(nn.Module):
    """
    1D Temporal Convolutional Network (1D-TCN) for dynamic ASL coordinate sequence modeling.
    
    Features:
    - Multi-scale dilated convolutions with exponential dilation factors [1, 2, 4, 8].
    - Wide temporal receptive field: RF = 1 + sum_i 2 * (K - 1) * d_i (covers >= 61 frames).
    - Residual skip connections preserving low-level joint position and high-level motion dynamics.
    - Highly parallelizable non-recurrent inference for ultra-low latency.
    """

    def __init__(
        self,
        input_dim: int = 63,
        num_classes: int = 30,
        num_channels: Optional[List[int]] = None,
        kernel_size: int = 3,
        dropout: float = 0.25,
        dilations: Optional[List[int]] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        if num_channels is None:
            num_channels = [64, 128, 128, 256]
        if dilations is None:
            dilations = [1, 2, 4, 8]

        assert len(num_channels) == len(dilations), "Channels and dilations must have equal length"

        layers = []
        in_c = input_dim
        receptive_field = 1

        for i, (out_c, d) in enumerate(zip(num_channels, dilations)):
            # Symmetric padding preserves exact temporal sequence length T=30
            padding = (kernel_size - 1) * d // 2
            layers.append(
                TemporalBlock(
                    in_channels=in_c,
                    out_channels=out_c,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=d,
                    padding=padding,
                    dropout=dropout,
                )
            )
            receptive_field += 2 * (kernel_size - 1) * d
            in_c = out_c

        self.tcn = nn.Sequential(*layers)
        self.receptive_field = receptive_field

        # Global temporal pooling: Concatenate Adaptive Max-Pool and Adaptive Avg-Pool
        top_channels = num_channels[-1]
        self.classifier = nn.Sequential(
            nn.Linear(top_channels * 2, top_channels),
            nn.BatchNorm1d(top_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(top_channels, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input coordinate tensor of shape (batch_size, seq_len, input_dim).

        Returns:
            logits: Output classification logits of shape (batch_size, num_classes).
        """
        # Permute to (batch_size, input_dim, seq_len) for 1D convolution
        x_conv = x.transpose(1, 2)

        features = self.tcn(x_conv)  # (batch_size, top_channels, seq_len)

        # Multi-scale global temporal pooling
        avg_pool = torch.mean(features, dim=2)               # (batch_size, top_channels)
        max_pool, _ = torch.max(features, dim=2)             # (batch_size, top_channels)
        pooled = torch.cat([avg_pool, max_pool], dim=1)      # (batch_size, top_channels * 2)

        logits = self.classifier(pooled)                     # (batch_size, num_classes)
        return logits


def count_parameters(model: nn.Module) -> int:
    """Computes total trainable parameters in a PyTorch module."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
