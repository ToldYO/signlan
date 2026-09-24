"""
Spatial Normalization and Temporal Resampling Preprocessor.
Implements wrist-centered, MCP-scaled spatial normalization to achieve scale
and position invariance, alongside temporal interpolation to uniform T=30 frames.
"""

import numpy as np
from scipy.interpolate import interp1d
from typing import Optional, Union, Tuple
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Keypoint indices according to MediaPipe Hand Landmark topology
WRIST_IDX = 0
MIDDLE_FINGER_MCP_IDX = 9


def normalize_spatial_coordinates(
    trajectory: np.ndarray,
    num_hands: int = 1,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Performs spatial normalization on hand landmark coordinates:
    1. Zero-centers coordinates at the wrist (Landmark 0) for translation invariance:
       P'_i = P_i - P_wrist
    2. Scales coordinates by the Euclidean distance between the wrist and the
       middle-finger MCP joint (Landmark 9) for scale invariance:
       P''_i = P'_i / (||P'_mcp - P'_wrist||_2 + eps)

    Args:
        trajectory: Array of shape (T, 21 * num_hands * 3) or (T, 21 * num_hands, 3).
        num_hands: Number of hands (1 or 2).
        eps: Small epsilon to prevent division by zero.

    Returns:
        Normalized trajectory array of identical shape.
    """
    orig_shape = trajectory.shape
    T = orig_shape[0]

    # Reshape to (T, num_hands, 21, 3)
    reshaped = trajectory.reshape(T, num_hands, 21, 3).copy()
    normalized = np.zeros_like(reshaped)

    for t in range(T):
        for h in range(num_hands):
            hand_pts = reshaped[t, h]  # Shape: (21, 3)

            # Check if hand is detected (not all zeros)
            if np.all(np.abs(hand_pts) < eps):
                normalized[t, h] = 0.0
                continue

            wrist = hand_pts[WRIST_IDX]           # (3,)
            mcp = hand_pts[MIDDLE_FINGER_MCP_IDX] # (3,)

            # Zero-center relative to wrist
            centered = hand_pts - wrist

            # Scale by distance between wrist and middle MCP joint
            scale_dist = np.linalg.norm(mcp - wrist)
            if scale_dist < eps:
                scale_dist = 1.0

            normalized[t, h] = centered / scale_dist

    return normalized.reshape(orig_shape)


def resample_temporal_trajectory(
    trajectory: np.ndarray,
    target_frames: int = 30,
    kind: str = "linear",
) -> np.ndarray:
    """
    Resamples a variable-length trajectory of shape (L, D) to a uniform
    fixed temporal window of shape (target_frames, D).

    Args:
        trajectory: Array of shape (L, D), where L is current frame count and D is feature dimension.
        target_frames: Fixed temporal target length (default: 30 frames / 1.0s at 30 FPS).
        kind: 'linear' or 'cubic' spline interpolation.

    Returns:
        Resampled array of shape (target_frames, D).
    """
    L, D = trajectory.shape

    # Trivial case: already target length
    if L == target_frames:
        return trajectory.copy()

    # Extreme edge case: single frame (repeat across time)
    if L == 1:
        return np.tile(trajectory, (target_frames, 1))

    # Time axes
    orig_time = np.linspace(0.0, 1.0, num=L, endpoint=True)
    target_time = np.linspace(0.0, 1.0, num=target_frames, endpoint=True)

    # Use linear if length is too short for cubic spline
    interp_kind = kind if (kind == "cubic" and L >= 4) else "linear"

    f = interp1d(orig_time, trajectory, axis=0, kind=interp_kind, assume_sorted=True)
    resampled = f(target_time).astype(np.float32)

    return resampled


def preprocess_sequence(
    raw_trajectory: np.ndarray,
    target_frames: int = 30,
    num_hands: int = 1,
    interp_kind: str = "linear",
) -> np.ndarray:
    """
    Complete preprocessing pipeline:
    1. Spatial normalization (wrist zero-centering + middle MCP scale division)
    2. Temporal resampling to fixed window T (default: 30)

    Args:
        raw_trajectory: Array of shape (L, 21 * num_hands * 3).
        target_frames: Target sequence length T.
        num_hands: Hand count (1 or 2).
        interp_kind: Resampling interpolation method.

    Returns:
        Preprocessed trajectory of shape (target_frames, 21 * num_hands * 3).
    """
    # 1. Spatial normalization
    spatially_normed = normalize_spatial_coordinates(raw_trajectory, num_hands=num_hands)

    # 2. Temporal resampling
    uniform_trajectory = resample_temporal_trajectory(
        spatially_normed, target_frames=target_frames, kind=interp_kind
    )

    return uniform_trajectory
