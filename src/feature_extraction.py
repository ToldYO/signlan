"""
Trajectory Feature Engineering Module.
Extracts aggregated spatial and kinematic statistics (mean, variance, velocity,
acceleration, displacement, path length) from 30-frame coordinate sequences
to train classical machine learning baselines (Random Forest, SVM/SVC).
"""

import numpy as np
from typing import List, Dict, Any, Union


def extract_trajectory_features_single(
    trajectory: np.ndarray,
    dt: float = 1.0 / 30.0,
) -> np.ndarray:
    """
    Extracts rich spatial and kinematic summary statistics from a single
    preprocessed trajectory of shape (T, D) where T=30 and D=63 (or 126).

    Extracted features include:
    1. Spatial distribution: Mean, Standard Deviation, Min, Max, Range (5 * D)
    2. Net spatial displacement: End pos - Start pos (1 * D)
    3. First-order dynamics (Velocity):
       - Mean velocity, Max velocity, Std of velocity (3 * D)
    4. Second-order dynamics (Acceleration):
       - Mean acceleration, Max acceleration (2 * D)
    5. Per-landmark 3D Euclidean path lengths (D // 3)

    Returns:
        1D feature vector of shape (K,)
    """
    T, D = trajectory.shape
    num_pts = D // 3
    coords_3d = trajectory.reshape(T, num_pts, 3)

    # 1. Spatial summary statistics
    mean_pos = np.mean(trajectory, axis=0)      # (D,)
    std_pos = np.std(trajectory, axis=0)        # (D,)
    min_pos = np.min(trajectory, axis=0)        # (D,)
    max_pos = np.max(trajectory, axis=0)        # (D,)
    range_pos = max_pos - min_pos               # (D,)

    # 2. Net displacement
    displacement = trajectory[-1] - trajectory[0] # (D,)

    # 3. Velocity (first temporal difference)
    velocity = np.diff(trajectory, axis=0) / dt   # (T-1, D)
    mean_vel = np.mean(velocity, axis=0)          # (D,)
    std_vel = np.std(velocity, axis=0)            # (D,)
    max_vel = np.max(np.abs(velocity), axis=0)    # (D,)

    # 4. Acceleration (second temporal difference)
    acceleration = np.diff(velocity, axis=0) / dt # (T-2, D)
    mean_acc = np.mean(acceleration, axis=0)      # (D,)
    max_acc = np.max(np.abs(acceleration), axis=0)# (D,)

    # 5. Cumulative 3D path length per landmark
    diffs_3d = np.diff(coords_3d, axis=0)         # (T-1, num_pts, 3)
    step_lengths = np.linalg.norm(diffs_3d, axis=2) # (T-1, num_pts)
    cum_path_lengths = np.sum(step_lengths, axis=0) # (num_pts,)

    # Combine all feature vectors
    feature_vector = np.concatenate([
        mean_pos,
        std_pos,
        min_pos,
        max_pos,
        range_pos,
        displacement,
        mean_vel,
        std_vel,
        max_vel,
        mean_acc,
        max_acc,
        cum_path_lengths,
    ]).astype(np.float32)

    return feature_vector


def extract_trajectory_features_batch(
    trajectories: Union[List[np.ndarray], np.ndarray],
    dt: float = 1.0 / 30.0,
) -> np.ndarray:
    """
    Extracts trajectory features for a batch of sequences.
    Input: List of (T, D) arrays or (N, T, D) array.
    Output: (N, K) feature matrix.
    """
    features = [extract_trajectory_features_single(t, dt=dt) for t in trajectories]
    return np.vstack(features)
