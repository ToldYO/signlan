"""
PyTorch Dataset and DataLoader Module for ASL Keypoint Sequences.
Serves normalized, interpolated tensor batches of shape (batch_size, 30, 63)
with optional kinematic data augmentations.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Dict, Optional, Tuple, Union, Any
from src.preprocessor import preprocess_sequence


class ASLKeypointAugmentation:
    """
    Kinematic coordinate data augmentations for ASL keypoint trajectories.
    Enhances generalization against real-world camera angles, user distances,
    and detector jitter.
    """

    def __init__(
        self,
        jitter_std: float = 0.01,
        scale_range: Tuple[float, float] = (0.92, 1.08),
        rotation_deg: float = 12.0,
        enable_jitter: bool = True,
        enable_scale: bool = True,
        enable_rotation: bool = True,
    ):
        self.jitter_std = jitter_std
        self.scale_range = scale_range
        self.rotation_deg = rotation_deg
        self.enable_jitter = enable_jitter
        self.enable_scale = enable_scale
        self.enable_rotation = enable_rotation

    def __call__(self, trajectory: np.ndarray) -> np.ndarray:
        """
        trajectory: (T, 63) or (T, 126)
        """
        T, D = trajectory.shape
        num_pts = D // 3
        coords = trajectory.reshape(T, num_pts, 3).copy()

        # 1. Coordinate Jitter (Sensor noise simulation)
        if self.enable_jitter and self.jitter_std > 0:
            noise = np.random.normal(0, self.jitter_std, size=coords.shape).astype(np.float32)
            coords += noise

        # 2. Random Scale (User distance variation)
        if self.enable_scale and self.scale_range:
            scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
            coords *= scale

        # 3. Random In-Plane Rotation (Camera tilt / wrist angle)
        if self.enable_rotation and self.rotation_deg > 0:
            angle_rad = np.radians(np.random.uniform(-self.rotation_deg, self.rotation_deg))
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            rot_matrix = np.array([
                [cos_a, -sin_a, 0],
                [sin_a,  cos_a, 0],
                [0,      0,     1],
            ], dtype=np.float32)

            # Apply rotation per landmark per frame
            coords = np.matmul(coords, rot_matrix.T)

        return coords.reshape(T, D).astype(np.float32)


class ASLKeypointDataset(Dataset):
    """
    PyTorch Dataset providing normalized (30, 63) or (30, 126) ASL keypoint trajectories.
    """

    def __init__(
        self,
        trajectories: List[np.ndarray],
        labels: List[int],
        target_frames: int = 30,
        num_hands: int = 1,
        is_preprocessed: bool = False,
        augment: bool = False,
        augmentation_cfg: Optional[ASLKeypointAugmentation] = None,
    ):
        """
        Args:
            trajectories: List of raw or preprocessed trajectory arrays.
            labels: List of integer class labels.
            target_frames: Uniform temporal window (default 30).
            num_hands: 1 (63 features) or 2 (126 features).
            is_preprocessed: If True, skips spatial norm and resampling.
            augment: If True, applies random geometric augmentations.
        """
        assert len(trajectories) == len(labels), "Trajectories and labels must match in length"
        self.labels = np.array(labels, dtype=np.int64)
        self.target_frames = target_frames
        self.num_hands = num_hands
        self.augment = augment
        self.augmenter = augmentation_cfg or ASLKeypointAugmentation()

        # Preprocess upfront if not already uniform
        if is_preprocessed:
            self.data = [np.array(t, dtype=np.float32) for t in trajectories]
        else:
            self.data = [
                preprocess_sequence(
                    t, target_frames=target_frames, num_hands=num_hands
                ).astype(np.float32)
                for t in trajectories
            ]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.data[idx]

        if self.augment:
            sample = self.augmenter(sample)

        tensor_sample = torch.from_numpy(sample).float()  # (30, 63)
        tensor_label = torch.tensor(self.labels[idx], dtype=torch.long)

        return tensor_sample, tensor_label


def create_dataloaders(
    train_data: Tuple[List[np.ndarray], List[int]],
    val_data: Optional[Tuple[List[np.ndarray], List[int]]] = None,
    test_data: Optional[Tuple[List[np.ndarray], List[int]]] = None,
    batch_size: int = 16,
    num_hands: int = 1,
    target_frames: int = 30,
    num_workers: int = 0,
) -> Dict[str, DataLoader]:
    """
    Builds PyTorch DataLoaders for train, validation, and test splits.
    Batches have shape (batch_size, 30, 21 * num_hands * 3).
    """
    train_traj, train_labels = train_data
    train_dataset = ASLKeypointDataset(
        trajectories=train_traj,
        labels=train_labels,
        target_frames=target_frames,
        num_hands=num_hands,
        augment=True,
    )

    loaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    }

    if val_data is not None:
        val_traj, val_labels = val_data
        val_dataset = ASLKeypointDataset(
            trajectories=val_traj,
            labels=val_labels,
            target_frames=target_frames,
            num_hands=num_hands,
            augment=False,
        )
        loaders["val"] = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    if test_data is not None:
        test_traj, test_labels = test_data
        test_dataset = ASLKeypointDataset(
            trajectories=test_traj,
            labels=test_labels,
            target_frames=target_frames,
            num_hands=num_hands,
            augment=False,
        )
        loaders["test"] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    return loaders
