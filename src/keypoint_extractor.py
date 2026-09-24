"""
MediaPipe Keypoint Extraction and Missing-Frame Interpolation Module.
Extracts 21 3D hand keypoints (x, y, z) per frame from video feeds and handles
missing or occluded frames using robust linear interpolation.
"""

import os
import logging
import urllib.request
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
DEFAULT_MODEL_PATH = "models/hand_landmarker.task"


class MediaPipeKeypointExtractor:
    """
    High-performance MediaPipe HandLandmarker wrapper for extracting 3D hand
    landmarks from video streams or webcam feeds.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        num_hands: int = 1,
        min_hand_detection_confidence: float = 0.5,
        min_hand_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self.model_path = model_path
        self.num_hands = num_hands
        self.min_hand_detection_confidence = min_hand_detection_confidence
        self.min_hand_presence_confidence = min_hand_presence_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self._ensure_model_exists()
        self._init_detector()

    def _ensure_model_exists(self):
        """Downloads the hand landmarker model bundle if missing."""
        if not os.path.exists(self.model_path):
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            logger.info(f"Downloading MediaPipe HandLandmarker model to {self.model_path}...")
            urllib.request.urlretrieve(DEFAULT_MODEL_URL, self.model_path)
            logger.info("Model download complete.")

    def _init_detector(self):
        """Initializes the MediaPipe HandLandmarker."""
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=self.num_hands,
            min_hand_detection_confidence=self.min_hand_detection_confidence,
            min_hand_presence_confidence=self.min_hand_presence_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def extract_from_frame(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Extracts keypoints from a single BGR image.
        Returns array of shape (21 * num_hands, 3), or None if no hand detected.
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        detection_result = self.detector.detect(mp_image)

        if not detection_result.hand_landmarks:
            return None

        # Build output array
        keypoints = []
        for h_idx in range(self.num_hands):
            if h_idx < len(detection_result.hand_landmarks):
                landmarks = detection_result.hand_landmarks[h_idx]
                hand_coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
            else:
                # Zero-pad if expected hand is missing
                hand_coords = np.zeros((21, 3), dtype=np.float32)
            keypoints.append(hand_coords)

        return np.vstack(keypoints)  # Shape: (21 * num_hands, 3)

    def extract_from_video(
        self,
        video_path: str,
        interpolate_missing: bool = True,
    ) -> Dict[str, Any]:
        """
        Reads a video file frame-by-frame, extracts keypoint trajectories,
        and optionally interpolates missing/occluded frames.

        Returns dictionary with:
            - 'trajectory': np.ndarray of shape (num_frames, 21 * num_hands * 3)
            - 'valid_mask': np.ndarray of shape (num_frames,) indicating valid detections
            - 'original_frame_count': int
            - 'detected_count': int
            - 'interpolated_count': int
            - 'fps': float
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        raw_keypoints = []
        valid_mask = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            kp = self.extract_from_frame(frame)
            if kp is not None:
                raw_keypoints.append(kp)
                valid_mask.append(True)
            else:
                raw_keypoints.append(None)
                valid_mask.append(False)

        cap.release()

        num_frames = len(raw_keypoints)
        valid_mask = np.array(valid_mask, dtype=bool)
        detected_count = int(np.sum(valid_mask))

        if num_frames == 0:
            raise ValueError(f"Video {video_path} contained 0 readable frames.")

        feature_dim = 21 * self.num_hands * 3

        if interpolate_missing:
            trajectory = self.interpolate_missing_frames(
                raw_keypoints=raw_keypoints,
                valid_mask=valid_mask,
                num_hands=self.num_hands,
            )
            interpolated_count = num_frames - detected_count
        else:
            # Replace None with zeros
            trajectory = np.zeros((num_frames, feature_dim), dtype=np.float32)
            for idx, kp in enumerate(raw_keypoints):
                if kp is not None:
                    trajectory[idx] = kp.reshape(-1)
            interpolated_count = 0

        return {
            "trajectory": trajectory,  # Shape: (num_frames, feature_dim)
            "valid_mask": valid_mask,
            "original_frame_count": num_frames,
            "detected_count": detected_count,
            "interpolated_count": interpolated_count,
            "fps": fps,
        }

    @staticmethod
    def interpolate_missing_frames(
        raw_keypoints: List[Optional[np.ndarray]],
        valid_mask: np.ndarray,
        num_hands: int = 1,
    ) -> np.ndarray:
        """
        Interpolates missing landmark detections across neighboring valid frames.
        Uses linear interpolation between detected anchors, with forward and
        backward fills at sequence boundaries.
        """
        num_frames = len(raw_keypoints)
        feature_dim = 21 * num_hands * 3
        output = np.zeros((num_frames, feature_dim), dtype=np.float32)

        valid_indices = np.where(valid_mask)[0]

        # Case 1: No frames detected (fallback to zeros)
        if len(valid_indices) == 0:
            logger.warning("No hand detected across all frames; returning zeros.")
            return output

        # Populate valid frames
        for idx in valid_indices:
            output[idx] = raw_keypoints[idx].reshape(-1)

        # Case 2: Only 1 frame detected (replicate everywhere)
        if len(valid_indices) == 1:
            return np.tile(output[valid_indices[0]], (num_frames, 1))

        # Case 3: Multiple frames detected -> Linear interpolation
        # Fill leading frames (before first detection)
        first_valid = valid_indices[0]
        if first_valid > 0:
            output[:first_valid] = output[first_valid]

        # Fill trailing frames (after last detection)
        last_valid = valid_indices[-1]
        if last_valid < num_frames - 1:
            output[last_valid + 1:] = output[last_valid]

        # Interpolate interior gaps
        for i in range(len(valid_indices) - 1):
            start_idx = valid_indices[i]
            end_idx = valid_indices[i + 1]
            gap = end_idx - start_idx

            if gap > 1:
                start_val = output[start_idx]
                end_val = output[end_idx]
                # Linear steps from start to end
                for step in range(1, gap):
                    alpha = step / float(gap)
                    interp_idx = start_idx + step
                    output[interp_idx] = (1.0 - alpha) * start_val + alpha * end_val

        return output

    def close(self):
        """Releases underlying MediaPipe detector resources."""
        if hasattr(self, "detector") and self.detector is not None:
            self.detector.close()
            self.detector = None
