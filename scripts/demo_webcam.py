"""
Interactive Real-Time Webcam ASL Keypoint Visualizer & Dynamic Recognition Demo.
Captures live frames from a monocular webcam, tracks 21 3D hand landmarks via MediaPipe,
normalizes coordinates (wrist-centered, MCP-scaled), buffers a 30-frame temporal window,
and classifies dynamic gestures in real time using Week 3 Deep Models (BiGRU / 1D-TCN)
or Week 2 ML Baselines (Random Forest / SVC).
"""

import os
import sys
import time
import argparse
import collections
import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import cv2
import torch

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.keypoint_extractor import MediaPipeKeypointExtractor
from src.preprocessor import preprocess_sequence
from src.feature_extraction import extract_trajectory_features_single
from src.baselines import ASLBaselineBenchmarks
from src.data_ingestion import load_vocabulary
from src.models import BiGRUClassifier, TemporalConvNet

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Landmark connection lines for skeleton rendering
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (0, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17),             # Palm MCP base
]


def draw_skeleton(frame: np.ndarray, landmarks_norm: np.ndarray) -> np.ndarray:
    """
    Renders 21 hand landmarks and connection lines over the video frame.
    landmarks_norm: (21, 3) in [0, 1] normalized screen space.
    """
    h, w, _ = frame.shape
    px_coords = []
    for lm in landmarks_norm:
        cx, cy = int(lm[0] * w), int(lm[1] * h)
        px_coords.append((cx, cy))

    # Draw connection lines
    for p1, p2 in HAND_CONNECTIONS:
        cv2.line(frame, px_coords[p1], px_coords[p2], (0, 230, 255), 2)

    # Draw landmark joints
    for idx, (cx, cy) in enumerate(px_coords):
        color = (0, 255, 100) if idx in [0, 9] else (255, 100, 0)
        radius = 5 if idx in [0, 9] else 3
        cv2.circle(frame, (cx, cy), radius, color, -1)

    return frame


def load_inference_model(
    model_type: str,
    vocab: List[str],
    checkpoint_path: Optional[str] = None,
    device: Optional[torch.device] = None,
):
    """Loads either a PyTorch deep sequence model or a Scikit-Learn baseline."""
    dev = device or torch.device("cpu")
    num_classes = len(vocab)

    if model_type == "bigru":
        path = checkpoint_path or "models/bigru_best.pth"
        if not os.path.exists(path):
            logger.warning(f"BiGRU checkpoint {path} not found.")
            return None, "pytorch"
        model = BiGRUClassifier(input_dim=63, num_classes=num_classes, hidden_dim=128)
        ckpt = torch.load(path, map_location=dev)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict)
        model.to(dev)
        model.eval()
        logger.info(f"Loaded BiGRU model from {path}")
        return model, "pytorch"

    elif model_type == "tcn":
        path = checkpoint_path or "models/tcn_best.pth"
        if not os.path.exists(path):
            logger.warning(f"1D-TCN checkpoint {path} not found.")
            return None, "pytorch"
        model = TemporalConvNet(input_dim=63, num_classes=num_classes)
        ckpt = torch.load(path, map_location=dev)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict)
        model.to(dev)
        model.eval()
        logger.info(f"Loaded 1D-TCN model from {path}")
        return model, "pytorch"

    elif model_type == "random_forest":
        path = checkpoint_path or "models/random_forest_baseline.joblib"
        if not os.path.exists(path):
            logger.warning(f"Random Forest model {path} not found.")
            return None, "sklearn"
        model = ASLBaselineBenchmarks.load_model(path)
        logger.info(f"Loaded Random Forest baseline from {path}")
        return model, "sklearn"

    elif model_type == "svc":
        path = checkpoint_path or "models/svc_baseline.joblib"
        if not os.path.exists(path):
            logger.warning(f"SVC model {path} not found.")
            return None, "sklearn"
        model = ASLBaselineBenchmarks.load_model(path)
        logger.info(f"Loaded SVC baseline from {path}")
        return model, "sklearn"

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def run_webcam_demo(
    model_type: str = "bigru",
    checkpoint_path: Optional[str] = None,
    vocab_path: str = "config/vocabulary.json",
    cam_index: int = 0,
):
    print("=" * 65)
    print(f"      REAL-TIME ASL RECOGNITION DEMO [{model_type.upper()}]")
    print("=" * 65)

    vocab = load_vocabulary(vocab_path)
    device = torch.device("cpu")
    model, backend = load_inference_model(model_type, vocab, checkpoint_path, device)

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f"Error: Could not open camera device {cam_index}. Exiting demo.")
        return

    extractor = MediaPipeKeypointExtractor(num_hands=1)
    buffer_len = 30
    frame_buffer = collections.deque(maxlen=buffer_len)

    fps_history = collections.deque(maxlen=20)
    current_prediction = "Waiting for gesture..."
    top_5_candidates = []
    confidence = 0.0

    print("\nControls:")
    print("  - Press 'q' to exit")
    print("  - Press 'm' to cycle models (BiGRU -> TCN -> RF -> SVC)")

    models_cycle = ["bigru", "tcn", "random_forest", "svc"]
    current_model_idx = models_cycle.index(model_type) if model_type in models_cycle else 0

    while True:
        t0 = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break

        # Flip horizontally for natural mirror feel
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Extract 21 landmarks
        kp = extractor.extract_from_frame(frame)

        if kp is not None:
            # Draw skeleton
            frame = draw_skeleton(frame, kp)
            frame_buffer.append(kp.reshape(-1))
        else:
            # Occluded/missing frame: pad with previous frame
            if len(frame_buffer) > 0:
                frame_buffer.append(frame_buffer[-1])

        # If 30 frames are ready, run real-time inference
        if len(frame_buffer) == buffer_len and model is not None:
            seq_array = np.array(frame_buffer, dtype=np.float32)  # (30, 63)
            # Spatial normalization & temporal resample
            norm_seq = preprocess_sequence(seq_array, target_frames=30, num_hands=1)

            if backend == "pytorch":
                tensor_in = torch.from_numpy(norm_seq).float().unsqueeze(0).to(device)  # (1, 30, 63)
                with torch.no_grad():
                    logits = model(tensor_in)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            else:
                # Scikit-Learn (extract 714 kinematic trajectory features)
                feat = extract_trajectory_features_single(norm_seq).reshape(1, -1)
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(feat)[0]
                else:
                    pred_idx = int(model.predict(feat)[0])
                    probs = np.zeros(len(vocab))
                    probs[pred_idx] = 1.0

            top_idx = int(np.argmax(probs))
            confidence = float(probs[top_idx])
            current_prediction = vocab[top_idx] if top_idx < len(vocab) else f"Class {top_idx}"

            # Top 5 candidates
            top_5_ids = np.argsort(probs)[-5:][::-1]
            top_5_candidates = [(vocab[i] if i < len(vocab) else f"C{i}", float(probs[i])) for i in top_5_ids]

        # Compute FPS
        t1 = time.perf_counter()
        fps_val = 1.0 / max(1e-5, (t1 - t0))
        fps_history.append(fps_val)
        avg_fps = sum(fps_history) / len(fps_history)

        # UI Overlay HUD
        cv2.rectangle(frame, (10, 10), (330, 190), (20, 20, 20), -1)
        cv2.rectangle(frame, (10, 10), (330, 190), (0, 255, 200), 2)

        cv2.putText(frame, f"Model: {model_type.upper()}", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (180, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 200), 2)
        cv2.putText(frame, f"Buffer: {len(frame_buffer)}/{buffer_len}", (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        cv2.putText(frame, "Predicted Sign:", (20, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        cv2.putText(frame, current_prediction.upper(), (20, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)

        if confidence > 0.0:
            cv2.putText(frame, f"Conf: {confidence*100:.1f}%", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 1)

        # Top 3 candidates display
        if top_5_candidates:
            y_start = 152
            for name, score in top_5_candidates[:3]:
                cv2.putText(frame, f"{name}: {score*100:.1f}%", (20, y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 170, 170), 1)
                y_start += 16

        cv2.imshow("ASL Real-Time Recognition HUD (Week 3)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'):
            current_model_idx = (current_model_idx + 1) % len(models_cycle)
            model_type = models_cycle[current_model_idx]
            model, backend = load_inference_model(model_type, vocab, None, device)
            print(f"Switched model to: {model_type.upper()}")

    cap.release()
    extractor.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-Time Webcam ASL Recognition HUD")
    parser.add_argument("--model_type", type=str, default="bigru", choices=["bigru", "tcn", "random_forest", "svc"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--cam", type=int, default=0)
    args = parser.parse_args()

    run_webcam_demo(model_type=args.model_type, checkpoint_path=args.checkpoint, cam_index=args.cam)
