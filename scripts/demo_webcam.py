"""
Interactive Real-Time Webcam ASL Keypoint Visualizer & Recognition Demo.
Captures live frames from a webcam, extracts 21 3D hand landmarks via MediaPipe,
normalizes coordinates (wrist zero-centered, MCP scaled), buffers a 30-frame
dynamic window, and predicts the sign using the trained baseline classifier.
"""

import os
import sys
import time
import collections
import logging
import numpy as np
import cv2

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.keypoint_extractor import MediaPipeKeypointExtractor
from src.preprocessor import preprocess_sequence
from src.feature_extraction import extract_trajectory_features_single
from src.baselines import ASLBaselineBenchmarks
from src.data_ingestion import load_vocabulary

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


def run_webcam_demo(
    model_path: str = "models/random_forest_baseline.joblib",
    vocab_path: str = "config/vocabulary.json",
    cam_index: int = 0,
):
    print("=" * 60)
    print("      REAL-TIME ASL WEBCAM RECOGNITION DEMO")
    print("=" * 60)

    vocab = load_vocabulary(vocab_path)
    model = None
    if os.path.exists(model_path):
        model = ASLBaselineBenchmarks.load_model(model_path)
        logger.info(f"Loaded trained baseline model from {model_path}")
    else:
        logger.warning(f"No trained model found at {model_path}. Running in visualization-only mode.")

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

    print("\nPress 'q' in the camera window to exit.")

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
            # Occluded/missing frame: pad with zeros or previous frame
            if len(frame_buffer) > 0:
                frame_buffer.append(frame_buffer[-1])

        # If 30 frames are ready, run real-time inference
        if len(frame_buffer) == buffer_len and model is not None:
            seq_array = np.array(frame_buffer)  # (30, 63)
            # Spatial normalization & temporal resample
            norm_seq = preprocess_sequence(seq_array, target_frames=30, num_hands=1)
            # Trajectory kinematic features
            feat = extract_trajectory_features_single(norm_seq).reshape(1, -1)

            # Predict
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(feat)[0]
                top_idx = int(np.argmax(probs))
                confidence = float(probs[top_idx])
                current_prediction = vocab[top_idx] if top_idx < len(vocab) else f"Class {top_idx}"

                # Top 5
                top_5_ids = np.argsort(probs)[-5:][::-1]
                top_5_candidates = [(vocab[i] if i < len(vocab) else f"C{i}", float(probs[i])) for i in top_5_ids]
            else:
                pred_idx = int(model.predict(feat)[0])
                current_prediction = vocab[pred_idx] if pred_idx < len(vocab) else f"Class {pred_idx}"

        # Compute FPS
        t1 = time.perf_counter()
        fps_val = 1.0 / max(1e-5, (t1 - t0))
        fps_history.append(fps_val)
        avg_fps = sum(fps_history) / len(fps_history)

        # UI Overlay HUD
        cv2.rectangle(frame, (10, 10), (320, 160), (20, 20, 20), -1)
        cv2.rectangle(frame, (10, 10), (320, 160), (0, 255, 200), 2)

        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 2)
        cv2.putText(frame, f"Buffer: {len(frame_buffer)}/{buffer_len}", (150, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.putText(frame, "Predicted Sign:", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        cv2.putText(frame, current_prediction.upper(), (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        if confidence > 0.0:
            cv2.putText(frame, f"Conf: {confidence*100:.1f}%", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 1)

        # Top candidates sidebar
        if top_5_candidates:
            y_start = 145
            for name, score in top_5_candidates[:3]:
                cv2.putText(frame, f"{name}: {score*100:.1f}%", (20, y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 170, 170), 1)
                y_start += 18

        cv2.imshow("ASL Real-Time Recognition HUD", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    extractor.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_webcam_demo()
