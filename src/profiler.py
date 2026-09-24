"""
Profiling and Throughput Benchmarking Module.
Profiles MediaPipe perception latency, throughput (FPS), CPU utilization,
and end-to-end preprocessing overhead to confirm real-time operational feasibility.
"""

import os
import time
import logging
from typing import Dict, List, Optional, Any
import numpy as np
import cv2
import psutil

from src.keypoint_extractor import MediaPipeKeypointExtractor
from src.preprocessor import preprocess_sequence

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def profile_keypoint_extractor(
    extractor: MediaPipeKeypointExtractor,
    video_path: Optional[str] = None,
    num_synthetic_frames: int = 150,
    warmup_frames: int = 15,
) -> Dict[str, Any]:
    """
    Measures per-frame inference latency, throughput (FPS), and CPU load.
    Can run on an existing video file or synthetic test frames.
    """
    logger.info("Starting throughput profiling...")

    # Load frames
    frames = []
    if video_path and os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        logger.info(f"Loaded {len(frames)} frames from {video_path}")
    else:
        logger.info(f"Generating {num_synthetic_frames} synthetic test frames (640x480)...")
        for _ in range(num_synthetic_frames):
            dummy = np.random.randint(40, 200, (480, 640, 3), dtype=np.uint8)
            frames.append(dummy)

    total_frames = len(frames)
    if total_frames < warmup_frames + 5:
        frames = frames * ((warmup_frames + 10) // total_frames + 1)
        total_frames = len(frames)

    # Warmup phase
    for i in range(warmup_frames):
        extractor.extract_from_frame(frames[i])

    # Benchmarking phase
    latencies = []
    cpu_measurements = []
    process = psutil.Process()

    eval_frames = frames[warmup_frames:]
    start_total_time = time.perf_counter()

    for frame in eval_frames:
        t0 = time.perf_counter()
        _ = extractor.extract_from_frame(frame)
        t1 = time.perf_counter()

        latencies.append((t1 - t0) * 1000.0)  # milliseconds
        cpu_measurements.append(process.cpu_percent(interval=None))

    total_time_sec = time.perf_counter() - start_total_time
    num_eval = len(eval_frames)

    latencies = np.array(latencies)
    avg_fps = num_eval / total_time_sec

    results = {
        "num_frames_evaluated": num_eval,
        "total_time_seconds": float(total_time_sec),
        "average_fps": float(avg_fps),
        "mean_latency_ms": float(np.mean(latencies)),
        "median_latency_ms": float(np.median(latencies)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "p99_latency_ms": float(np.percentile(latencies, 99)),
        "min_latency_ms": float(np.min(latencies)),
        "max_latency_ms": float(np.max(latencies)),
        "mean_cpu_percent": float(np.mean(cpu_measurements)),
        "realtime_feasible": bool(avg_fps >= 30.0),
    }

    logger.info(
        f"Profiling Results: Average FPS: {results['average_fps']:.1f} | "
        f"Mean Latency: {results['mean_latency_ms']:.2f} ms | "
        f"P95: {results['p95_latency_ms']:.2f} ms | "
        f"Real-time (>30 FPS): {results['realtime_feasible']}"
    )

    return results


def profile_preprocessing_pipeline(
    raw_lengths: List[int] = [15, 30, 45, 60, 75],
    target_frames: int = 30,
    iterations: int = 100,
) -> Dict[str, Any]:
    """
    Profiles the mathematical preprocessing overhead (wrist zero-centering,
    middle MCP joint Euclidean distance scaling, and 1D temporal spline interpolation).
    """
    logger.info("Profiling spatial normalization and temporal resampling overhead...")
    benchmarks = {}

    for length in raw_lengths:
        latencies = []
        dummy_seq = np.random.randn(length, 63).astype(np.float32)

        for _ in range(iterations):
            t0 = time.perf_counter()
            _ = preprocess_sequence(dummy_seq, target_frames=target_frames)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

        benchmarks[f"len_{length}_to_{target_frames}"] = {
            "mean_ms": float(np.mean(latencies)),
            "std_ms": float(np.std(latencies)),
            "throughput_seq_per_sec": float(1000.0 / np.mean(latencies)),
        }

    return benchmarks
