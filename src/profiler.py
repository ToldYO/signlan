"""
Profiling and Throughput Benchmarking Module.
Profiles MediaPipe perception latency, throughput (FPS), CPU utilization,
and end-to-end preprocessing overhead to confirm real-time operational feasibility.
"""

import os
import time
import logging
from typing import Dict, List, Optional, Any, Union
import numpy as np
import cv2
import psutil
import torch

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


def profile_model_latency(
    model: Any,
    input_sample: Union[np.ndarray, torch.Tensor],
    is_pytorch: bool = True,
    iterations: int = 200,
    warmup: int = 20,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Benchmarks model inference latency percentiles (p50, p95, p99), mean latency,
    and throughput on a single input instance (simulating real-time webcam frame processing).

    Args:
        model: PyTorch nn.Module or Scikit-Learn Estimator.
        input_sample: Input tensor (1, T, D) or feature vector (1, N).
        is_pytorch: If True, uses torch eval mode and torch.no_grad().
        iterations: Number of evaluation benchmark runs.
        warmup: Number of initial warmup runs.
        device: Torch device (defaults to CPU for on-device latency evaluation).

    Returns:
        Dictionary of latency statistics in milliseconds and throughput in inferences/sec.
    """
    if is_pytorch:
        dev = device or torch.device("cpu")
        model = model.to(dev)
        model.eval()

        if isinstance(input_sample, np.ndarray):
            input_tensor = torch.from_numpy(input_sample).float().to(dev)
        else:
            input_tensor = input_sample.to(dev)

        if len(input_tensor.shape) == 2:
            input_tensor = input_tensor.unsqueeze(0)  # (1, T, D)

        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(input_tensor)

        # Benchmark
        latencies = []
        with torch.no_grad():
            for _ in range(iterations):
                t0 = time.perf_counter()
                _ = model(input_tensor)
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000.0)

        # Param count
        param_count = sum(p.numel() for p in model.parameters())
    else:
        # Scikit-Learn
        if isinstance(input_sample, torch.Tensor):
            input_arr = input_sample.detach().cpu().numpy()
        else:
            input_arr = np.array(input_sample)

        if len(input_arr.shape) == 1:
            input_arr = input_arr.reshape(1, -1)

        # Warmup
        for _ in range(warmup):
            _ = model.predict(input_arr)

        latencies = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            _ = model.predict(input_arr)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

        param_count = None

    latencies = np.array(latencies)
    mean_ms = float(np.mean(latencies))

    return {
        "iterations": iterations,
        "mean_latency_ms": mean_ms,
        "std_latency_ms": float(np.std(latencies)),
        "median_latency_ms": float(np.median(latencies)),
        "p90_latency_ms": float(np.percentile(latencies, 90)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "p99_latency_ms": float(np.percentile(latencies, 99)),
        "min_latency_ms": float(np.min(latencies)),
        "max_latency_ms": float(np.max(latencies)),
        "throughput_fps": float(1000.0 / max(1e-4, mean_ms)),
        "parameter_count": param_count,
    }
