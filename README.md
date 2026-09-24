# Low-Latency Privacy-Preserving ASL Recognition via Coordinate Trajectory Tracking

This repository contains the complete implementation for **Week 1** and **Week 2** of the American Sign Language (ASL) recognition system.

The project investigates low-latency, privacy-preserving dynamic word-level sign language recognition from monocular webcam feeds. Rather than relying on computationally heavy 3D-CNNs operating directly on high-dimensional raw video frames, the architecture decouples geometric feature extraction from temporal sequence modeling:
1. **Geometric Feature Extraction**: Extracts 21 3D hand keypoints per frame via Google MediaPipe Hands (63 float coordinates for single hand, 126 for dual-hand).
2. **Spatial Normalization**: Zero-centers coordinates at the wrist (Landmark 0) and scales by the Euclidean distance between the wrist and the middle-finger MCP joint (Landmark 9) for scale and translation invariance.
3. **Temporal Resampling**: Resamples variable-length coordinate trajectories (15–75 frames) to a uniform temporal window of $T = 30$ frames (1.0 second at 30 FPS).
4. **PyTorch Dataset & DataLoader**: Batches tensors into shapes of `(batch_size, 30, 63)` with kinematic coordinate augmentations.
5. **Non-Deep-Learning Baselines**: Extracts rich spatial and kinematic trajectory statistics (mean, variance, velocity, acceleration, path length) and evaluates **Random Forest** and **Support Vector Classifiers (SVC)** with **Top-1**, **Top-5**, and **Macro F1-score** metrics.

---

## Repository Structure

```
handsignlang/
├── config/
│   └── vocabulary.json             # 30-word dynamic target vocabulary configuration
├── data/
│   ├── annotations/
│   │   ├── WLASL_v0.3.json         # Full WLASL index and video metadata
│   │   └── wlasl_30_subset.json    # Isolated 30-word subset metadata (547 instances)
│   ├── raw/                        # Video clips & generated benchmark instances
│   └── processed/
│       └── benchmark_results.json  # Exported metrics (Top-1, Top-5, Macro F1, confusion matrices)
├── models/
│   ├── hand_landmarker.task        # MediaPipe HandLandmarker model bundle
│   ├── random_forest_baseline.joblib # Persisted Random Forest classifier
│   └── svc_baseline.joblib          # Persisted Support Vector Classifier pipeline
├── src/
│   ├── __init__.py
│   ├── data_ingestion.py           # WLASL metadata ingestion, subset isolation, synthetic clip generation
│   ├── keypoint_extractor.py       # MediaPipe HandLandmarker loop & missing frame linear interpolation
│   ├── preprocessor.py             # Spatial normalization & temporal spline resampling
│   ├── dataset.py                  # PyTorch ASLKeypointDataset, DataLoader & kinematic augmentations
│   ├── feature_extraction.py       # Trajectory feature engineering (spatial stats, velocity, acceleration)
│   ├── baselines.py                # ML baseline models (Random Forest, SVC) and Top-K evaluation
│   └── profiler.py                 # Latency, FPS throughput, and CPU profiling
├── scripts/
│   ├── run_week1_pipeline.py       # Week 1 end-to-end verification and profiling runner
│   ├── run_week2_pipeline.py       # Week 2 spatial norm, PyTorch DataLoader & ML baseline benchmarks
│   └── demo_webcam.py              # Real-time interactive webcam sign HUD and classifier demo
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py            # Unit tests for invariance, interpolation, dataset, and models
├── requirements.txt
└── README.md
```

---

## Week 1 Deliverables Summary

1. **Environment Configuration & Dependencies**:
   - Python 3.12 verified with `PyTorch 2.14.0+cpu`, `OpenCV 5.0.0`, `MediaPipe 1.0.1`, and `Scikit-Learn 1.9.0`.
   - Automatic model bundle fetching for MediaPipe's modern Tasks API (`models/hand_landmarker.task`).
2. **Data Ingestion and Subset Isolation**:
   - Downloaded and parsed `WLASL_v0.3.json`.
   - Filtered down to an initial target **30-word dynamic vocabulary subset** (greetings, common verbs, emergency terms, nouns, adjectives).
   - Isolated **547 instances** (Train: 383, Val: 94, Test: 70) and saved to `data/annotations/wlasl_30_subset.json`.
3. **Keypoint Extraction Script**:
   - Implemented `MediaPipeKeypointExtractor` with OpenCV stream reading.
   - Extracts 21 3D coordinates $(x, y, z)$ per frame ($63$ float features per frame).
   - Handled missing and occluded frames via **linear interpolation** with leading/trailing boundary fills.
4. **Frame Rate and Throughput Profiling**:
   - Measured average inference throughput: **43.7 FPS on standard CPU** (latency: 22.86 ms/frame).
   - Confirmed real-time operational feasibility exceeding the 30 FPS webcam standard without requiring GPU acceleration.
5. **Temporal Resampling Utility**:
   - Implemented `resample_temporal_trajectory` to convert variable-length sequences (15–75 frames) into a uniform $T = 30$ frames window.

---

## Week 2 Deliverables Summary

1. **Spatial Normalization**:
   - **Wrist Centering**: Zero-centers coordinates relative to Landmark 0 (wrist):
     $$P'_i = P_i - P_{\text{wrist}} \quad \forall i \in [0, 20]$$
   - **MCP Scale Normalization**: Scales by the Euclidean distance to Landmark 9 (middle MCP joint):
     $$d = \|P_{\text{mcp}} - P_{\text{wrist}}\|_2, \quad P''_i = \frac{P'_i}{d + \epsilon}$$
   - Formally proved scale and position invariance with max absolute difference $< 10^{-4}$ under arbitrary affine shifts.
2. **PyTorch Dataset and DataLoader**:
   - Implemented `ASLKeypointDataset` and `create_dataloaders`.
   - Yields batches of shape `(batch_size, 30, 63)` as `torch.FloatTensor`.
   - Includes kinematic data augmentations: coordinate Gaussian jittering, random spatial scaling, and in-plane rotation.
3. **Trajectory Feature Engineering**:
   - Implemented `extract_trajectory_features_single` and `extract_trajectory_features_batch`:
     - Mean, standard deviation, min, max, range per landmark.
     - Net displacement ($P_{T-1} - P_0$).
     - First-order velocity dynamics ($\Delta P / \Delta t$): mean, standard deviation, max velocity.
     - Second-order acceleration dynamics ($\Delta^2 P / \Delta t^2$): mean, max acceleration.
     - Cumulative 3D Euclidean path length per landmark.
4. **Non-Deep-Learning Baselines**:
   - Trained **Random Forest** (150 trees, balanced weights) and **Support Vector Classifier (SVC)** (RBF kernel, Standard Scaler).
   - Evaluated using **Top-1 Accuracy**, **Top-5 Accuracy**, and **Macro-Averaged F1-Score**.
   - Saved models to `models/` and exported metrics to `data/processed/benchmark_results.json`.

---

## How to Run

### 1. Execute Week 1 Pipeline
Runs environment check, WLASL metadata ingestion, keypoint extraction, occlusion interpolation, throughput profiling, and temporal resampling:
```bash
python scripts/run_week1_pipeline.py
```

### 2. Execute Week 2 Pipeline
Runs spatial invariance verification, benchmark dataset generation, PyTorch DataLoader validation, trajectory feature engineering, baseline training (Random Forest & SVC), and evaluation:
```bash
python scripts/run_week2_pipeline.py
```

### 3. Run Real-Time Webcam Demo
Launches the live webcam feed with skeleton tracking HUD and dynamic sign classification:
```bash
python scripts/demo_webcam.py
```

### 4. Run Unit Tests
Executes the comprehensive unit test suite:
```bash
python -m unittest tests/test_pipeline.py
```

---

## Answers to Questions & Feedback

### Question 1: Single-hand (63-d) vs. Two-hand (126-d zero-padded) Representation
> **Recommendation:** We recommend adopting the **zero-padded 126-dimensional representation (42 keypoints $\times$ 3 coordinates)** as the standard tensor format across the pipeline.
>
> **Rationale:**
> 1. In WLASL, many vital signs (e.g., *"book"*, *"family"*, *"more"*, *"stop"*, *"house"*) are fundamentally two-handed signs. Restricting the architecture to 63 dimensions (single hand) artificially limits vocabulary scope and causes severe ambiguity when signs require interacting hands.
> 2. Zero-padding the second hand when absent preserves temporal and dimensional homogeneity for PyTorch batching without distorting single-handed trajectories.
> 3. We have implemented flexible support: single-hand mode outputs `(30, 63)` and dual-hand mode outputs `(30, 126)` seamlessly with automated zero-padding.

### Question 2: Top-1 Accuracy vs. Top-5 Accuracy & Macro-Averaged F1-Score
> **Recommendation:** We strongly recommend reporting **both Top-1 and Top-5 accuracy, accompanied by the Macro-Averaged F1-score**.
>
> **Rationale:**
> 1. **Top-5 Accuracy** is standard in sign language benchmarks (e.g., WLASL, MS-ASL, Signum) because sign language contains subtle fine-grained phonetic variations (minimal pairs differing only by minor finger curl or facial context). Top-5 demonstrates whether the model's posterior probability distribution consistently places the correct sign in the active candidate pool.
> 2. **Macro-Averaged F1-Score** is essential because real-world sign datasets exhibit natural class imbalance across words. Macro F1 weights every gloss equally, preventing the benchmark from being dominated by frequent signs and exposing per-class deficiencies.
> 3. Our evaluation module (`src/baselines.py`) computes and exports Top-1, Top-5, Macro F1, Weighted F1, precision, recall, and confusion matrices.
