"""
Non-Deep-Learning Benchmark Baselines Module.
Trains and evaluates Random Forest and Support Vector Classifier (SVC) models
on kinematic trajectory statistics. Computes Top-1, Top-5 accuracy, and Macro F1.
"""

import os
import logging
import joblib
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)
from sklearn.calibration import CalibratedClassifierCV

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_top_k_accuracy(y_true: np.ndarray, y_proba: np.ndarray, k: int = 5) -> float:
    """
    Computes categorical Top-K accuracy given true integer labels and predicted probability matrix.
    """
    k = min(k, y_proba.shape[1])
    top_k_preds = np.argsort(y_proba, axis=1)[:, -k:]
    hits = [y in top_k_preds[i] for i, y in enumerate(y_true)]
    return float(np.mean(hits))


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    vocab: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Comprehensive evaluation of a trained classifier.
    Computes:
      - Top-1 Categorical Accuracy
      - Top-5 Categorical Accuracy (if model outputs probabilities)
      - Macro-Averaged F1-Score
      - Weighted F1-Score, Precision, Recall
      - Per-class classification report
      - Confusion matrix
    """
    y_pred = model.predict(X_test)

    top_1 = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_test, y_pred, average="macro", zero_division=0)

    # Top-5 Accuracy
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)
        top_5 = compute_top_k_accuracy(y_test, y_proba, k=5)
    else:
        top_5 = top_1

    target_names = vocab if (vocab and len(vocab) == len(np.unique(y_test))) else None
    report = classification_report(y_test, y_pred, target_names=target_names, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "top_1_accuracy": float(top_1),
        "top_5_accuracy": float(top_5),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }

    return metrics


class ASLBaselineBenchmarks:
    """
    Manager for training, evaluating, and persisting classical ML baselines.
    """

    def __init__(self, vocab: Optional[List[str]] = None):
        self.vocab = vocab
        self.models = {}

    def train_random_forest(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_estimators: int = 150,
        max_depth: Optional[int] = 18,
        random_state: int = 42,
    ) -> RandomForestClassifier:
        """Trains a tuned Random Forest Classifier."""
        logger.info(f"Training Random Forest (n_estimators={n_estimators}, max_depth={max_depth})...")
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)
        self.models["random_forest"] = rf
        return rf

    def train_svc(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        C: float = 3.0,
        gamma: str = "scale",
        random_state: int = 42,
    ) -> Pipeline:
        """Trains an RBF Support Vector Classifier with standard scaling pipeline."""
        logger.info(f"Training Support Vector Classifier (RBF kernel, C={C})...")
        base_svc = SVC(C=C, kernel="rbf", gamma=gamma, random_state=random_state)
        calibrated_svc = CalibratedClassifierCV(estimator=base_svc, ensemble=False)
        svc_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("svc", calibrated_svc),
        ])
        svc_pipe.fit(X_train, y_train)
        self.models["svc"] = svc_pipe
        return svc_pipe

    def evaluate_all(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, Dict[str, Any]]:
        """Evaluates all trained models on test set."""
        results = {}
        for name, model in self.models.items():
            logger.info(f"Evaluating model: {name} on {len(y_test)} test instances...")
            res = evaluate_model(model, X_test, y_test, vocab=self.vocab)
            results[name] = res
            logger.info(
                f"[{name.upper()}] Top-1: {res['top_1_accuracy']*100:.2f}% | "
                f"Top-5: {res['top_5_accuracy']*100:.2f}% | "
                f"Macro F1: {res['macro_f1']*100:.2f}%"
            )
        return results

    def save_model(self, model_name: str, output_path: str):
        """Persists trained model using joblib."""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' has not been trained.")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        joblib.dump(self.models[model_name], output_path)
        logger.info(f"Saved {model_name} to {output_path}")

    @staticmethod
    def load_model(model_path: str) -> Any:
        """Loads a persisted model."""
        return joblib.load(model_path)
