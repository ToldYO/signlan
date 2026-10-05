"""
Training and Evaluation Pipeline for Deep ASL Dynamic Models.
Implements:
1. Training loop with AdamW, Cosine Annealing, Gradient Clipping, and Label Smoothing.
2. Multi-metric evaluation (Cross-Entropy Loss, Top-1 Acc, Top-5 Acc, Macro F1, Weighted F1).
3. Early stopping with patience and model checkpoint persistence.
4. Comprehensive training curves tracking.
"""

import os
import time
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report, confusion_matrix

from src.baselines import compute_top_k_accuracy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class ASLModelTrainer:
    """
    Standardized trainer for PyTorch ASL sequence models (BiGRU, 1D-TCN).
    """

    def __init__(
        self,
        model: nn.Module,
        num_classes: int = 30,
        device: Optional[torch.device] = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        label_smoothing: float = 0.05,
        grad_clip: float = 1.0,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.num_classes = num_classes
        self.grad_clip = grad_clip

        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Runs a single training epoch."""
        self.model.train()
        total_loss = 0.0
        correct_top1 = 0
        total_samples = 0

        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(batch_x)
            loss = self.criterion(logits, batch_y)
            loss.backward()

            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()

            bs = batch_x.size(0)
            total_loss += loss.item() * bs
            preds = torch.argmax(logits, dim=1)
            correct_top1 += (preds == batch_y).sum().item()
            total_samples += bs

        epoch_loss = total_loss / max(1, total_samples)
        epoch_top1 = correct_top1 / max(1, total_samples)

        return {"loss": epoch_loss, "top_1_accuracy": epoch_top1}

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
        target_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Runs model evaluation over a validation or test DataLoader.
        Computes Top-1, Top-5, Macro F1, Weighted F1, precision, recall, and confusion matrix.
        """
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        all_preds = []
        all_probs = []
        all_targets = []

        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            logits = self.model(batch_x)
            loss = self.criterion(logits, batch_y)

            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            bs = batch_x.size(0)
            total_loss += loss.item() * bs
            total_samples += bs

            all_preds.extend(preds.tolist())
            all_probs.append(probs)
            all_targets.extend(batch_y.cpu().numpy().tolist())

        all_probs = np.vstack(all_probs)
        all_targets = np.array(all_targets, dtype=np.int64)
        all_preds = np.array(all_preds, dtype=np.int64)

        avg_loss = total_loss / max(1, total_samples)
        top1_acc = float(np.mean(all_preds == all_targets))
        top5_acc = compute_top_k_accuracy(all_targets, all_probs, k=min(5, self.num_classes))
        macro_f1 = float(f1_score(all_targets, all_preds, average="macro", zero_division=0))
        weighted_f1 = float(f1_score(all_targets, all_preds, average="weighted", zero_division=0))
        macro_prec = float(precision_score(all_targets, all_preds, average="macro", zero_division=0))
        macro_rec = float(recall_score(all_targets, all_preds, average="macro", zero_division=0))

        report_str = classification_report(
            all_targets,
            all_preds,
            target_names=target_names,
            zero_division=0,
        )
        conf_mat = confusion_matrix(
            all_targets,
            all_preds,
            labels=list(range(self.num_classes)),
        ).tolist()

        return {
            "loss": avg_loss,
            "top_1_accuracy": top1_acc,
            "top_5_accuracy": top5_acc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "macro_precision": macro_prec,
            "macro_recall": macro_rec,
            "classification_report": report_str,
            "confusion_matrix": conf_mat,
            "probabilities": all_probs,
            "predictions": all_preds,
            "targets": all_targets,
        }

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 40,
        patience: int = 12,
        save_path: Optional[str] = None,
        target_names: Optional[List[str]] = None,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Trains model with early stopping based on validation Macro F1 score and Top-1 accuracy.
        """
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=1e-5
        )

        history = {
            "train_loss": [],
            "train_top1": [],
            "val_loss": [],
            "val_top1": [],
            "val_top5": [],
            "val_macro_f1": [],
        }

        best_score = -1.0
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_metrics = self.train_epoch(train_loader)
            self.scheduler.step()

            history["train_loss"].append(train_metrics["loss"])
            history["train_top1"].append(train_metrics["top_1_accuracy"])

            if val_loader is not None:
                val_metrics = self.evaluate(val_loader, target_names=target_names)
                history["val_loss"].append(val_metrics["loss"])
                history["val_top1"].append(val_metrics["top_1_accuracy"])
                history["val_top5"].append(val_metrics["top_5_accuracy"])
                history["val_macro_f1"].append(val_metrics["macro_f1"])

                # Combined validation score (primary: macro F1 + secondary: top 1)
                combined_score = val_metrics["macro_f1"] * 0.6 + val_metrics["top_1_accuracy"] * 0.4

                elapsed = time.time() - t0
                if verbose and (epoch % 5 == 0 or epoch == 1 or epoch == epochs):
                    logger.info(
                        f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) - "
                        f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['top_1_accuracy']*100:.1f}% | "
                        f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['top_1_accuracy']*100:.1f}%, "
                        f"Val Top-5: {val_metrics['top_5_accuracy']*100:.1f}%, Val F1: {val_metrics['macro_f1']*100:.1f}%"
                    )

                if combined_score > best_score:
                    best_score = combined_score
                    patience_counter = 0
                    if save_path:
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        torch.save({
                            "epoch": epoch,
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "val_metrics": {
                                "top_1": val_metrics["top_1_accuracy"],
                                "top_5": val_metrics["top_5_accuracy"],
                                "macro_f1": val_metrics["macro_f1"],
                            },
                        }, save_path)
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        if verbose:
                            logger.info(f"Early stopping triggered at epoch {epoch}.")
                        break
            else:
                if verbose and (epoch % 5 == 0 or epoch == epochs):
                    logger.info(
                        f"Epoch [{epoch:02d}/{epochs:02d}] - "
                        f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['top_1_accuracy']*100:.1f}%"
                    )

        # Reload best checkpoint if saved
        if save_path and os.path.exists(save_path):
            checkpoint = torch.load(save_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            if verbose:
                logger.info(f"Reloaded best model weights from epoch {checkpoint.get('epoch', '?')}.")

        return history
