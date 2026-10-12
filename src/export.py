"""ONNX export and optional ONNX Runtime dynamic INT8 quantization helpers."""

from pathlib import Path
from typing import Dict
import torch
import torch.nn as nn


def export_onnx_model(model: nn.Module, output_path: str, input_dim: int, seq_len: int = 30) -> str:
    """Exports a CPU model with dynamic batch/time axes and returns its path."""
    model = model.cpu().eval()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sample = torch.zeros(1, seq_len, input_dim, dtype=torch.float32)
    with torch.no_grad():
        torch.onnx.export(
            model, sample, str(destination), input_names=["keypoints"], output_names=["logits"],
            dynamic_axes={"keypoints": {0: "batch", 1: "frames"}, "logits": {0: "batch"},},
            opset_version=17,
        )
    return str(destination)


def quantize_onnx_dynamic(onnx_path: str, output_path: str) -> Dict[str, str]:
    """Dynamically quantizes ONNX weights to INT8 when onnxruntime is installed."""
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError as error:
        raise RuntimeError("ONNX Runtime is optional; install onnx and onnxruntime to quantize exports.") from error
    quantize_dynamic(onnx_path, output_path, weight_type=QuantType.QInt8)
    return {"onnx_path": onnx_path, "quantized_path": output_path}
