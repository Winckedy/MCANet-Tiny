"""INT8 quantization and mobile export utilities."""
import copy
import traceback
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from torch.utils.mobile_optimizer import optimize_for_mobile


def build_qconfig_mapping(backend: str = "qnnpack"):
    """Build quantization config mapping for MCANet-Tiny."""
    qconfig_mapping = get_default_qconfig_mapping(backend)
    return qconfig_mapping


def quantize_int8_static_fx(
    model: nn.Module,
    calib_loader: DataLoader,
    backend: str = "qnnpack"
) -> nn.Module:
    """
    Perform INT8 static quantization using FX graph mode.
    Model must be on CPU and in eval mode.
    """
    model = copy.deepcopy(model).cpu().eval()
    qconfig_mapping = build_qconfig_mapping(backend)
    example_inputs = (torch.randn(1, 3, 224, 224),)
    prepared = prepare_fx(model, qconfig_mapping, example_inputs)

    print("Performing INT8 calibration (50 batches)...")
    with torch.no_grad():
        for i, (x, _) in enumerate(calib_loader):
            if i >= 50:
                break
            prepared(x.cpu())

    quantized_model = convert_fx(prepared)
    print("INT8 quantization completed.")
    return quantized_model


def export_to_ptl(model: nn.Module, save_path: Path, model_name: str = "model"):
    """
    Export model to PyTorch Mobile compatible .ptl file.
    Model must be on CPU and in eval mode.
    """
    model = model.cpu().eval()
    example_input = torch.randn(1, 3, 224, 224)
    try:
        traced = torch.jit.trace(model, example_input)
        optimized = optimize_for_mobile(traced)
        optimized._save_for_lite_interpreter(str(save_path))
        print(f"Successfully exported PTL model: {save_path} ({model_name})")
    except Exception as e:
        print(f"Failed to export PTL model ({model_name}): {e}")
        traceback.print_exc()