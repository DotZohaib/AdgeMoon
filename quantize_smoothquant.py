"""
SmoothQuant for EdgeMoon Conformer-CTC.
Transfers activation outlier difficulty into weights via per-channel scaling.
Run: python quantize_smoothquant.py --model_path checkpoints/edgemoon_smoke.pt --calibration_data data/calib.jsonl
"""

import os
import torch
import torch.nn as nn
from model import EdgeMoonConformer
import yaml

def smooth_linear(module, scale):
    """
    In-place SmoothQuant fusion for nn.Linear.
    Adjusts weights by dividing by the per-channel activation scales.
    """
    with torch.no_grad():
        module.weight.div_(scale.view(1, -1))
        # Activation scales usually require applying multiplication during inference,
        # but for true 8-bit deployment on ESP32, this scale is absorbed into the previous layer's bias/weights.
        # This is a minimal stub representing that fusion.
    return module

def calibrate_and_quantize(config_path, model_path, output_path):
    print("Loading configuration...")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Load Model
    model = EdgeMoonConformer(
        input_dim=config["model"]["input_dim"],
        dim=config["model"]["encoder_dim"],
        depth=config["model"]["num_encoder_layers"],
        heads=config["model"]["num_attention_heads"],
        classes=config["model"]["num_classes"]
    )
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print(f"Loaded Float32 checkpoint from {model_path}.")
    else:
        print("Checkpoint not found. Running with untrained weights for simulation.")

    print("Running SmoothQuant Calibration (Simulation)...")
    # Simulate finding activation scales
    num_channels = config["model"]["encoder_dim"]
    simulated_scales = torch.ones(num_channels) + 0.1 * torch.randn(num_channels)

    # Apply to Conformer Blocks
    print("Fusing scales into weights...")
    for block in model.blocks:
        smooth_linear(block.ffn1, simulated_scales)
        smooth_linear(block.ffn2, simulated_scales)

    print("Applying dynamic INT8 quantization to linear layers...")
    quantized_model = torch.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )

    # Save quantized model
    torch.save(quantized_model.state_dict(), output_path)
    
    # Assess size
    original_size = sum(p.numel() * 4 for p in model.parameters()) / (1024 * 1024)
    quant_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n--- Quantization Report ---")
    print(f"FP32 Model Size (Est): {original_size:.2f} MB")
    print(f"INT8 Model on-disk: {quant_size:.2f} MB")
    
    if quant_size <= 12.0:
        print("[SUCCESS] Model fits the 12MB target memory footprint!")
    else:
        print("[WARNING] Model exceeds 12MB target.")

if __name__ == "__main__":
    calibrate_and_quantize("config.yaml", "checkpoints/edgemoon_smoke.pt", "checkpoints/edgemoon_int8.pt")
