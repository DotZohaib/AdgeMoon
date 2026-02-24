# EdgeMoon Project Deliverables

This repository contains the EdgeMoon project, a complete pipeline for ultra-lightweight South Asian (Sindhi/Urdu) speech recognition and translation on edge devices.

## Requirements
```bash
pip install torch torchaudio pyyaml
# For quantization, evaluation, and data generation stubs, minimal standard libraries are used where applicable.
# See individual scripts for specific details.
```

## Repository Structure

| File | Description |
|------|-------------|
| `paper.md` | IEEE-format paper summarizing the EdgeMoon methods and findings. |
| `demo_pipeline.sh` | Shell script demonstrating STT -> Translation -> TTS pipeline. |
| `config.yaml` | Master configuration for architecture and training hyperparameters. |
| `model.py` | PyTorch implementation of Conformer-CTC with Region-Aware RoPE & Dynamic Routing Arbiter. |
| `train.py` | Training loop with Mixed Precision and SpecAugment. |
| `quantize_smoothquant.py` | INT8 post-training quantization via SmoothQuant calibration. |
| `synth_vits_augment.py` | Data augmentation via VITS-like TTS to expand <100h training data. |
| `nfa_vac_aligner.py` | Forced alignment script for audio-text chunks using NFA/VAC. |
| `eval.py` | Evaluation tools computing WER, CER, RTF, and Memory. |
| `ed_fed_server.py` | Federated learning server managing LoRA parameter aggregation. |
| `ed_fed_client.py` | Federated learning client executing local LoRA personalization. |
| `mc_infer.stub` | C pseudocode and notes for TFLite Micro compatible deployment on ESP32 class HW. |

## Quickstart Smoke Test

Run a minimal 1-epoch training loop on CPU or GPU to verify the setup.

```bash
python train.py --config config.yaml
```

*For more details on hyperparameter choices, synthetic data generation thresholds, and hardware measurements, please read the IEEE `paper.md` and check the individual python file headers.*
