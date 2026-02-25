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

### Local Bilingual Voice Bridge Extension
| File | Description |
|------|-------------|
| `lan_server.py` | TCP Router simulating a WebSocket Relay holding client-maps without raw audio access. |
| `lan_client.py` | Local execution relay to transmit/receive JSON STT/TTS instructions. |
| `stream_audio.py` | Generates 20ms chunks formatted as 16kHz PCM blocks. |
| `translator.py` | 10M Parameter Stub simulating offline model matching NLLB-200. |
| `tts.py` | VITS / FastSpeech offline payload generation. |
| `realtime_pipeline.py` | Multithreaded orchestration of Stream->STT->Translate->Network->TTS. |
| `demo_lan.py` | Spins up local server and two client instances (Simulates Sindhi User A talking to English User B). |

## Quickstart Smoke Tests

**Model Training Simulation**
Run a minimal 1-epoch training loop on CPU to verify the setup.
```bash
python train.py --config config.yaml
```

**Local LAN Voice Bridge Demo**
Runs a simulated multi-process LAN interaction. (Use python 3.11 environment).
```bash
python demo_lan.py
or
py -3.11 demo_lan.py
```

*For more details on hyperparameter choices, synthetic data generation thresholds, and hardware measurements, please read the IEEE `paper.md` and check the individual python file headers.*
