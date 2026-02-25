#!/usr/bin/env bash

# EdgeMoon Large-Scale Training Pipeline Execution Script

echo "======================================================"
echo "    EdgeMoon Multilingual Large-Scale Pretraining     "
echo "======================================================"

echo "Step 1: Downloading & Preprocessing Datasets..."
# This generates the unified metadata_large.jsonl manifest containing all curated speech chunks
python prep_dataset.py
sleep 1

echo ""
echo "Step 2: Commencing Phase 1 (Multilingual Pretraining)..."
# Trains the model on the full 10k hours of mixed English/Sindhi corpus.
# High Learning Rate, All Layers Unfrozen, Warmup 10k steps.
python train.py --config config_large.yaml --phase 1
sleep 1

echo ""
echo "Step 3: Commencing Phase 2 (Sindhi Focused Fine-tuning)..."
# Freezes lower Conformer layers. Focuses solely on Sindhi data distribution.
# Learning Rate reduced 10x natively via the config parser.
python train.py --config config_large.yaml --phase 2

echo ""
echo "======================================================"
echo "    Pipeline Simulation Complete. Found Weights.      "
echo "======================================================"
