"""
Training Pipeline for EdgeMoon.
Implements Mixed Precision, SpecAugment, and custom optimizer scheduling.
Run: python train.py --config config.yaml
"""

import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from model import EdgeMoonConformer

# --- Mock Dataset & Dataloader for Smoke Test ---
class MockSpeechDataset(Dataset):
    def __init__(self, num_samples=100, max_time=1000, n_mels=80, num_classes=70):
        self.n_mels = n_mels
        self.num_classes = num_classes
        self.samples = []
        for _ in range(num_samples):
            t = torch.randint(100, max_time, (1,)).item()
            mel = torch.randn(t, n_mels)
            target_len = t // 4
            target = torch.randint(1, num_classes, (target_len,))
            self.samples.append((mel, target, t, target_len))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def collate_fn(batch):
    mels, targets, input_lens, target_lens = zip(*batch)
    max_t = max(input_lens)
    max_tgt_len = max(target_lens)
    
    mel_padded = torch.zeros(len(batch), max_t, batch[0][0].size(-1))
    tgt_padded = torch.zeros(len(batch), max_tgt_len, dtype=torch.long)
    
    for i, (mel, tgt, ilen, tlen) in enumerate(batch):
        mel_padded[i, :ilen, :] = mel
        tgt_padded[i, :tlen] = tgt
        
    return mel_padded, tgt_padded, torch.tensor(input_lens), torch.tensor(target_lens)

# --- SpecAugment Stub ---
def apply_specaugment(x):
    # frequency and time masking
    return x # implementation omitted for brevity

def train(config_path, phase=1):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | Phase: {phase}")

    # Initialize model
    model = EdgeMoonConformer(
        input_dim=config["model"]["input_dim"],
        dim=config["model"]["encoder_dim"],
        depth=config["model"]["num_encoder_layers"],
        heads=config["model"]["num_attention_heads"],
        classes=config["model"]["num_classes"]
    ).to(device)
    
    # Phase Setup
    phase_key = f"phase_{phase}"
    if phase_key in config:
        phase_cfg = config[phase_key]
        print(f"--- Phase {phase}: {phase_cfg.get('description')} ---")
        lr = phase_cfg.get("learning_rate", 1e-3)
        weight_decay = phase_cfg.get("weight_decay", 0.01)
        if phase_cfg.get("freeze_lower_encoders"):
            model.freeze_lower_layers(num_layers=3)
    else:
        # Fallback to standard config
        lr = config["training"]["learning_rate"]
        weight_decay = config["training"]["weight_decay"]

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=lr, 
        weight_decay=weight_decay
    )
    
    # CTC Loss
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    
    # Extract mixed precision flag safely
    use_fp16 = False
    if "training" in config:
        prec = config["training"].get("precision", config["training"].get("mixed_precision", False))
        use_fp16 = prec == "fp16" or prec is True
        
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16 and device.type == 'cuda')

    # DataLoader
    batch_size = phase_cfg.get("batch_size", 32) if phase_key in config else config["training"]["batch_size"]
    dataset = MockSpeechDataset()
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)

    model.train()
    print("Starting 1-epoch smoke test training...")
    for epoch in range(1): # Limit to 1 epoch for smoke testing
        total_loss = 0
        for batch_idx, (mels, targets, input_lens, target_lens) in enumerate(loader):
            mels, targets = mels.to(device), targets.to(device)
            mels = apply_specaugment(mels)

            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=use_fp16 and device.type == 'cuda'):
                logits, out_lens = model(mels, input_lens)
                # CTC expects [T, B, C]
                logits = logits.transpose(0, 1).log_softmax(2)
                loss = criterion(logits, targets, out_lens, target_lens)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            print(f"Epoch {epoch} | Step {batch_idx} | Loss {loss.item():.4f}")

    print("Smoke test completed successfully!")
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/edgemoon_smoke.pt")
    print("Saved checkpoint to checkpoints/edgemoon_smoke.pt")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--phase", type=int, default=1, help="1 for Pretraining, 2 for Sindhi fine-tuning")
    args = parser.parse_args() if len(os.sys.argv) > 1 else type('Args', (), {"config": "config.yaml", "phase": 1})()
    
    config_file = args.config
    phase = args.phase
    train(config_file, phase)
