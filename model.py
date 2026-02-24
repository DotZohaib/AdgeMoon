"""
EdgeMoon Core Model: Conformer-CTC with Region-Aware RoPE and Dynamic Routing Arbiter
Targets <= 30M parameters for edge devices.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RegionAwareRoPE(nn.Module):
    """
    Region-Aware Rotary Position Embedding.
    Applies concentrated attention resolution to local acoustic context windows
    and decays long-range embeddings exponentially.
    """
    def __init__(self, dim, base=10000, max_seq_len=2048):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = max_seq_len
        t = torch.arange(max_seq_len).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        # Apply exponential decay for long-range context
        decay = torch.exp(-0.01 * t).unsqueeze(1)
        self.register_buffer("cos_cached", (emb.cos() * decay)[None, None, :, :])
        self.register_buffer("sin_cached", (emb.sin() * decay)[None, None, :, :])

    def forward(self, x, seq_len=None):
        # x shape: [batch, heads, seq_len, head_dim]
        if seq_len > self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            decay = torch.exp(-0.01 * t).unsqueeze(1)
            self.register_buffer("cos_cached", (emb.cos() * decay)[None, None, :, :])
            self.register_buffer("sin_cached", (emb.sin() * decay)[None, None, :, :])
        return self.cos_cached[:, :, :seq_len, ...], self.sin_cached[:, :, :seq_len, ...]

def apply_rotary_pos_emb(q, k, cos, sin):
    # rotate half
    q_rotate = torch.cat((-q[..., q.shape[-1]//2:], q[..., :q.shape[-1]//2]), dim=-1)
    k_rotate = torch.cat((-k[..., k.shape[-1]//2:], k[..., :k.shape[-1]//2]), dim=-1)
    q_embed = (q * cos) + (q_rotate * sin)
    k_embed = (k * cos) + (k_rotate * sin)
    return q_embed, k_embed

class DynamicRoutingArbiter(nn.Module):
    """
    Gates and drops uninformative frames dynamically to save computation.
    """
    def __init__(self, dim, threshold=0.1):
        super().__init__()
        self.scorer = nn.Linear(dim, 1)
        self.threshold = threshold

    def forward(self, x, lengths):
        # x: [batch, time, dim]
        scores = torch.sigmoid(self.scorer(x)) # [batch, time, 1]
        mask = (scores > self.threshold).float()
        # In actual implementation for efficiency, we would physically drop frames.
        # For this stub, we apply a soft-masking to simulate the routing.
        return x * mask, mask.squeeze(-1)

class ConformerBlock(nn.Module):
    def __init__(self, dim, heads, rope_module):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.heads = heads
        self.rope = rope_module
        
        # FFN
        self.norm2 = nn.LayerNorm(dim)
        self.ffn1 = nn.Linear(dim, dim * 4)
        self.ffn2 = nn.Linear(dim * 4, dim)
        self.act = nn.SiLU()

    def forward(self, x):
        B, T, C = x.shape
        # Attention
        norm_x = self.norm1(x)
        qkv = self.qkv(norm_x).reshape(B, T, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        cos, sin = self.rope(q, seq_len=T)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(q.size(-1)))
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        x = x + self.proj(out)
        
        # FFN
        x = x + self.ffn2(self.act(self.ffn1(self.norm2(x))))
        return x

class EdgeMoonConformer(nn.Module):
    def __init__(self, input_dim=80, dim=256, depth=6, heads=4, classes=70):
        super().__init__()
        # Conv Subsampling
        self.subsample = nn.Sequential(
            nn.Conv2d(1, dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.proj = nn.Linear(dim * (((input_dim - 1) // 2 + 1) // 2), dim)
        
        self.rope = RegionAwareRoPE(dim // heads)
        self.dra = DynamicRoutingArbiter(dim)
        self.blocks = nn.ModuleList([ConformerBlock(dim, heads, self.rope) for _ in range(depth)])
        self.classifier = nn.Linear(dim, classes)

    def forward(self, mel_spec, lengths):
        # mel_spec: [batch, time, freq]
        x = mel_spec.unsqueeze(1).transpose(2, 3) # [batch, 1, freq, time]
        x = self.subsample(x) # [batch, dim, freq', time']
        B, C, F, T = x.shape
        x = x.reshape(B, C * F, T).transpose(1, 2) # [batch, time, dim_proj]
        x = self.proj(x)
        
        # Subsample lengths (divide by 4 due to 2x stride-2 convs)
        lengths = lengths // 4
        
        x, routing_mask = self.dra(x, lengths)
        for block in self.blocks:
            x = block(x)
            
        logits = self.classifier(x)
        return logits, lengths

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # Smoke test
    model = EdgeMoonConformer()
    print(f"EdgeMoon Conformer Parameters: {count_parameters(model) / 1e6:.2f} M (Target: <30M)")
    assert count_parameters(model) < 30000000, "Model is too large!"
    
    dummy_input = torch.randn(2, 500, 80)
    dummy_lengths = torch.tensor([500, 400])
    logits, lengths = model(dummy_input, dummy_lengths)
    print(f"Output shape: {logits.shape}")

