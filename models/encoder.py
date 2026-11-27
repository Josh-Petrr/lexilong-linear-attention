"""
Tiny Transformer encoder classifier shared by all attention variants.
Only the attention module differs between the baseline and the improved models,
so any change in speed / memory / accuracy is attributable to attention alone.
"""
import math

import torch
import torch.nn as nn

from .attention import build_attention


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sin/cos encoding (Vaswani et al.) - computed on the fly, so any length works."""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(self, x):
        N, D = x.shape[1], self.d_model
        pos = torch.arange(N, device=x.device, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, D, 2, device=x.device, dtype=torch.float32) * (-math.log(10000.0) / D))
        pe = torch.zeros(N, D, device=x.device)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return x + pe.unsqueeze(0).to(x.dtype)


class EncoderBlock(nn.Module):
    """Pre-LayerNorm Transformer block: x + Attn(LN(x)); x + FFN(LN(x))."""

    def __init__(self, attn_type, d_model, n_heads, d_ff, dropout, max_len):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = build_attention(attn_type, d_model, n_heads, dropout, max_len)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model)
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        x = x + self.drop(self.attn(self.ln1(x), mask))
        x = x + self.drop(self.ffn(self.ln2(x)))
        return x


class LexiLongClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        attn_type: str = "softmax",
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 512,
        n_classes: int = 2,
        dropout: float = 0.1,
        max_len: int = 8192,
        pad_id: int = 0,
    ):
        super().__init__()
        self.config = dict(
            vocab_size=vocab_size, attn_type=attn_type, d_model=d_model, n_heads=n_heads,
            n_layers=n_layers, d_ff=d_ff, n_classes=n_classes, dropout=dropout,
            max_len=max_len, pad_id=pad_id,
        )
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos = SinusoidalPositionalEncoding(d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [EncoderBlock(attn_type, d_model, n_heads, d_ff, dropout, max_len) for _ in range(n_layers)]
        )
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, input_ids, attention_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        mask = attention_mask.bool()
        x = self.drop(self.pos(self.embed(input_ids) * math.sqrt(self.config["d_model"])))
        for blk in self.blocks:
            x = blk(x, mask)
        x = self.ln(x)
        # Masked mean pooling over real tokens.
        m = mask.unsqueeze(-1).to(x.dtype)
        pooled = (x * m).sum(1) / m.sum(1).clamp(min=1.0)
        return self.head(pooled)

    @classmethod
    def from_checkpoint(cls, path, map_location="cpu"):
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(**ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
        return model
