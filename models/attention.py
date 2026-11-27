"""
Attention variants for LexiLong.

All three modules share the same interface:
    forward(x: [B, N, D], mask: [B, N] bool (True = real token)) -> [B, N, D]

1. SoftmaxAttention   - standard scaled dot-product attention, O(N^2) time & memory.
                        Implemented with explicit matmuls (no fused/flash kernel) so the
                        quadratic attention matrix is actually materialised - this is the
                        "textbook" Transformer from Vaswani et al. (2017).
2. LinearAttention    - Katharopoulos et al. (2020), "Transformers are RNNs".
                        softmax(QK^T)V  ->  phi(Q) (phi(K)^T V) / (phi(Q) sum_j phi(K_j))
                        with phi(x) = elu(x) + 1.  O(N d^2) time, O(N d + d^2) memory.
3. LinformerAttention - Wang et al. (2020). Projects K and V along the sequence axis
                        from N down to a fixed k with a learned matrix E, giving an
                        N x k attention matrix: O(N k) time & memory.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _MultiHeadBase(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split(self, x):
        # [B, N, 3D] -> 3 x [B, H, N, Dh]
        B, N, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        shape = (B, N, self.n_heads, self.d_head)
        return (t.view(shape).transpose(1, 2) for t in (q, k, v))

    def _merge(self, x):
        # [B, H, N, Dh] -> [B, N, D]
        B, H, N, Dh = x.shape
        return self.out(x.transpose(1, 2).reshape(B, N, H * Dh))


class SoftmaxAttention(_MultiHeadBase):
    """Baseline: full O(N^2) softmax attention."""

    def forward(self, x, mask=None):
        q, k, v = self._split(x)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)  # [B,H,N,N]
        if mask is not None:
            scores = scores.masked_fill(~mask[:, None, None, :], float("-inf"))
        attn = self.dropout(torch.softmax(scores, dim=-1))
        return self._merge(torch.matmul(attn, v))


class LinearAttention(_MultiHeadBase):
    """Improved: kernelised linear attention with phi(x) = elu(x) + 1."""

    def __init__(self, d_model, n_heads, dropout=0.1, eps: float = 1e-6):
        super().__init__(d_model, n_heads, dropout)
        self.eps = eps

    @staticmethod
    def feature_map(x):
        return F.elu(x) + 1.0

    def forward(self, x, mask=None):
        q, k, v = self._split(x)
        # The running sums over N can overflow in half precision -> compute in fp32.
        orig_dtype = q.dtype
        q, k, v = q.float(), k.float(), v.float()
        q, k = self.feature_map(q), self.feature_map(k)
        if mask is not None:
            # Padding tokens must contribute nothing to the K^T V summary or the normaliser.
            m = mask[:, None, :, None].to(k.dtype)
            k = k * m
            v = v * m
        kv = torch.einsum("bhnd,bhne->bhde", k, v)             # [B,H,Dh,Dh]  - size independent of N
        k_sum = k.sum(dim=2)                                   # [B,H,Dh]
        z = 1.0 / (torch.einsum("bhnd,bhd->bhn", q, k_sum) + self.eps)  # [B,H,N]
        out = torch.einsum("bhnd,bhde,bhn->bhne", q, kv, z)    # [B,H,N,Dh]
        out = self.dropout(out).to(orig_dtype)
        return self._merge(out)


class LinformerAttention(_MultiHeadBase):
    """Second improvement: low-rank projection of K and V along the sequence axis."""

    def __init__(self, d_model, n_heads, dropout=0.1, max_len: int = 8192, k_proj: int = 128):
        super().__init__(d_model, n_heads, dropout)
        self.k_proj = k_proj
        # Shared E/F projections across heads (the "headwise sharing" variant of the paper).
        self.E = nn.Parameter(torch.randn(max_len, k_proj) / math.sqrt(k_proj))
        self.F = nn.Parameter(torch.randn(max_len, k_proj) / math.sqrt(k_proj))

    def forward(self, x, mask=None):
        q, k, v = self._split(x)
        N = x.shape[1]
        if N > self.E.shape[0]:
            raise ValueError(f"Linformer max_len={self.E.shape[0]} < sequence length {N}")
        if mask is not None:
            m = mask[:, None, :, None].to(k.dtype)
            k = k * m
            v = v * m
        E, Fm = self.E[:N], self.F[:N]                          # [N, k]
        k = torch.einsum("bhnd,nk->bhkd", k, E)                 # [B,H,k,Dh]
        v = torch.einsum("bhnd,nk->bhkd", v, Fm)                # [B,H,k,Dh]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)  # [B,H,N,k]
        attn = self.dropout(torch.softmax(scores, dim=-1))
        return self._merge(torch.matmul(attn, v))


ATTENTION_TYPES = {
    "softmax": SoftmaxAttention,
    "linear": LinearAttention,
    "linformer": LinformerAttention,
}


def build_attention(kind: str, d_model: int, n_heads: int, dropout: float = 0.1, max_len: int = 8192):
    if kind not in ATTENTION_TYPES:
        raise ValueError(f"Unknown attention '{kind}'. Choose from {list(ATTENTION_TYPES)}")
    if kind == "linformer":
        return LinformerAttention(d_model, n_heads, dropout, max_len=max_len)
    return ATTENTION_TYPES[kind](d_model, n_heads, dropout)
