"""Attention written for ONNX export to TensorRT 10.3.

TensorRT 10.3 mis-fuses the graph that torch.onnx exports from F.scaled_dot_product_attention (queries
and keys each scaled by sqrt(scale)), giving wrong results; the classic form softmax((q*scale) k^T) v
builds correctly. At full Middlebury resolution (~29k tokens) the classic form would trace a
29k x 29k matrix per head, so queries are processed in blocks. The result is identical: softmax is
taken over keys, which every block sees in full.
"""
import torch


def classic_attention(q, k, v, scale, block=2048):
    """q, k, v: (B, heads, N, d) -> (B, heads, N, d)."""
    kt = k.transpose(-2, -1)
    n = q.shape[-2]
    if n <= block:
        return ((q * scale) @ kt).softmax(dim=-1) @ v
    return torch.cat([((q[:, :, i:i + block] * scale) @ kt).softmax(dim=-1) @ v for i in range(0, n, block)], dim=2)


def patch_sdpa():
    """Route torch.nn.functional.scaled_dot_product_attention through classic_attention (export only)."""
    import math

    import torch.nn.functional as F

    def sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kwargs):
        assert attn_mask is None and not is_causal, "only plain attention is exported"
        return classic_attention(q, k, v, scale if scale is not None else 1.0 / math.sqrt(q.shape[-1]))
    F.scaled_dot_product_attention = sdpa
