"""Optional, numerically faithful fused RoPE for random-remasking LLaDA forwards.

The kernel keeps the eager FP32 products and addition separate (no FMA), then
rounds to the input dtype. No probabilities, model weights or attention context
are approximated. CPU and unsupported models retain their original forwards.
"""
from contextlib import contextmanager
from functools import partial

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


if triton is not None:
    @triton.jit
    def _rope_pair(Q, K, SIN, COS, OQ, OK,
                   N: tl.constexpr, H: tl.constexpr, T: tl.constexpr, D: tl.constexpr,
                   Q0: tl.constexpr, Q1: tl.constexpr, Q2: tl.constexpr, Q3: tl.constexpr,
                   K0: tl.constexpr, K1: tl.constexpr, K2: tl.constexpr, K3: tl.constexpr,
                   OQ0: tl.constexpr, OQ1: tl.constexpr, OQ2: tl.constexpr, OQ3: tl.constexpr,
                   OK0: tl.constexpr, OK1: tl.constexpr, OK2: tl.constexpr, OK3: tl.constexpr,
                   S0: tl.constexpr, S1: tl.constexpr, C0: tl.constexpr, C1: tl.constexpr,
                   BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = i < N
        d = i % D
        t = (i // D) % T
        h = (i // (D * T)) % H
        b = i // (D * T * H)
        rotated_d = (d + D // 2) % D
        qbase = b * Q0 + h * Q1 + t * Q2
        kbase = b * K0 + h * K1 + t * K2
        q = tl.load(Q + qbase + d * Q3, valid, other=0).to(tl.float32)
        k = tl.load(K + kbase + d * K3, valid, other=0).to(tl.float32)
        qr = tl.load(Q + qbase + rotated_d * Q3, valid, other=0).to(tl.float32)
        kr = tl.load(K + kbase + rotated_d * K3, valid, other=0).to(tl.float32)
        qr = tl.where(d < D // 2, -qr, qr)
        kr = tl.where(d < D // 2, -kr, kr)
        s = tl.load(SIN + t * S0 + d * S1, valid, other=0).to(tl.float32)
        c = tl.load(COS + t * C0 + d * C1, valid, other=0).to(tl.float32)
        # Disable FMA at launch: eager PyTorch rounds both FP32 products before add.
        qo = q * c + qr * s
        ko = k * c + kr * s
        tl.store(OQ + b * OQ0 + h * OQ1 + t * OQ2 + d * OQ3, qo, valid)
        tl.store(OK + b * OK0 + h * OK1 + t * OK2 + d * OK3, ko, valid)


def fused_rope(q, k, sin, cos):
    oq, ok = torch.empty_like(q), torch.empty_like(k)
    b, h, t, d = q.shape
    _rope_pair[(triton.cdiv(q.numel(), 512),)](
        q, k, sin, cos, oq, ok, q.numel(), h, t, d,
        *q.stride(), *k.stride(), *oq.stride(), *ok.stride(),
        sin.stride(-2), sin.stride(-1), cos.stride(-2), cos.stride(-1),
        BLOCK=512, num_warps=4, enable_fp_fusion=False,
    )
    return oq, ok


def _rotary_forward(module, original, q, k):
    if (q.device.type != 'cuda' or q.device != k.device or q.ndim != 4
            or q.shape != k.shape or q.shape[-1] % 2 or q.numel() == 0
            or q.dtype not in (torch.bfloat16, torch.float16) or k.dtype != q.dtype
            or not module.config.rope_full_precision):
        return original(q, k)
    sin, cos = module.get_rotary_embedding(k.shape[-2], q.device)
    if (sin.shape != (1, 1, k.shape[-2], k.shape[-1]) or cos.shape != sin.shape
            or sin.device != q.device or cos.device != q.device):
        return original(q, k)
    return fused_rope(q, k, sin.float(), cos.float())


@contextmanager
def _rotary_context(modules):
    saved = []
    try:
        for module in modules:
            had_override = 'forward' in module.__dict__
            saved.append((module, had_override, module.__dict__.get('forward')))
            original = module.forward
            module.forward = partial(_rotary_forward, module, original)
        yield
    finally:
        for module, had_override, previous in reversed(saved):
            if had_override:
                module.forward = previous
            else:
                del module.forward


def get_rotary_context(model, device):
    """Return a scoped forward context only for the inspected LLaDA layout."""
    if triton is None or device.type != 'cuda':
        return None
    if torch.cuda.get_device_capability(device)[0] < 8:
        return None
    if next(model.parameters()).dtype not in (torch.bfloat16, torch.float16):
        return None
    core = getattr(model, 'model', None)
    if (type(core).__name__ != 'LLaDAModel'
            or getattr(getattr(model, 'config', None), 'model_type', None) != 'llada'):
        return None
    blocks = getattr(getattr(core, 'transformer', None), 'blocks', None)
    if blocks is None:
        return None
    modules = tuple(getattr(block, 'rotary_emb', None) for block in blocks)
    if not modules or any(
        type(module).__name__ != 'RotaryEmbedding'
        or not getattr(getattr(module, 'config', None), 'rope_full_precision', False)
        or 'forward' in module.__dict__
        for module in modules
    ):
        return None
    return partial(_rotary_context, modules)
