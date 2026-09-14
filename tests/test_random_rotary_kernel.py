"""Exact RoPE arithmetic and restoration of scoped model overrides."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from test_low_confidence_alignment import pe
import random_remasking_kernels as kernels


class RotaryEmbedding(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(rope_full_precision=True)

    def forward(self, q, k):
        return q + 1, k + 2


class RandomRotaryTests(unittest.TestCase):
    def test_cpu_and_missing_triton_keep_native_execution(self):
        self.assertIsNone(kernels.get_rotary_context(None, torch.device('cpu')))
        with patch.object(kernels, 'triton', None):
            self.assertIsNone(kernels.get_rotary_context(None, torch.device('cuda')))

    def test_context_preserves_hooks_overrides_and_restores_after_failure(self):
        for custom in (False, True):
            module = RotaryEmbedding()
            if custom:
                module.forward = lambda q, k: (q + 3, k + 4)
            before = dict(module.__dict__)
            calls = []
            handle = module.register_forward_hook(lambda *args: calls.append(True))
            q, k = torch.zeros(3), torch.ones(3)
            expected = module(q, k)
            with self.assertRaisesRegex(RuntimeError, 'injected'):
                with kernels._rotary_context([module]):
                    actual = module(q, k)  # CPU falls back to original forward.
                    for a, b in zip(actual, expected):
                        torch.testing.assert_close(a, b, rtol=0, atol=0)
                    raise RuntimeError('injected')
            self.assertEqual(len(calls), 2)
            self.assertEqual('forward' in module.__dict__, 'forward' in before)
            if custom:
                self.assertIs(module.forward, before['forward'])
            handle.remove()

    @unittest.skipUnless(torch.cuda.is_available() and kernels.triton is not None, 'CUDA/Triton required')
    def test_fused_rotary_is_exact_for_layouts_dtypes_and_tail_tiles(self):
        gen = torch.Generator().manual_seed(1234)
        for dtype in (torch.bfloat16, torch.float16):
            for batch, heads, length, dim in ((1, 2, 7, 64), (3, 4, 100, 128)):
                for transposed in (False, True):
                    shape = (batch, length, heads, dim) if transposed else (batch, heads, length, dim)
                    q = torch.randn(shape, generator=gen).to(device='cuda', dtype=dtype)
                    k = torch.randn(shape, generator=gen).to(device='cuda', dtype=dtype)
                    if transposed:
                        q, k = q.transpose(1, 2), k.transpose(1, 2)
                    angles = torch.randn((1, 1, length, dim), generator=gen).to('cuda')
                    sin, cos = angles.sin(), angles.cos()
                    def reference(x):
                        xf = x.float()
                        half = dim // 2
                        rotated = torch.cat((-xf[..., half:], xf[..., :half]), dim=-1)
                        return (xf * cos + rotated * sin).to(dtype)
                    actual = kernels.fused_rope(q, k, sin, cos)
                    for a, b in zip(actual, (reference(q), reference(k))):
                        self.assertTrue(torch.equal(a, b))

    @unittest.skipUnless(torch.cuda.is_available() and kernels.triton is not None, 'CUDA/Triton required')
    def test_fused_rotary_preserves_small_finite_values(self):
        for dtype in (torch.bfloat16, torch.float16):
            q = torch.linspace(-1, 1, 128, device='cuda').to(dtype).view(1, 1, 1, 128)
            q = q * torch.finfo(dtype).tiny
            k = -q
            angle = torch.arange(128, device='cuda', dtype=torch.float32).view(1, 1, 1, 128)
            sin, cos = angle.sin(), angle.cos()
            for x, actual in zip((q, k), kernels.fused_rope(q, k, sin, cos)):
                xf = x.float()
                rotated = torch.cat((-xf[..., 64:], xf[..., :64]), dim=-1)
                expected = (xf * cos + rotated * sin).to(dtype)
                self.assertTrue(torch.equal(actual, expected))


if __name__ == '__main__':
    unittest.main()
