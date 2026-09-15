"""LLaDA final-norm selection preserves context, scores, and model state."""
import copy
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from test_low_confidence_alignment import pe


class LLaDAModel(torch.nn.Module):
    """Small instance of the inspected LLaDA terminal projection layout."""
    def __init__(self, tied=False, scaled=False):
        super().__init__()
        self.config = SimpleNamespace(weight_tying=tied, scale_logits=scaled, d_model=8)
        self.transformer = torch.nn.ModuleDict({
            'wte': torch.nn.Embedding(11, 8),
            'ln_f': torch.nn.LayerNorm(8),
            'ff_out': torch.nn.Linear(8, 11),
        })
        self.context_shapes = []
        self.projected_shapes = []
        self.fail_after_norm = False

    def forward(self, input_ids, attention_mask=None):
        x = self.transformer.wte(input_ids)
        self.context_shapes.append(tuple(x.shape))
        # Bidirectional mixing must use ALL 100 positions before selection.
        if attention_mask is None:
            context = x.mean(dim=1, keepdim=True)
        else:
            weights = attention_mask[:, :, None].to(x.dtype)
            context = (x * weights).sum(dim=1, keepdim=True) / weights.sum(dim=1, keepdim=True)
        x = self.transformer.ln_f(x + context)
        if self.fail_after_norm:
            raise RuntimeError('injected head failure')
        self.projected_shapes.append(tuple(x.shape))
        logits = (F.linear(x, self.transformer.wte.weight) if self.config.weight_tying else
                  self.transformer.ff_out(x))
        if self.config.scale_logits:
            logits = logits * (1 / math.sqrt(self.config.d_model))
        return SimpleNamespace(logits=logits)


class TinyLLaDA(torch.nn.Module):
    def __init__(self, tied=False, scaled=False, constant_p=None):
        super().__init__()
        self.config = SimpleNamespace(model_type='llada')
        with torch.random.fork_rng():
            torch.manual_seed(73)
            self.model = LLaDAModel(tied=tied, scaled=scaled).double()
        if constant_p is not None:
            with torch.no_grad():
                self.model.transformer.ff_out.weight.zero_()
                bias = self.model.transformer.ff_out.bias
                bias.fill_(-math.inf)
                bias[0] = math.log(constant_p)
                bias[1] = math.log1p(-constant_p)

    @property
    def device(self):
        return self.model.transformer.wte.weight.device

    def forward(self, input_ids, attention_mask=None):
        return self.model.forward(input_ids, attention_mask=attention_mask)


class RandomSelectedProjectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def args(self, model):
        return dict(model=model, sequence_tokens=torch.zeros((1, 100), dtype=torch.long),
                    masked_indexes=list(range(1, 51)), steps=50, attention_mask=None,
                    mask_id=7, num_samples=500, seed=51, decoding_scheme='full', k=2,
                    temperature=1.0, max_path_samples=None, stratified_paths=False,
                    max_path_steps=None)

    def test_selected_logits_match_full_context_projection_for_tied_and_scaled_heads(self):
        generator = torch.Generator().manual_seed(7)
        tokens = torch.randint(0, 11, (3, 100), generator=generator)
        positions = torch.tensor([[29, 0, 88], [4, 99, 17], [12, 51, 22]])
        attention = torch.ones((3, 100), dtype=torch.long)
        attention[:, -1] = 0
        for tied in (False, True):
            for scaled in (False, True):
                model = TinyLLaDA(tied=tied, scaled=scaled).eval()
                original = model(tokens, attention_mask=attention).logits
                expected = original.gather(1, positions[:, :, None].expand(-1, -1, 11))
                layer = pe._random_remasking_projection_layer(model)
                actual = pe._random_remasking_forward_logits(model, tokens, attention, positions, layer)
                torch.testing.assert_close(actual, expected, rtol=0, atol=2e-14)
                self.assertEqual(model.model.context_shapes[-1], (3, 100, 8))
                self.assertEqual(model.model.projected_shapes[-1], (3, 3, 8))
                self.assertEqual(len(layer._forward_hooks), 0)
                self.assertEqual(model(tokens).logits.shape, (3, 100, 11))

    def test_500_samples_use_50_calls_and_keep_1e_minus_100_accuracy(self):
        model = TinyLLaDA(constant_p=0.01)
        result = pe._path_sampling_random_probability_from_partially_masked(**self.args(model))
        self.assertTrue(result['selected_position_logits'])
        self.assertEqual(result['model_forward_batch_size'], 500)
        self.assertEqual(result['model_forward_max_batch_size'], 500)
        self.assertEqual(result['model_forward_calls'], 50)
        self.assertEqual(result['vocabulary_projection_rows'], 24550)
        self.assertEqual(model.model.projected_shapes[0], (1, 50, 8))
        self.assertEqual(model.model.projected_shapes[1:], [(500, 1, 8)] * 49)
        self.assertTrue(all(shape[1] == 100 for shape in model.model.context_shapes))
        self.assertAlmostEqual(result['log_probability'], math.log(1e-100), places=4)
        self.assertLess(abs(result['probability'] / 1e-100 - 1), 1e-4)
        self.assertEqual(result['token_probability_dtype'], 'float32')
        self.assertTrue(model.training)  # Caller mode restored.

    def test_full_and_selected_heads_agree_for_multitoken_blocks_and_topk(self):
        for scheme in ('full', 'top_k'):
            for temperature in (0.5, 1.0, 2.0):
                args = self.args(TinyLLaDA())
                args.update(num_samples=9, batch_size=4, steps=7,
                            masked_indexes=list(range(99, 0, -2)),
                            temperature=temperature, decoding_scheme=scheme, k=9)
                reference_args = dict(args, model=copy.deepcopy(args['model']), use_selected_logits=False)
                actual = pe._path_sampling_random_probability_from_partially_masked(**args)
                reference = pe._path_sampling_random_probability_from_partially_masked(**reference_args)
                torch.testing.assert_close(
                    torch.tensor(actual['sample_log_probabilities'], dtype=torch.float64),
                    torch.tensor(reference['sample_log_probabilities'], dtype=torch.float64),
                    rtol=0, atol=2e-6,
                )
                self.assertLess(actual['vocabulary_projection_rows'], reference['vocabulary_projection_rows'])
                self.assertEqual(len(args['model'].model.transformer.ln_f._forward_hooks), 0)

    def test_opt_out_retains_generic_batch_limit_and_full_projection(self):
        result = pe._path_sampling_random_probability_from_partially_masked(
            **dict(self.args(TinyLLaDA(constant_p=0.01)), use_selected_logits=False),
        )
        self.assertFalse(result['selected_position_logits'])
        self.assertEqual(result['model_forward_batch_size'], 128)
        self.assertEqual(result['model_forward_calls'], 197)
        self.assertEqual(result['vocabulary_projection_rows'], 2450100)

    def test_hook_and_model_modes_restore_when_forward_raises(self):
        model = TinyLLaDA().train()
        model.model.transformer.ln_f.eval()
        model.model.fail_after_norm = True
        layer = model.model.transformer.ln_f
        existing = layer.register_forward_hook(lambda module, args, output: None)
        original_hooks = list(layer._forward_hooks)
        try:
            with self.assertRaisesRegex(RuntimeError, 'injected head failure'):
                pe._path_sampling_random_probability_from_partially_masked(**self.args(model))
            self.assertEqual(list(layer._forward_hooks), original_hooks)
            self.assertTrue(model.training)
            self.assertFalse(layer.training)
            model.model.fail_after_norm = False
            self.assertEqual(model(torch.zeros((2, 100), dtype=torch.long)).logits.shape, (2, 100, 11))
        finally:
            existing.remove()

    def test_unknown_architecture_does_not_install_projection_hook(self):
        model = TinyLLaDA()
        model.config.model_type = 'other'
        self.assertIsNone(pe._random_remasking_projection_layer(model))
        args = self.args(model)
        args.update(num_samples=3, steps=1)
        result = pe._path_sampling_random_probability_from_partially_masked(**args)
        self.assertFalse(result['selected_position_logits'])
        self.assertEqual(model.model.projected_shapes, [(1, 100, 8)])

    def test_forward_does_not_mutate_workspace_fill_setting(self):
        original_fill = torch.utils.deterministic.fill_uninitialized_memory
        try:
            for caller_fill in (True, False):
                for selected in (True, False):
                    for fail in (True, False):
                        model = TinyLLaDA()
                        layer = pe._random_remasking_projection_layer(model) if selected else None
                        tokens = torch.zeros((2, 100), dtype=torch.long)
                        positions = torch.zeros((2, 1), dtype=torch.long)
                        forward = model.forward

                        def observe(*args, **kwargs):
                            self.assertEqual(torch.utils.deterministic.fill_uninitialized_memory, caller_fill)
                            if fail:
                                raise RuntimeError('injected forward failure')
                            return forward(*args, **kwargs)

                        torch.utils.deterministic.fill_uninitialized_memory = caller_fill
                        with patch.object(model, 'forward', side_effect=observe):
                            if fail:
                                with self.assertRaisesRegex(RuntimeError, 'injected forward failure'):
                                    pe._random_remasking_forward_logits(model, tokens, None, positions, layer)
                            else:
                                pe._random_remasking_forward_logits(model, tokens, None, positions, layer)
                        self.assertEqual(torch.utils.deterministic.fill_uninitialized_memory, caller_fill)
                        self.assertEqual(len(model.model.transformer.ln_f._forward_hooks), 0)
        finally:
            torch.utils.deterministic.fill_uninitialized_memory = original_fill


if __name__ == '__main__':
    unittest.main()
