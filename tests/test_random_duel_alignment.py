"""Numerical alignment and state reuse for random remasking and DUEL."""
import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from test_low_confidence_alignment import pe


RANDOM = pe._path_sampling_random_probability_from_partially_masked
DUEL = pe._duel_low_confidence_probability_fast_from_partially_masked
MASK_ID = 7


class StateModel(torch.nn.Module):
    def __init__(self, bad_value=None, constant_p=None):
        super().__init__()
        self.register_buffer('anchor', torch.zeros((), dtype=torch.float32))
        self.dropout = torch.nn.Dropout(0.9)
        self.bad_value = bad_value
        self.constant_p = constant_p
        self.calls = []
        self.batch_sizes = []
        self.allow_batched = False
        self.last_logits = None

    @property
    def device(self):
        return self.anchor.device

    def forward(self, tokens, attention_mask=None):
        assert self.allow_batched or tokens.shape[0] == 1
        self.batch_sizes.append(tokens.shape[0])
        rows = [self._forward_single(
            tokens[i:i + 1], None if attention_mask is None else attention_mask[i:i + 1],
        ).logits for i in range(tokens.shape[0])]
        self.last_logits = torch.cat(rows, dim=0)
        return SimpleNamespace(logits=self.last_logits)

    def _forward_single(self, tokens, attention_mask=None):
        assert tokens.shape == (1, 100)
        assert not self.training and not self.dropout.training
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.backends.cudnn.benchmark
        assert not torch.is_autocast_enabled('cpu')
        if attention_mask is not None:
            assert attention_mask.shape == (1, 100)
        state = tuple(i for i in range(50) if tokens[0, i] != MASK_ID)
        assert torch.equal(tokens[0, 50:], torch.zeros(50, dtype=torch.long))
        self.calls.append(state)
        if self.constant_p is None:
            # Nontrivial state dependence, with exact confidence ties between
            # neighboring positions. Target token 0 is sometimes outside top-1.
            logits = torch.zeros((1, 100, 3), dtype=torch.float32)
            for i in range(50):
                logits[0, i] = torch.tensor([
                    0.5 + (i // 2) % 3 + (0.25 if 0 in state else 0),
                    1.0 + (len(state) % 4) / 4,
                    -0.5,
                ])
        else:
            p = self.constant_p
            row = torch.tensor([
                math.log(p) if p else -math.inf,
                math.log1p(-p) if p < 1 else -math.inf,
                -math.inf,
            ], dtype=torch.float64)
            logits = row.expand(1, 100, -1).clone()
        if self.bad_value is not None:
            logits[..., 0] = self.bad_value
        self.last_logits = logits
        return SimpleNamespace(logits=logits)


class RandomDuelAlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def args(self, model, temperature=1.0):
        return dict(
            model=model, sequence_tokens=torch.zeros((1, 100), dtype=torch.long),
            masked_indexes=list(range(50, 0, -1)), steps=50,
            attention_mask=torch.ones((1, 100), dtype=torch.long),
            mask_id=MASK_ID, temperature=temperature,
        )

    def random_args(self, model, temperature=1.0):
        model.allow_batched = True
        return dict(self.args(model, temperature), num_samples=8, seed=51,
                    decoding_scheme='full', k=1)

    def test_duel_confidence_ranking_matches_sts_and_target_scores_remain_fp64(self):
        original = pe._duel_state_scores
        for temperature in (0.5, 1.0, 2.0):
            for verbose in (False, True):
                model = StateModel()

                def observe(model, x_row, positions, target_ids, attention_mask,
                            temperature, context, *args, **kwargs):
                    result = original(model, x_row, positions, target_ids,
                                      attention_mask, temperature, context, *args, **kwargs)
                    logits = model.last_logits[0, positions, :].contiguous()
                    distribution = pe._low_confidence_distribution(logits, temperature, context)
                    expected = distribution.logits.gather(-1, target_ids[:, None])[:, 0]
                    expected = expected / temperature - distribution.log_Z_sample
                    expected_conf = distribution.logits.max(dim=-1).values - distribution.log_Z_conf
                    self.assertEqual(result[0], expected_conf.argmax().item())
                    self.assertTrue(torch.equal(result[2], expected_conf))
                    self.assertAlmostEqual(result[1].item(), expected[result[0]].item(), places=13)
                    if result[3] is not None:
                        torch.testing.assert_close(result[3], expected, rtol=0, atol=2e-14)
                    return result

                with patch.object(pe, '_duel_state_scores', side_effect=observe) as calls:
                    DUEL(**self.args(model, temperature), verbose=verbose)
                self.assertEqual(calls.call_count, 50)

    def test_random_scores_match_across_batches_for_a_batch_independent_model(self):
        results = []
        for batch_size in (1, 3, 512):
            model = StateModel()
            args = self.random_args(model)
            args.update(steps=7, num_samples=13, batch_size=batch_size)
            result = RANDOM(**args)
            self.assertEqual(result['model_forward_calls'], len(model.batch_sizes))
            self.assertEqual(result['model_forward_rows'], len(model.calls))
            self.assertLessEqual(max(model.batch_sizes), batch_size)
            results.append(result)
        for result in results[1:]:
            for actual, reference in zip(result['sample_log_probabilities'], results[0]['sample_log_probabilities']):
                self.assertAlmostEqual(actual, reference, places=11)
            self.assertAlmostEqual(result['log_probability'], results[0]['log_probability'], places=11)

    def test_native_bfloat16_and_float16_logits_match_sts_after_widening(self):
        class NativeModel(StateModel):
            def forward(self, tokens, attention_mask=None):
                result = super().forward(tokens, attention_mask)
                result.logits = result.logits.to(self.output_dtype)
                self.last_logits = result.logits
                return result

        @pe._low_confidence_eval_mode
        def compare(model, temperature):
            x_row = torch.zeros(100, dtype=torch.long)
            x_row[:50] = MASK_ID
            positions = torch.arange(50)
            ids = torch.zeros(50, dtype=torch.long)
            chosen, actual, confidence, all_targets = pe._duel_state_scores(
                model, x_row, positions, ids, None, temperature, 'native logits',
                full_diagnostics=True,
            )
            distribution = pe._low_confidence_distribution(
                model.last_logits[0, positions, :].contiguous(), temperature, 'native logits',
            )
            expected = distribution.logits[:, 0] / temperature - distribution.log_Z_sample
            expected_conf = distribution.logits.max(dim=-1).values - distribution.log_Z_conf
            self.assertEqual(chosen, expected_conf.argmax().item())
            self.assertAlmostEqual(actual.item(), expected[chosen].item(), places=13)
            torch.testing.assert_close(all_targets, expected, rtol=0, atol=2e-14)
            self.assertTrue(torch.equal(confidence, expected_conf))

        for dtype in (torch.bfloat16, torch.float16):
            for temperature in (0.5, 1.0, 2.0):
                model = NativeModel()
                model.output_dtype = dtype
                compare(model, temperature)

    def test_random_schedule_scores_every_block_before_revealing_any_of_it(self):
        for scheme, k in (('full', 1), ('top_k', 2), ('top_k', 3), ('top_k', 99)):
            for temperature in (0.5, 2.0):
                args = self.random_args(StateModel(), temperature)
                args.update(steps=7, num_samples=5, decoding_scheme=scheme, k=k)
                result = RANDOM(**args)
                generator = torch.Generator().manual_seed(51)
                permutations = torch.rand((5, 50), dtype=torch.float64, generator=generator).argsort(dim=-1).tolist()
                expected_logs = []
                for permutation in permutations:
                    state, log_weight, start = set(), 0.0, 0
                    for size in [8] + [7] * 6:
                        token_logs = []
                        for slot in permutation[start:start + size]:
                            logits = [0.5 + (slot // 2) % 3 + (0.25 if 0 in state else 0),
                                      1.0 + (len(state) % 4) / 4, -0.5]
                            support = sorted(range(3), key=lambda i: logits[i], reverse=True)
                            if scheme == 'top_k':
                                support = support[:k]
                            if 0 not in support:
                                token_logs.append(-math.inf)
                            else:
                                denominator = math.fsum(math.exp(logits[i] / temperature) for i in support)
                                token_logs.append(logits[0] / temperature - math.log(denominator))
                        log_weight += math.fsum(token_logs)
                        state.update(permutation[start:start + size])
                        start += size
                    expected_logs.append(log_weight)
                for actual, expected in zip(result['sample_log_probabilities'], expected_logs):
                    if expected == -math.inf:
                        self.assertEqual(actual, expected)
                    else:
                        self.assertAlmostEqual(actual, expected, places=11)

    def test_default_500_samples_reuse_initial_state_and_do_not_sort_vocab(self):
        args = self.random_args(StateModel(constant_p=0.01))
        args.update(num_samples=500, steps=1)
        with patch.object(torch, 'sort', side_effect=AssertionError('Unused vocabulary sort')):
            result = RANDOM(**args)
        self.assertEqual(result['trajectory_batch_size'], 128)
        self.assertEqual(result['model_forward_calls'], 1)
        self.assertTrue(result['initial_state_reused'])
        self.assertAlmostEqual(result['log_probability'], math.log(1e-100), places=11)
        self.assertLess(abs(result['probability'] / 1e-100 - 1), 1e-11)

    def test_duel_path_and_score_match_independent_greedy_confidence_oracle(self):
        for temperature in (0.5, 1.0, 2.0):
            model = StateModel()
            with patch.object(torch, 'sort', side_effect=AssertionError('Unused vocabulary sort')):
                result = DUEL(**self.args(model, temperature), verbose=True)
            state, path, log_weight = set(), [], 0.0
            for _ in range(50):
                rows = {}
                for i in range(50):
                    if i not in state:
                        logits = [0.5 + (i // 2) % 3 + (0.25 if 0 in state else 0),
                                  1.0 + (len(state) % 4) / 4, -0.5]
                        confidence = max(logits) - math.log(math.fsum(math.exp(v) for v in logits))
                        log_target = logits[0] / temperature - math.log(math.fsum(math.exp(v / temperature) for v in logits))
                        rows[i] = confidence, log_target
                chosen = max(rows, key=lambda i: (rows[i][0], -i))
                log_weight += rows[chosen][1]
                path.append(chosen + 1)
                state.add(chosen)
            self.assertEqual(result['reveal_path_indices'], path)
            self.assertAlmostEqual(result['log_probability'], log_weight, places=12)
            self.assertEqual(result['model_forward_calls'], 50)
            self.assertEqual(len(model.calls), 50)
            compact = DUEL(**self.args(StateModel(), temperature), verbose=True, verbose_compact=True)
            self.assertEqual(result['log_probability'], compact['log_probability'])
            self.assertEqual(path, compact['reveal_path_indices'])

    def test_duel_tiny_probability_and_exact_ties(self):
        result = DUEL(**self.args(StateModel(constant_p=0.01)))
        self.assertEqual(result['reveal_path_indices'], list(range(1, 51)))
        self.assertAlmostEqual(result['log_probability'], math.log(1e-100), places=11)
        self.assertLess(abs(result['probability'] / 1e-100 - 1), 1e-11)

    def test_duel_normalizes_only_chosen_tempered_rows_and_diagnostics_do_not_change_scores(self):
        for temperature in (0.5, 1.0, 2.0):
            original = torch.log_softmax
            shapes = []

            def observe(values, *args, **kwargs):
                shapes.append(tuple(values.shape))
                self.assertEqual(values.dtype, torch.float64)
                return original(values, *args, **kwargs)

            with patch.object(torch, 'log_softmax', side_effect=observe):
                quiet = DUEL(**self.args(StateModel(), temperature))
            if temperature == 1.0:
                self.assertEqual(shapes, [])  # Reuses the ranking normalizer.
            else:
                self.assertEqual(shapes, [(1, 3)] * 50)
            for compact in (False, True):
                verbose = DUEL(**self.args(StateModel(), temperature),
                               verbose=True, verbose_compact=compact)
                self.assertEqual(verbose['log_probability'], quiet['log_probability'])
                self.assertEqual(verbose['reveal_path_indices'], quiet['reveal_path_indices'])

    def test_duel_tempered_tiny_probabilities_stay_in_log_space(self):
        for temperature in (0.1, 0.5, 2.0):
            p = 0.01
            logits = [math.log(p) / temperature, math.log1p(-p) / temperature]
            largest = max(logits)
            expected_log = 50 * (logits[0] - largest - math.log(
                math.fsum(math.exp(v - largest) for v in logits)))
            result = DUEL(**self.args(StateModel(constant_p=p), temperature))
            self.assertAlmostEqual(result['log_probability'], expected_log, places=10)
            self.assertEqual(result['reveal_path_indices'], list(range(1, 51)))
            self.assertEqual(result['model_forward_calls'], 50)
            self.assertTrue(math.isfinite(result['log_probability']))
            if temperature == 0.1:
                self.assertEqual(result['probability'], 0.0)

    def test_zero_paths_and_topk_exclusion_are_valid(self):
        for scheme in ('full', 'top_k'):
            model = StateModel(constant_p=0.0 if scheme == 'full' else 0.01)
            result = RANDOM(**dict(self.random_args(model), decoding_scheme=scheme))
            self.assertEqual(result['probability'], 0.0)
            self.assertEqual(result['log_probability'], -math.inf)
            self.assertEqual(result['model_forward_calls'], 1)
        duel = DUEL(**self.args(StateModel(constant_p=0.0)))
        self.assertEqual(duel['probability'], 0.0)
        self.assertEqual(duel['log_probability'], -math.inf)
        self.assertEqual(len(duel['reveal_path_indices']), 50)

    def test_eval_and_backend_modes_restore_after_success_and_invalid_logits(self):
        def settings():
            return (torch.are_deterministic_algorithms_enabled(),
                    torch.is_deterministic_algorithms_warn_only_enabled(),
                    torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic)
        original = settings()
        try:
            torch.use_deterministic_algorithms(False, warn_only=True)
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            before = settings()
            for estimator in (RANDOM, DUEL):
                for value in (None, math.nan, math.inf):
                    model = StateModel(bad_value=value).train()
                    model.dropout.eval()
                    args = self.random_args(model) if estimator is RANDOM else self.args(model)
                    if estimator is RANDOM:
                        args.update(num_samples=2, steps=1)
                    with torch.autocast('cpu', dtype=torch.bfloat16):
                        if value is None:
                            estimator(**args)
                        else:
                            with self.assertRaisesRegex(FloatingPointError, 'step=0, revealed_indices='):
                                estimator(**args)
                    self.assertTrue(model.training)
                    self.assertFalse(model.dropout.training)
                    self.assertEqual(settings(), before)
        finally:
            torch.use_deterministic_algorithms(original[0], warn_only=original[1])
            torch.backends.cudnn.benchmark = original[2]
            torch.backends.cudnn.deterministic = original[3]

    def test_duplicate_masks_rejected_instead_of_silently_deduplicated(self):
        for estimator in (RANDOM, DUEL):
            args = self.random_args(StateModel()) if estimator is RANDOM else self.args(StateModel())
            args['masked_indexes'].append(1)
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                estimator(**args)

    def test_random_public_dispatch_preserves_log_scores_and_forward_metadata(self):
        result = pe.compute_diffusion_probabilistic_extraction(
            model=StateModel(constant_p=0.01),
            prompt_tokens=torch.zeros((1, 50), dtype=torch.long),
            target_tokens=torch.zeros((1, 50), dtype=torch.long),
            steps=1, mask_id=MASK_ID, remasking='random', estimation_method='path_sampling',
            num_samples=500, seed=51, temperature=1.0, masked_indexes=list(range(1, 51)),
        )
        self.assertEqual(result['model_forward_calls'], 1)
        self.assertEqual(len(result['sample_log_probabilities']), 500)
        self.assertAlmostEqual(result['log_probability'], math.log(1e-100), places=11)

    def test_random_500_paths_use_197_real_model_calls_and_preserve_tiny_weights(self):
        # A cheap vectorized model makes this a full 500 x 50-step regression.
        class ConstantBatchModel(torch.nn.Module):
            def __init__(self, probability):
                super().__init__()
                self.register_buffer('row', torch.tensor(
                    [math.log(probability), math.log1p(-probability)], dtype=torch.float64,
                ))
                self.batch_sizes = []

            @property
            def device(self):
                return self.row.device

            def forward(self, tokens, attention_mask=None):
                assert not self.training and not torch.is_autocast_enabled('cpu')
                self.batch_sizes.append(tokens.shape[0])
                return SimpleNamespace(logits=self.row.expand(tokens.shape[0], 100, 2))

        for probability in (0.01, 1e-10):
            model = ConstantBatchModel(probability)
            result = RANDOM(**dict(self.random_args(model), num_samples=500))
            self.assertEqual(result['model_forward_calls'], 197)
            self.assertEqual(len(model.batch_sizes), 197)
            self.assertEqual(max(model.batch_sizes), 128)
            self.assertEqual(result['model_forward_rows'], 24501)
            expected_log = 50 * math.log(probability)
            for log_weight in result['sample_log_probabilities']:
                self.assertAlmostEqual(log_weight, expected_log, places=10)
            self.assertAlmostEqual(result['log_probability'], expected_log, places=10)
            if probability == 0.01:
                self.assertLess(abs(result['probability'] / 1e-100 - 1), 1e-11)
            else:
                self.assertEqual(result['probability'], 0.0)
                self.assertTrue(math.isfinite(result['log_probability']))

    def test_selected_position_normalization_is_fp64_and_chunk_bounded(self):
        generator = torch.Generator().manual_seed(17)
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            logits = torch.randn((5, 100, 31), generator=generator).to(dtype)
            positions = torch.tensor([[2, 8, 13, 27]]).expand(5, -1)
            targets = torch.tensor([[0, 3, 9, 21]]).expand(5, -1)
            for temperature in (0.5, 1.0, 2.0):
                original = torch.log_softmax
                shapes = []

                def observe(values, *args, **kwargs):
                    self.assertEqual(values.dtype, torch.float64)
                    shapes.append(values.shape[0])
                    return original(values, *args, **kwargs)

                with patch.object(torch, 'log_softmax', side_effect=observe):
                    actual = pe._random_remasking_target_log_probs(
                        logits, positions, targets, temperature, 'full', 1, 'chunk test',
                        normalization_batch_size=3,
                    )
                self.assertEqual(sum(shapes), 20)
                self.assertLessEqual(max(shapes), 3)
                selected = logits[torch.arange(5)[:, None], positions].double() / temperature
                expected = selected.gather(-1, targets[..., None])[..., 0] - torch.logsumexp(selected, -1)
                torch.testing.assert_close(actual, expected, rtol=0, atol=2e-14)

    def test_nonfinite_normalizers_and_cancellation_are_rejected(self):
        # Also validate non-target logits and the distribution itself, rather
        # than silently treating malformed model output as zero probability.
        class MalformedModel(StateModel):
            def forward(self, tokens, attention_mask=None):
                result = super().forward(tokens, attention_mask)
                result.logits.fill_(self.fill_value)
                return result

        for value in (-math.inf, 1e16):
            for estimator in (RANDOM, DUEL):
                model = MalformedModel(constant_p=0.01)
                model.fill_value = value
                args = self.random_args(model) if estimator is RANDOM else self.args(model)
                if estimator is RANDOM and value == 1e16:
                    # Stable log_softmax correctly removes a common large
                    # offset rather than losing the normalization in subtraction.
                    result = estimator(**args)
                    self.assertAlmostEqual(result['log_probability'], -50 * math.log(3), places=11)
                else:
                    with self.assertRaises(FloatingPointError):
                        estimator(**args)


if __name__ == '__main__':
    unittest.main()
