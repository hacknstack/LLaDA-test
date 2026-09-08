"""CPU regression tests; run with python -B -m unittest discover -s tests -v.

Only torch is required. Unrelated ELBO/generation imports are isolated so these
tests do not require transformers, model downloads, or a GPU.
"""
import importlib.util
import math
from functools import lru_cache
from itertools import product
from pathlib import Path
import statistics
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch


def load_extraction_module():
    likelihood = ModuleType('get_log_likelihood')
    generation = ModuleType('generate')

    def unused(*args, **kwargs):
        raise AssertionError('An unrelated estimator dependency was called.')

    likelihood.get_log_likelihood = unused
    likelihood.get_log_likelihood_from_partially_masked = unused
    generation.add_gumbel_noise = unused
    spec = importlib.util.spec_from_file_location(
        '_extraction_alignment_test',
        Path(__file__).resolve().parents[1] / 'probabilistic_extraction.py',
    )
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {
        'get_log_likelihood': likelihood, 'generate': generation,
        spec.name: module,
    }):
        spec.loader.exec_module(module)
    return module


pe = load_extraction_module()
STS = pe._path_sampling_low_confidence_probability_fast_from_partially_masked
MC = pe._monte_carlo_probability_temperature_fast_from_partially_masked
POSITIONS = (0, 49, 99)
MASK_ID = 3


def toy_logits(state, slot):
    # Two identical rows create genuine confidence ties. Revealing positions
    # changes subsequent predictions, so choosing a wrong order changes p_z.
    if slot < 2:
        return (2.0 + len(state), 1.0, 0.0)
    return (1.0 + (2.0 if 0 in state else 0.0), 2.0, 0.0)


def enumerate_success_masses(state, temperature):
    """Independent oracle: enumerate ALL candidate vectors, then select winner."""
    remaining = [i for i in range(3) if i not in state]
    probabilities, confidence = {}, {}
    for i in remaining:
        row = toy_logits(state, i)
        weights = [math.exp(v / temperature) for v in row]
        probabilities[i] = [v / math.fsum(weights) for v in weights]
        norm = math.log(math.fsum(math.exp(v) for v in row))
        confidence[i] = [v - norm for v in row]
    masses = {i: 0.0 for i in remaining}
    for tokens in product(range(3), repeat=len(remaining)):
        candidates = dict(zip(remaining, tokens))
        winner = max(remaining, key=lambda i: (confidence[i][candidates[i]], -i))
        if candidates[winner] == 0:
            masses[winner] += math.prod(
                probabilities[i][candidates[i]] for i in remaining
            )
    return masses


def exact_probability(temperature):
    @lru_cache(None)
    def visit(state):
        if len(state) == 3:
            return 1.0
        return math.fsum(
            mass * visit(tuple(sorted((*state, slot))))
            for slot, mass in enumerate_success_masses(state, temperature).items()
        )
    return visit(())


class ToyModel(torch.nn.Module):
    def __init__(self, bad_value=None, zero_target=False):
        super().__init__()
        self.register_buffer('anchor', torch.zeros((), dtype=torch.float32))
        self.dropout = torch.nn.Dropout(0.9)
        self.bad_value = bad_value
        self.zero_target = zero_target
        self.calls = []

    @property
    def device(self):
        return self.anchor.device

    def forward(self, tokens, attention_mask=None):
        # Fail immediately if either estimator regresses to batched or training
        # forwards. These assertions cover diagnostics and uncached paths too.
        assert tokens.shape == (1, 100), tokens.shape
        assert not self.training and not self.dropout.training
        assert not torch.is_autocast_enabled('cpu')
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.backends.cudnn.benchmark
        if attention_mask is not None:
            assert attention_mask.shape == (1, 100)
        state = tuple(i for i, pos in enumerate(POSITIONS) if tokens[0, pos] != MASK_ID)
        self.calls.append(state)
        logits = torch.zeros((1, 100, 3), dtype=torch.float32)
        for i, pos in enumerate(POSITIONS):
            logits[0, pos] = torch.tensor(toy_logits(state, i))
        if self.zero_target:
            logits[..., 0] = float('-inf')
        if self.bad_value is not None:
            logits[..., 0] = self.bad_value
        return SimpleNamespace(logits=logits)


class LowConfidenceAlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def args(self, model, temperature=1.0, samples=64):
        return dict(
            model=model, sequence_tokens=torch.zeros((1, 100), dtype=torch.long),
            masked_indexes=[100, 1, 50], steps=3,
            attention_mask=torch.ones((1, 100), dtype=torch.long),
            mask_id=MASK_ID, num_samples=samples, seed=1729,
            temperature=temperature,
        )

    def test_both_estimators_agree_with_exhaustive_decoder(self):
        for temperature in (0.5, 1.0, 2.0):
            with self.subTest(temperature=temperature):
                expected = exact_probability(temperature)
                sts = STS(**self.args(ToyModel(), temperature, 1024), batch_size=31)
                mc = MC(**self.args(ToyModel(), temperature, 12000),
                        decoding_scheme='full', k=1, mc_batch_size=701,
                        model_batch_size=17)
                sts_se = statistics.stdev(sts['sample_probabilities']) / math.sqrt(1024)
                mc_se = math.sqrt(expected * (1 - expected) / 12000)
                self.assertLess(abs(sts['probability'] - expected), 6 * sts_se + 1e-12)
                self.assertLess(abs(mc.estimate - expected), 6 * mc_se)

    def test_every_logged_sts_transition_matches_enumerated_candidates(self):
        for temperature in (0.5, 1.0, 2.0):
            result = STS(**self.args(ToyModel(), temperature), verbose=True,
                         verbose_compact=True, batch_size=13)
            for sample in result['verbose_samples']:
                state = ()
                for record, position in zip(sample['steps'], sample['reveal_path_indices']):
                    masses = enumerate_success_masses(state, temperature)
                    self.assertAlmostEqual(math.exp(record['log_A']), sum(masses.values()), places=12)
                    for index, log_a in zip(record['sequence_indices'], record['log_a_active']):
                        self.assertAlmostEqual(math.exp(log_a), masses[POSITIONS.index(index - 1)], places=12)
                    state = tuple(sorted((*state, POSITIONS.index(position - 1))))

    def test_cache_and_verbose_do_not_change_results(self):
        for estimator in (STS, MC):
            with self.subTest(estimator=estimator.__name__):
                extra = ({'batch_size': 11} if estimator is STS else
                         {'decoding_scheme': 'full', 'k': 1, 'mc_batch_size': 29})
                model = ToyModel()
                baseline = estimator(**self.args(model), **extra)
                # Bounded logits caching reuses states across later batches.
                self.assertEqual(len(model.calls), len(set(model.calls)))
                for cached in (False, True):
                    for verbose in (False, True):
                        result = estimator(**self.args(ToyModel()), **extra,
                                           use_state_cache=cached, verbose=verbose,
                                           verbose_compact=verbose)
                        if estimator is STS:
                            self.assertEqual(result['sample_log_probabilities'],
                                             baseline['sample_log_probabilities'])
                        else:
                            self.assertEqual(result.hits, baseline.hits)

    def test_temporary_eval_mode_restores_mixed_modes_and_disables_autocast(self):
        for estimator in (STS, MC):
            model = ToyModel().train()
            model.dropout.eval()
            extra = {} if estimator is STS else {'decoding_scheme': 'full', 'k': 1}
            with torch.autocast('cpu', dtype=torch.bfloat16):
                estimator(**self.args(model, samples=4), **extra)
            self.assertTrue(model.training)
            self.assertFalse(model.dropout.training)

    def test_invalid_logits_raise_with_context_and_restore_model_mode(self):
        for value in (float('nan'), float('inf')):
            for estimator in (STS, MC):
                model = ToyModel(bad_value=value).train()
                extra = {} if estimator is STS else {'decoding_scheme': 'full', 'k': 1}
                with self.assertRaisesRegex(FloatingPointError, 'step=0, revealed_indices='):
                    estimator(**self.args(model, samples=2), **extra)
                self.assertTrue(model.training)
                self.assertTrue(model.dropout.training)

    def test_zero_success_mass_is_valid_with_or_without_cache(self):
        for cached in (False, True):
            model = ToyModel(zero_target=True)
            sts = STS(**self.args(model, samples=5),
                      use_state_cache=cached, verbose=True)
            mc = MC(**self.args(ToyModel(zero_target=True), samples=5),
                    decoding_scheme='full', k=1, use_state_cache=cached)
            self.assertEqual(sts['probability'], 0.0)
            self.assertEqual(sts['log_probability'], float('-inf'))
            self.assertEqual(mc.hits, 0)
            self.assertTrue(all(state == () for state in model.calls))

    def test_backend_settings_restored_after_success_and_failure(self):
        def settings():
            return (torch.are_deterministic_algorithms_enabled(),
                    torch.is_deterministic_algorithms_warn_only_enabled(),
                    torch.backends.cudnn.benchmark,
                    torch.backends.cudnn.deterministic)

        original = settings()
        try:
            torch.use_deterministic_algorithms(False, warn_only=True)
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            before = settings()
            for estimator in (STS, MC):
                extra = {} if estimator is STS else {'decoding_scheme': 'full', 'k': 1}
                estimator(**self.args(ToyModel(), samples=2), **extra)
                self.assertEqual(settings(), before)
                with self.assertRaises(FloatingPointError):
                    estimator(**self.args(ToyModel(bad_value=float('nan')), samples=2), **extra)
                self.assertEqual(settings(), before)
        finally:
            torch.use_deterministic_algorithms(original[0], warn_only=original[1])
            torch.backends.cudnn.benchmark = original[2]
            torch.backends.cudnn.deterministic = original[3]

    def test_sts_rejects_total_success_mass_above_one(self):
        # Simulate corruption after distribution validation. Each individual
        # a_i stays <= 1, but their sum exceeds one and must never be returned.
        original = pe._low_confidence_distribution

        def corrupted(*args, **kwargs):
            distribution = original(*args, **kwargs)
            distribution.log_cdf.fill_(0.0)
            return distribution

        with patch.object(pe, '_low_confidence_distribution', side_effect=corrupted):
            with self.assertRaisesRegex(FloatingPointError, r'Invalid A\(S\).*step=0'):
                STS(**self.args(ToyModel(), samples=2))

    def test_single_position_and_streamed_diagnostics(self):
        args = self.args(ToyModel(), samples=5000)
        args.update(masked_indexes=[100], steps=1)
        expected = math.exp(3) / (math.exp(3) + math.exp(2) + 1)
        sts = STS(**args, return_samples=False)
        self.assertAlmostEqual(sts['probability'], expected, places=12)
        self.assertIsNone(sts['sample_probabilities'])
        streamed = []
        mc = MC(**args, decoding_scheme='full', k=1, verbose=True,
                verbose_callback=streamed.extend)
        self.assertIsNone(mc.verbose_samples)
        self.assertEqual(len(streamed), 5000)
        self.assertEqual(sum(s['is_hit'] for s in streamed), mc.hits)

    def test_log_mass_guard_rejects_invalid_A_but_allows_zero(self):
        for value in (float('nan'), float('inf'), math.log(1.01)):
            with self.assertRaisesRegex(FloatingPointError, r'Invalid A\(S\)'):
                pe._check_low_confidence_log_mass(torch.tensor([value]), 'A(S)', 'step=2')
        pe._check_low_confidence_log_mass(
            torch.tensor([float('-inf'), 0.0, 1e-12], dtype=torch.float64), 'A(S)', 'step=2',
        )

    def test_invalid_distribution_normalizers_are_rejected(self):
        for logits, temperature in (
            (torch.full((1, 3), float('-inf')), 1.0),
            (torch.ones((1, 3)), 1e-320),
            (torch.full((1, 3), 1e16, dtype=torch.float64), 1.0),
        ):
            with self.assertRaises(FloatingPointError):
                pe._low_confidence_distribution(logits, temperature, 'step=0')

    def test_logits_cache_is_bounded_and_eviction_preserves_values(self):
        # This unit test invokes the evaluator directly, outside its public
        # estimator decorator, so provide the same execution context explicitly.
        model = ToyModel().eval()
        positions = torch.tensor(POSITIONS)
        evaluator = pe._LowConfidenceStateEvaluator(
            model, None, positions, cache_max_bytes=36,
        )
        tokens = torch.zeros(100, dtype=torch.long)
        tokens[positions] = MASK_ID
        revealed = torch.zeros(3, dtype=torch.bool)
        @pe._low_confidence_eval_mode
        def check_cache(model):
            first = evaluator.distribution(tokens, revealed, 1.0)
            later_tokens, later_revealed = tokens.clone(), revealed.clone()
            later_tokens[0], later_revealed[0] = 0, True
            evaluator.distribution(later_tokens, later_revealed, 1.0)
            self.assertLessEqual(evaluator.cache_bytes, 36)
            repeated = evaluator.distribution(tokens, revealed, 1.0)
            self.assertTrue(torch.equal(first.log_cdf, repeated.log_cdf))
            self.assertEqual(evaluator.forward_rows, 3)

        check_cache(model)


if __name__ == '__main__':
    unittest.main()
