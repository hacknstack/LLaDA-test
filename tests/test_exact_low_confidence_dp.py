"""Exact DP alignment, execution policy, and small-probability regressions."""
import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from test_low_confidence_alignment import (
    MASK_ID, POSITIONS, STS, ToyModel, enumerate_success_masses,
    exact_probability, pe,
)


DP = pe._exact_low_confidence_probability_dp_from_partially_masked


class ConstantLogitsModel(torch.nn.Module):
    def __init__(self, target_probability):
        super().__init__()
        p = target_probability
        self.register_buffer('row', torch.tensor([
            math.log(p) if p else -math.inf,
            math.log1p(-p) if p < 1 else -math.inf,
        ], dtype=torch.float64))
        self.calls = 0

    @property
    def device(self):
        return self.row.device

    def forward(self, tokens, attention_mask=None):
        assert tokens.shape == (1, 100)
        assert not self.training
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.is_autocast_enabled('cpu')
        self.calls += 1
        return SimpleNamespace(logits=self.row.expand(1, 100, -1).clone())


class ExactLowConfidenceDPTests(unittest.TestCase):
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
            masked_indexes=[100, 1, 50], steps=3,
            attention_mask=torch.ones((1, 100), dtype=torch.long),
            mask_id=MASK_ID, temperature=temperature,
        )

    def test_all_state_masses_and_total_match_exhaustive_decoder(self):
        original = pe._exact_low_conf_log_a
        for temperature in (0.5, 1.0, 2.0):
            with self.subTest(temperature=temperature):
                model = ToyModel()
                states = {}

                def observe(*args, **kwargs):
                    values = original(*args, **kwargs)
                    state = model.calls[-1]
                    states[state] = values[0].tolist()
                    expected = enumerate_success_masses(state, temperature)
                    for log_mass, mass in zip(states[state], expected.values()):
                        if mass == 0:
                            self.assertEqual(log_mass, -math.inf)
                        else:
                            self.assertAlmostEqual(log_mass, math.log(mass), places=12)
                    return values

                with patch.object(pe, '_exact_low_conf_log_a', side_effect=observe):
                    result = DP(**self.args(model, temperature))
                self.assertEqual(len(states), 7)
                self.assertEqual(len(model.calls), 7)
                self.assertAlmostEqual(result['log_probability'],
                                       math.log(exact_probability(temperature)), places=12)

    def test_dp_transition_logs_equal_sts_transition_logs(self):
        original = pe._exact_low_conf_log_a
        for temperature in (0.5, 1.0, 2.0):
            with self.subTest(temperature=temperature):
                model = ToyModel()
                states = {}

                def observe(*args, **kwargs):
                    values = original(*args, **kwargs)
                    states[model.calls[-1]] = values[0].tolist()
                    return values

                with patch.object(pe, '_exact_low_conf_log_a', side_effect=observe):
                    DP(**self.args(model, temperature))
                sts = STS(**self.args(ToyModel(), temperature), num_samples=128,
                          seed=1729, verbose=True, verbose_compact=True)
                for sample in sts['verbose_samples']:
                    state = ()
                    for record, position in zip(sample['steps'], sample['reveal_path_indices']):
                        self.assertEqual(record['log_a_active'], states[state])
                        state = tuple(sorted((*state, POSITIONS.index(position - 1))))

    def test_scheduling_chunks_preserve_result_and_singleton_forwards_without_cache(self):
        results = []
        evaluator_class = pe._LowConfidenceStateEvaluator
        instances = []

        class ObservedEvaluator(evaluator_class):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                instances.append(self)

        for batch_size in (1, 2, 64):
            model = ToyModel()
            with patch.object(pe, '_LowConfidenceStateEvaluator', ObservedEvaluator):
                result = DP(**self.args(model), state_batch_size=batch_size)
            results.append(result['log_probability'])
            self.assertEqual(result['state_batch_size'], batch_size)
            self.assertEqual(result['model_forward_batch_size'], 1)
            self.assertTrue(result['model_eval_mode'])
            self.assertEqual(result['model_forward_calls'], 7)
            self.assertEqual(len(set(model.calls)), 7)
            self.assertEqual(instances[-1].cache_max_bytes, 0)
            self.assertEqual(instances[-1].cache_bytes, 0)
        self.assertEqual(results, [results[0]] * len(results))

    def test_ten_masked_positions_near_1e_minus_100_match_analytic_result_and_sts(self):
        # Identical binary distributions with rare target probability p < 1/2:
        # every remaining candidate must be the target to beat all wrong tokens.
        # Smallest-index ties force a single successful path, so p_z=p^(10+...+1).
        target_probability = 10 ** (-100 / 55)
        model = ConstantLogitsModel(target_probability)
        args = self.args(model)
        args.update(masked_indexes=list(range(10, 0, -1)), steps=10)
        result = DP(**args)
        expected_log = 55 * math.log(target_probability)
        self.assertAlmostEqual(result['log_probability'], expected_log, places=11)
        self.assertGreater(result['probability'], 0)
        self.assertLess(abs(result['probability'] / 1e-100 - 1), 1e-11)
        self.assertEqual(model.calls, 1023)
        self.assertEqual(result['model_forward_calls'], 1023)
        args['model'] = ConstantLogitsModel(target_probability)
        sts = STS(**args, num_samples=8, seed=1729)
        self.assertAlmostEqual(result['log_probability'], sts['log_probability'], places=11)
        for sample_log in sts['sample_log_probabilities']:
            self.assertAlmostEqual(sample_log, expected_log, places=11)

    def test_zero_one_and_single_position_probabilities(self):
        for probability in (0.0, 1.0):
            result = DP(**self.args(ConstantLogitsModel(probability)))
            self.assertEqual(result['probability'], probability)
            self.assertEqual(result['log_probability'], -math.inf if probability == 0 else 0)
        args = self.args(ConstantLogitsModel(1e-100))
        args.update(masked_indexes=[50], steps=1, attention_mask=None)
        result = DP(**args)
        self.assertAlmostEqual(result['log_probability'], math.log(1e-100), places=12)
        self.assertLess(abs(result['probability'] / 1e-100 - 1), 1e-12)
        self.assertEqual(result['model_forward_calls'], 1)

    def test_settings_and_mixed_module_modes_restored_on_success_and_failure(self):
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
            for bad_value in (None, math.nan, math.inf, -math.inf):
                model = ToyModel(bad_value=bad_value).train()
                model.dropout.eval()
                with torch.autocast('cpu', dtype=torch.bfloat16):
                    if bad_value is None or bad_value == -math.inf:
                        DP(**self.args(model))
                    else:
                        with self.assertRaisesRegex(FloatingPointError, 'step=0, revealed_indices='):
                            DP(**self.args(model))
                self.assertTrue(model.training)
                self.assertFalse(model.dropout.training)
                self.assertEqual(settings(), before)
        finally:
            torch.use_deterministic_algorithms(original[0], warn_only=original[1])
            torch.backends.cudnn.benchmark = original[2]
            torch.backends.cudnn.deterministic = original[3]

    def test_invalid_transition_and_total_masses_raise(self):
        original = pe._exact_low_conf_log_a
        for value, error in (
            (math.nan, 'successful transition masses'),
            (math.inf, 'successful transition masses'),
            (math.log(1.01), 'successful transition masses'),
            (math.log(0.5), r'A\(S\)'),  # Three valid individual masses sum to 1.5.
        ):
            def corrupted(*args, **kwargs):
                return torch.full_like(original(*args, **kwargs), value)

            with self.subTest(value=value):
                with patch.object(pe, '_exact_low_conf_log_a', side_effect=corrupted):
                    with self.assertRaisesRegex(FloatingPointError, error + '.*step=0'):
                        DP(**self.args(ToyModel()))

    def test_invalid_final_probability_never_becomes_zero(self):
        for value in (math.nan, math.inf, math.log(1.01)):
            with self.subTest(value=value):
                with patch.object(pe, '_logaddexp_scalar', return_value=value):
                    with self.assertRaisesRegex(FloatingPointError, 'final DP probability'):
                        DP(**self.args(ToyModel()))


if __name__ == '__main__':
    unittest.main()
