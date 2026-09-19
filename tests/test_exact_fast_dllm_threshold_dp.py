"""Exact fast-dLLM subset-DP alignment and precision regressions."""
import math
import statistics
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from test_fast_dllm_threshold_alignment import (
    MASK_ID, MC, POSITIONS, STS, ToyModel, enumerate_successful_subsets,
    exact_probability, pe, toy_logits,
)


DP = pe._exact_fast_dllm_threshold_probability_dp_from_partially_masked


class ConstantLogitsModel(torch.nn.Module):
    def __init__(self, target_probability):
        super().__init__()
        probability = float(target_probability)
        self.register_buffer("row", torch.tensor([
            math.log(probability) if probability else -math.inf,
            math.log1p(-probability) if probability < 1 else -math.inf,
        ], dtype=torch.float64))
        self.calls = 0

    @property
    def device(self):
        return self.row.device

    def forward(self, tokens, attention_mask=None):
        assert tokens.shape == (1, 100)
        assert not self.training
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.is_autocast_enabled("cpu")
        self.calls += 1
        return SimpleNamespace(logits=self.row.expand(1, 100, -1).clone())


class _ToyTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ln_f = torch.nn.Identity()


class LLaDAModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = _ToyTransformer()


class BatchToyModel(torch.nn.Module):
    """Batch-independent toy with the standard LLaDA projection structure."""
    def __init__(self):
        super().__init__()
        self.register_buffer("anchor", torch.zeros((), dtype=torch.float32))
        self.config = SimpleNamespace(model_type="llada")
        self.model = LLaDAModel()
        self.calls = []

    @property
    def device(self):
        return self.anchor.device

    def forward(self, tokens, attention_mask=None):
        assert not self.training
        batch_size = tokens.shape[0]
        hidden = torch.zeros((batch_size, 100, 3), dtype=torch.float32)
        states = []
        for row in range(batch_size):
            state = tuple(
                slot for slot, position in enumerate(POSITIONS)
                if tokens[row, position] != MASK_ID
            )
            states.append(state)
            for slot, position in enumerate(POSITIONS):
                hidden[row, position] = torch.tensor(toy_logits(state, slot))
        self.calls.append(tuple(states))
        return SimpleNamespace(logits=self.model.transformer.ln_f(hidden))


class BatchSensitiveToyModel(BatchToyModel):
    """Simulate reduced-precision logits changing with model batch shape."""
    def forward(self, tokens, attention_mask=None):
        output = super().forward(tokens, attention_mask=attention_mask)
        if tokens.shape[0] > 1:
            output.logits[..., 0] += 0.1
        return output


class ExactFastDLLMThresholdDPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def args(self, model, temperature=1.0, threshold=0.6):
        return dict(
            model=model,
            sequence_tokens=torch.zeros((1, 100), dtype=torch.long),
            masked_indexes=[100, 1, 50],
            steps=3,
            attention_mask=torch.ones((1, 100), dtype=torch.long),
            mask_id=MASK_ID,
            temperature=temperature,
            confidence_threshold=threshold,
        )

    def test_exact_dp_matches_exhaustive_decoder_at_boundaries_and_temperatures(self):
        for temperature, threshold in (
            (1.0, 0.0),
            (0.5, 0.2),
            (1.0, 0.6),
            (2.0, 0.8),
            (1.0, 1.0),
        ):
            with self.subTest(temperature=temperature, threshold=threshold):
                result = DP(**self.args(ToyModel(), temperature, threshold))
                expected = exact_probability(temperature, threshold)
                self.assertAlmostEqual(result["log_probability"], math.log(expected), places=12)
                self.assertAlmostEqual(result["probability"], expected, places=14)
                self.assertLessEqual(result["model_forward_calls"], 7)

    def test_every_dp_subset_transition_matches_exhaustive_enumeration(self):
        original = pe._enumerate_fast_dllm_transition_logs_from_lists
        model = ToyModel()
        observed_states = set()

        def observe(active_slots, log_ell, log_r, log_fallback):
            transitions = original(active_slots, log_ell, log_r, log_fallback)
            state = model.calls[-1]
            expected = enumerate_successful_subsets(state, 1.0, 0.6)
            for subset_bits, log_mass in transitions:
                selected = tuple(
                    slot for slot in range(3) if subset_bits & (1 << slot)
                )
                mass = expected.get(selected, 0.0)
                if mass == 0.0:
                    self.assertEqual(log_mass, -math.inf)
                else:
                    self.assertAlmostEqual(log_mass, math.log(mass), places=12)
            observed_states.add(state)
            return transitions

        with patch.object(
            pe, "_enumerate_fast_dllm_transition_logs_from_lists", side_effect=observe,
        ):
            result = DP(**self.args(model))
        self.assertEqual(len(observed_states), result["model_forward_calls"])
        self.assertEqual(len(model.calls), len(set(model.calls)))

    def test_exact_sts_and_mc_converge_to_same_probability(self):
        temperature, threshold = 2.0, 0.8
        exact = DP(**self.args(ToyModel(), temperature, threshold))["probability"]
        sts = STS(
            **self.args(ToyModel(), temperature, threshold),
            num_samples=4096,
            seed=1729,
            batch_size=509,
        )
        mc = MC(
            **self.args(ToyModel(), temperature, threshold),
            num_samples=40000,
            seed=1729,
            decoding_scheme="full",
            k=1,
            mc_batch_size=5003,
        )
        sts_se = statistics.stdev(sts["sample_probabilities"]) / math.sqrt(4096)
        mc_se = math.sqrt(exact * (1.0 - exact) / 40000)
        self.assertLess(abs(sts["probability"] - exact), 6 * sts_se + 1e-12)
        self.assertLess(abs(mc.estimate - exact), 6 * mc_se + 1e-12)

    def test_frontier_batching_matches_singleton_forwards_exactly(self):
        singleton_model = BatchToyModel()
        singleton = DP(
            **self.args(singleton_model),
        )
        batched_model = BatchToyModel()
        batched = DP(
            **self.args(batched_model),
            state_batch_size=64,
            use_selected_logits=True,
        )
        self.assertEqual(batched["log_probability"], singleton["log_probability"])
        self.assertEqual(batched["probability"], singleton["probability"])
        self.assertEqual(singleton["model_forward_calls"], 7)
        self.assertEqual(singleton["model_forward_batch_size"], 1)
        self.assertEqual(singleton["maximum_model_batch"], 1)
        self.assertFalse(singleton["selected_logits"])
        self.assertTrue(all(len(call) == 1 for call in singleton_model.calls))
        self.assertEqual(batched["model_forward_calls"], 3)
        self.assertEqual(batched["model_forward_rows"], 7)
        self.assertEqual(batched["maximum_model_batch"], 3)
        self.assertTrue(batched["selected_logits"])
        self.assertEqual(len(batched_model.calls), 3)

    def test_default_preserves_singleton_decoder_for_batch_sensitive_model(self):
        singleton_model = BatchSensitiveToyModel()
        exact = DP(**self.args(singleton_model))
        batched = DP(
            **self.args(BatchSensitiveToyModel()),
            use_selected_logits=True,
        )
        self.assertEqual(exact["model_forward_batch_size"], 1)
        self.assertTrue(all(len(call) == 1 for call in singleton_model.calls))
        self.assertAlmostEqual(
            exact["probability"], exact_probability(1.0, 0.6), places=14,
        )
        self.assertNotEqual(batched["probability"], exact["probability"])

    def test_log_space_preserves_tiny_probability_and_skips_unreachable_states(self):
        target_probability = 10 ** (-100 / 55)
        model = ConstantLogitsModel(target_probability)
        args = self.args(model, threshold=1.0)
        args.update(masked_indexes=list(range(10, 0, -1)), steps=10)
        result = DP(**args)
        expected_log = 55 * math.log(target_probability)
        self.assertAlmostEqual(result["log_probability"], expected_log, places=11)
        self.assertLess(abs(result["probability"] / 1e-100 - 1), 1e-11)
        self.assertEqual(result["model_forward_calls"], 10)
        self.assertEqual(result["num_skipped_unreachable_states"], 1013)

    def test_public_dispatch_and_audit_mode(self):
        common = dict(
            model=ToyModel(),
            prompt_tokens=torch.zeros((1, 50), dtype=torch.long),
            target_tokens=torch.zeros((1, 50), dtype=torch.long),
            steps=3,
            attention_mask=torch.ones((1, 100), dtype=torch.long),
            mask_id=MASK_ID,
            remasking="fast-dllm",
            estimation_method="exact",
            decoding_scheme="full",
            temperature=1.0,
            confidence_threshold=0.6,
            masked_indexes=[100, 1, 50],
        )
        routed = pe.compute_diffusion_probabilistic_extraction(**common)
        audit = DP(**self.args(ToyModel()), evaluate_all_states=True)
        self.assertEqual(routed["method"], "exact")
        self.assertEqual(routed["remasking"], "fast-dllm")
        self.assertEqual(routed["confidence_threshold"], 0.6)
        self.assertEqual(routed["log_probability"], audit["log_probability"])
        self.assertEqual(audit["model_forward_calls"], 7)

    def test_public_dispatch_accepts_ten_masks_without_fifty_index_guard(self):
        sentinel = {
            "probability": 0.25,
            "log_probability": math.log(0.25),
            "confidence_threshold": 0.9,
        }
        with patch.object(
            pe,
            "_exact_fast_dllm_threshold_probability_dp_from_partially_masked",
            return_value=sentinel,
        ) as exact_dp:
            result = pe.compute_diffusion_probabilistic_extraction(
                model=ToyModel(),
                prompt_tokens=torch.zeros((1, 50), dtype=torch.long),
                target_tokens=torch.zeros((1, 50), dtype=torch.long),
                steps=10,
                attention_mask=None,
                mask_id=MASK_ID,
                remasking="fast-dllm",
                estimation_method="exact",
                decoding_scheme="full",
                temperature=1.0,
                confidence_threshold=0.9,
                masked_indexes=list(range(91, 101)),
            )
        self.assertEqual(result["probability"], 0.25)
        self.assertEqual(result["method"], "exact")
        self.assertEqual(
            exact_dp.call_args.kwargs["masked_indexes"], list(range(91, 101)),
        )

    def test_exponential_guard_and_invalid_threshold(self):
        args = self.args(ToyModel())
        args.update(masked_indexes=list(range(1, 14)), steps=13)
        with self.assertRaisesRegex(ValueError, "exact fast-dLLM DP is exponential"):
            DP(**args)
        with self.assertRaisesRegex(ValueError, "confidence_threshold"):
            DP(**self.args(ToyModel(), threshold=math.nan))


if __name__ == "__main__":
    unittest.main()
