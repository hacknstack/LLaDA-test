"""Exhaustive alignment tests for fast-dLLM threshold STS and direct MC."""
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
    likelihood = ModuleType("get_log_likelihood")
    generation = ModuleType("generate")

    def unused(*args, **kwargs):
        raise AssertionError("An unrelated estimator dependency was called.")

    likelihood.get_log_likelihood = unused
    likelihood.get_log_likelihood_from_partially_masked = unused
    generation.add_gumbel_noise = unused
    spec = importlib.util.spec_from_file_location(
        "_fast_dllm_alignment_test",
        Path(__file__).resolve().parents[1] / "probabilistic_extraction.py",
    )
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {
        "get_log_likelihood": likelihood,
        "generate": generation,
        spec.name: module,
    }):
        spec.loader.exec_module(module)
    return module


pe = load_extraction_module()
STS = pe._path_sampling_fast_dllm_threshold_probability_fast_from_partially_masked
MC = pe._monte_carlo_fast_dllm_threshold_probability_fast_from_partially_masked
POSITIONS = (0, 49, 99)
MASK_ID = 3


def toy_logits(state, slot):
    if slot < 2:
        return (2.0 + 0.25 * len(state), 1.0, 0.0)
    return (1.0 + (1.25 if 0 in state else 0.0), 2.0, 0.0)


def enumerate_successful_subsets(state, temperature, threshold):
    """Enumerate every candidate vector under the actual threshold decoder."""
    remaining = [slot for slot in range(3) if slot not in state]
    probabilities, confidence = {}, {}
    for slot in remaining:
        row = toy_logits(state, slot)
        sample_weights = [math.exp(value / temperature) for value in row]
        probabilities[slot] = [
            value / math.fsum(sample_weights) for value in sample_weights
        ]
        confidence_weights = [math.exp(value) for value in row]
        confidence[slot] = [
            value / math.fsum(confidence_weights) for value in confidence_weights
        ]

    masses = {}
    for tokens in product(range(3), repeat=len(remaining)):
        candidates = dict(zip(remaining, tokens))
        selected = tuple(
            slot for slot in remaining
            if confidence[slot][candidates[slot]] >= threshold
        )
        if not selected:
            selected = (max(
                remaining,
                key=lambda slot: (confidence[slot][candidates[slot]], -slot),
            ),)
        if all(candidates[slot] == 0 for slot in selected):
            probability = math.prod(
                probabilities[slot][candidates[slot]] for slot in remaining
            )
            masses[selected] = masses.get(selected, 0.0) + probability
    return masses


def exact_probability(temperature, threshold):
    @lru_cache(None)
    def visit(state):
        if len(state) == 3:
            return 1.0
        return math.fsum(
            mass * visit(tuple(sorted((*state, *selected))))
            for selected, mass in enumerate_successful_subsets(
                state, temperature, threshold,
            ).items()
        )

    return visit(())


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("anchor", torch.zeros((), dtype=torch.float32))
        self.dropout = torch.nn.Dropout(0.9)
        self.calls = []

    @property
    def device(self):
        return self.anchor.device

    def forward(self, tokens, attention_mask=None):
        assert tokens.shape == (1, 100)
        assert not self.training and not self.dropout.training
        assert not torch.is_autocast_enabled("cpu")
        assert torch.are_deterministic_algorithms_enabled()
        state = tuple(
            slot for slot, position in enumerate(POSITIONS)
            if tokens[0, position] != MASK_ID
        )
        self.calls.append(state)
        logits = torch.zeros((1, 100, 3), dtype=torch.float32)
        for slot, position in enumerate(POSITIONS):
            logits[0, position] = torch.tensor(toy_logits(state, slot))
        return SimpleNamespace(logits=logits)


class FastDLLMThresholdAlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def args(self, model, temperature, threshold, samples):
        return dict(
            model=model,
            sequence_tokens=torch.zeros((1, 100), dtype=torch.long),
            masked_indexes=[100, 1, 50],
            steps=3,
            attention_mask=torch.ones((1, 100), dtype=torch.long),
            mask_id=MASK_ID,
            num_samples=samples,
            seed=1729,
            temperature=temperature,
            confidence_threshold=threshold,
        )

    def test_sts_and_mc_agree_with_exhaustive_decoder(self):
        for temperature, threshold in (
            (1.0, 0.0),
            (0.5, 0.2),
            (1.0, 0.6),
            (2.0, 0.8),
            (1.0, 1.0),
        ):
            with self.subTest(temperature=temperature, threshold=threshold):
                expected = exact_probability(temperature, threshold)
                sts = STS(
                    **self.args(ToyModel(), temperature, threshold, 2048),
                    batch_size=257,
                )
                mc = MC(
                    **self.args(ToyModel(), temperature, threshold, 20000),
                    decoding_scheme="full",
                    k=1,
                    mc_batch_size=3001,
                )
                sts_se = statistics.stdev(sts["sample_probabilities"]) / math.sqrt(2048)
                mc_se = math.sqrt(expected * (1.0 - expected) / 20000)
                self.assertLess(abs(sts["probability"] - expected), 6 * sts_se + 1e-12)
                self.assertLess(abs(mc.estimate - expected), 6 * mc_se + 1e-12)

    def test_public_dispatch_routes_both_estimators(self):
        common = dict(
            prompt_tokens=torch.zeros((1, 50), dtype=torch.long),
            target_tokens=torch.zeros((1, 50), dtype=torch.long),
            steps=3,
            attention_mask=torch.ones((1, 100), dtype=torch.long),
            mask_id=MASK_ID,
            remasking="fast-dllm",
            num_samples=64,
            seed=1729,
            decoding_scheme="full",
            temperature=1.0,
            confidence_threshold=0.6,
            masked_indexes=[100, 1, 50],
        )
        with patch.object(pe.time, 'perf_counter', side_effect=[1.0, 3.0, 5.0, 9.0]):
            sts = pe.compute_diffusion_probabilistic_extraction(
                model=ToyModel(), estimation_method="path_sampling",
                return_sample_times=True, **common,
            )
            mc = pe.compute_diffusion_probabilistic_extraction(
                model=ToyModel(), estimation_method="monte-carlo",
                return_sample_logs=True, return_sample_times=True, **common,
            )
        self.assertEqual(sts["method"], "path_sampling")
        self.assertEqual(mc["method"], "monte-carlo")
        self.assertEqual(sts["remasking"], "fast-dllm")
        self.assertEqual(mc["remasking"], "fast-dllm")
        self.assertEqual(sts["confidence_threshold"], 0.6)
        self.assertEqual(mc["confidence_threshold"], 0.6)
        self.assertEqual(len(mc["sample_log_probabilities"]), 64)
        self.assertEqual(set(mc["sample_log_probabilities"]), {0.0, -math.inf})
        self.assertEqual(mc["sample_log_probabilities"].count(0.0), mc["hits"])
        self.assertIsNone(mc["verbose_samples"])
        self.assertEqual(sts["sample_wall_time_seconds"], [2.0] * 64)
        self.assertEqual(mc["sample_wall_time_seconds"], [4.0] * 64)

    def test_every_sts_normalizer_matches_exhaustive_transition_mass(self):
        temperature, threshold = 1.0, 0.6
        result = STS(
            **self.args(ToyModel(), temperature, threshold, 128),
            batch_size=31,
            verbose=True,
            verbose_compact=True,
        )
        for sample in result["verbose_samples"]:
            state = ()
            for record in sample["steps"]:
                expected_A = math.fsum(
                    enumerate_successful_subsets(
                        state, temperature, threshold,
                    ).values()
                )
                self.assertAlmostEqual(math.exp(record["log_A"]), expected_A, places=12)
                selected = tuple(
                    POSITIONS.index(position - 1)
                    for position in record["revealed_indices"]
                )
                state = tuple(sorted((*state, *selected)))

    def test_cache_and_batching_do_not_change_seeded_results(self):
        args = self.args(ToyModel(), 1.0, 0.6, 128)
        baseline = STS(**args, batch_size=17)
        for cached in (False, True):
            result = STS(
                **self.args(ToyModel(), 1.0, 0.6, 128),
                batch_size=17,
                use_state_cache=cached,
            )
            self.assertEqual(
                result["sample_log_probabilities"],
                baseline["sample_log_probabilities"],
            )

        baseline_mc = MC(
            **self.args(ToyModel(), 1.0, 0.6, 512),
            decoding_scheme="full",
            k=1,
            mc_batch_size=71,
        )
        for cached in (False, True):
            result = MC(
                **self.args(ToyModel(), 1.0, 0.6, 512),
                decoding_scheme="full",
                k=1,
                mc_batch_size=71,
                use_state_cache=cached,
            )
            self.assertEqual(result.hits, baseline_mc.hits)

    def test_variable_step_states_are_evaluated_once_when_cached(self):
        sts_model = ToyModel()
        STS(
            **self.args(sts_model, 1.0, 0.6, 512),
            batch_size=512,
        )
        self.assertEqual(len(sts_model.calls), len(set(sts_model.calls)))

        mc_model = ToyModel()
        MC(
            **self.args(mc_model, 1.0, 0.6, 4096),
            decoding_scheme="full",
            k=1,
            mc_batch_size=4096,
        )
        self.assertEqual(len(mc_model.calls), len(set(mc_model.calls)))


if __name__ == "__main__":
    unittest.main()
