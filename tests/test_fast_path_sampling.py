"""The batched path must preserve successful-trajectory weights on batch-stable models."""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch


def load_module():
    likelihood = ModuleType("get_log_likelihood")
    generation = ModuleType("generate")
    def unused(*args, **kwargs):
        raise AssertionError("An unrelated estimator dependency was called.")
    likelihood.get_log_likelihood = unused
    likelihood.get_log_likelihood_from_partially_masked = unused
    generation.add_gumbel_noise = unused
    spec = importlib.util.spec_from_file_location(
        "_fast_path_sampling_test",
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


pe = load_module()


class BatchStableToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
        self.calls = 0

    def forward(self, tokens, attention_mask=None):
        self.calls += 1
        batch, length = tokens.shape
        assert length == 100
        logits = torch.zeros((batch, length, 3), device=tokens.device)
        positions = (0, 49, 99)
        for row in range(batch):
            revealed = sum(int(tokens[row, pos] != 3) for pos in positions)
            for slot, pos in enumerate(positions):
                logits[row, pos] = torch.tensor((
                    1.0 + slot * 0.15 + revealed * 0.1,
                    0.8 + (slot == 2) * 0.3,
                    -0.5,
                ), device=tokens.device)
        return SimpleNamespace(logits=logits)


class FastPathSamplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def check_mode(self, estimator, use_state_cache=True, verbose=False, **extra):
        common = dict(
            sequence_tokens=torch.zeros((1, 100), dtype=torch.long),
            masked_indexes=[1, 50, 100],
            steps=3,
            attention_mask=None,
            mask_id=3,
            num_samples=100,
            seed=1729,
            temperature=1.0,
            use_state_cache=use_state_cache,
            verbose=verbose,
            **extra,
        )
        regular = estimator(model=BatchStableToyModel(), fast=False, **common)
        faster = estimator(model=BatchStableToyModel(), fast=True, **common)
        self.assertEqual(regular["sample_log_probabilities"],
                         faster["sample_log_probabilities"])
        self.assertEqual(regular["probability"], faster["probability"])
        self.assertEqual(regular["model_forward_rows"],
                         faster["model_forward_rows"])
        self.assertLess(faster["model_forward_calls"],
                        regular["model_forward_calls"])

    def test_low_confidence(self):
        self.check_mode(
            pe._path_sampling_low_confidence_probability_fast_from_partially_masked,
        )

    def test_fast_dllm(self):
        self.check_mode(
            pe._path_sampling_fast_dllm_threshold_probability_fast_from_partially_masked,
            confidence_threshold=1.0,
        )

    def test_without_state_cache(self):
        self.check_mode(
            pe._path_sampling_low_confidence_probability_fast_from_partially_masked,
            use_state_cache=False,
        )
        self.check_mode(
            pe._path_sampling_fast_dllm_threshold_probability_fast_from_partially_masked,
            use_state_cache=False, confidence_threshold=1.0,
        )

    def test_verbose(self):
        self.check_mode(
            pe._path_sampling_low_confidence_probability_fast_from_partially_masked,
            verbose=True,
        )
        self.check_mode(
            pe._path_sampling_fast_dllm_threshold_probability_fast_from_partially_masked,
            verbose=True, confidence_threshold=1.0,
        )


if __name__ == "__main__":
    unittest.main()
