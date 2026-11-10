"""Tests for macfit. Run: python3 -m unittest discover -s tests

validation_data.json holds the four models of the out-of-sample test: each config.json, the weight bytes,
the estimates declared before the measurement, and the measured peaks (MLX 0.32.4, mlx-lm 0.32.2, one M5).
"""
import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import macfit  # noqa: E402

GIB = 2**30
DATA = json.load(open(HERE / "validation_data.json"))


def estimate(model, tokens):
    m = DATA[model]
    return macfit.peak_bytes(m["config"], m["weights_bytes"], tokens, 1)["peak"] / GIB


class Regression(unittest.TestCase):
    def test_declared_estimates_unchanged(self):
        """The estimator still gives the values declared before the measurement."""
        for model, m in DATA.items():
            for tokens, declared in m["declared_estimate_gib"].items():
                self.assertAlmostEqual(estimate(model, int(tokens)), declared, delta=0.0006, msg=(model, tokens))


class Accuracy(unittest.TestCase):
    """The accuracy that README.md and VALIDATION.md state."""

    def error(self, model, tokens):
        return estimate(model, tokens) / DATA[model]["measured_peak_gib"][str(tokens)] - 1

    def test_within_5_percent_from_2_gb(self):
        for model in ("qwen3-4b", "qwen3-8b", "qwen3-30b-a3b"):
            for tokens in (2048, 8192):
                self.assertLessEqual(abs(self.error(model, tokens)), 0.05, (model, tokens))

    def test_small_model_within_15_percent(self):
        for tokens in (2048, 8192):
            self.assertLessEqual(abs(self.error("qwen3-0.6b", tokens)), 0.15, tokens)

    def test_errors_are_low_side(self):
        """Every estimate is below the measurement. The 'tight' band exists for this reason."""
        for model, m in DATA.items():
            for tokens in m["measured_peak_gib"]:
                self.assertLess(self.error(model, int(tokens)), 0, (model, tokens))


class Behaviour(unittest.TestCase):
    def test_weight_files_match_mlx_lm(self):
        names = ["consolidated.safetensors", "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors",
                 "original/model.safetensors"]
        self.assertEqual(macfit.weight_files(names), names[1:3])
        self.assertEqual(macfit.weight_files(["weights.safetensors", "sub/x.safetensors"]), ["weights.safetensors"])

    def test_verdict(self):
        self.assertEqual([macfit.verdict(s) for s in (0.5, 0.95, 1.01)], ["yes", "tight", "no"])

    def test_unsupported(self):
        base = {"num_hidden_layers": 2, "num_attention_heads": 4, "hidden_size": 64}
        for extra in ({"kv_lora_rank": 512}, {"model_type": "qwen3_next"}, {"layer_types": ["mamba", "full_attention"]}):
            with self.assertRaises(macfit.Unsupported):
                macfit.attention_layout({**base, **extra})
        with self.assertRaises(macfit.Unsupported):
            macfit.attention_layout({"hidden_size": 64})

    def test_sliding_window_caps_cache(self):
        cfg = {"num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
               "layer_types": ["sliding_attention", "full_attention"], "sliding_window": 512}
        layers, KV, d, _ = macfit.attention_layout(cfg)
        full = macfit.kv_cache_bytes(["full"], KV, d, 8192, 16)
        self.assertEqual(macfit.kv_cache_bytes(layers, KV, d, 8192, 16), full + full * 512 / 8192)

    def test_kv_steps_of_256(self):
        self.assertEqual(macfit.kv_cache_bytes(["full"], 1, 1, 1, 16), macfit.kv_cache_bytes(["full"], 1, 1, 256, 16))


if __name__ == "__main__":
    unittest.main()
