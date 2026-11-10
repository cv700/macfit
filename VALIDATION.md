# How we checked macfit

## Setup

- **Mac:** M5 with 10 GPU cores and 32 GB, macOS 26.2. Metal's working-set limit is 26,800,603,136 bytes (24.96 GiB).
- **Software:** MLX 0.32.4 and mlx-lm 0.32.2, Python 3.12.
- **Settings:** the mlx-lm defaults, which are a 16-bit KV cache and a prefill step of 2,048 tokens.
- **Measurement:** reset the MLX peak counter, then run `generate_step` with `max_tokens=1` on seeded random token IDs. The value is `mx.get_peak_memory()`. It includes two warm-up calls and ten timed calls at each prompt length. Each model ran in 3 fresh processes, and the table gives the median.
- **Estimate:** `macfit` with `--max-tokens 1`, to match the measurement.

## Out of sample

We wrote these estimates down before the measurement ran. We had never measured three of the four models on this Mac. The Qwen3-4B measurements set the in-flight term earlier, so the 4B row is partly in sample.

| Model | Prompt | Estimate | Measured | Error |
|---|---|---|---|---|
| Qwen3-0.6B, 4-bit | 128 | 0.440 GiB | 0.589 GiB | −25.3% |
| | 2,048 | 1.009 | 1.152 | −12.4% |
| | 8,192 | 1.665 | 1.781 | −6.5% |
| Qwen3-4B, 4-bit | 128 | 2.243 | 2.427 | −7.6% |
| | 2,048 | 2.874 | 2.984 | −3.7% |
| | 8,192 | 3.718 | 3.726 | −0.2% |
| Qwen3-8B, 4-bit | 128 | 4.427 | 4.552 | −2.7% |
| | 2,048 | 5.058 | 5.119 | −1.2% |
| | 8,192 | 5.902 | 5.916 | −0.2% |
| Qwen3-30B-A3B, 4-bit | 128 | 16.119 | 16.178 | −0.4% |
| | 2,048 | 16.656 | 16.785 | −0.8% |
| | 8,192 | 17.219 | 17.378 | −0.9% |

Error = estimate / measured − 1. A negative error means macfit estimated low.

**The rule we declared:** within ±5% at 2,048 and 8,192 tokens on all four models.
- The 4B, 8B and 30B-A3B pass.
- The 0.6B fails, at −12.4% and −6.5%. In absolute terms, the miss is only 0.12 to 0.14 GiB, but on a 0.3 GB model that is a large share.
- **At 128 tokens, the in-flight term is too small.** macfit assumes it shrinks in proportion below 512 tokens. The data shows a floor of about 0.1 to 0.2 GiB.

Both misses are on the low side. The "tight" verdict, from 90% of the limit, covers them for any model that is near the limit of a real Mac.

## In sample

These Qwen3-4B points set the in-flight term, so they are not a fair test. They do check the quantized-cache and prefill-step options.

| Case | Estimate | Measured | Error |
|---|---|---|---|
| 1K prompt, defaults | 2.71 GiB | 2.73 GiB | −0.6% |
| 8K prompt, defaults | 3.72 GiB | 3.73 GiB | −0.3% |
| 8K prompt, prefill step 512 | 3.67 GiB | 3.67 GiB | −0.1% |
| 8K prompt, 4-bit cache, step 2048 | 4.88 GiB | 4.78 GiB | +2.2% |
| 8K prompt, 4-bit cache, step 512 | 3.33 GiB | 3.31 GiB | +0.7% |
| 1K prompt, prefill step 512 | 2.68 GiB | 2.77 GiB | −3.1% |

## Not yet checked

- Sliding-window layers.
- Any other Mac, or other MLX versions.
- Models from other families (Llama, Mistral, Gemma).

## Data

`tests/validation_data.json` holds each model's `config.json`, its weight bytes, the declared estimates and the measured peaks. `tests/test_macfit.py` checks them.
