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

## Several sessions at once

Run on 2026-11-10 with `mlx_lm.benchmark -b`, which prefills all prompts of a batch together. 8,192-token prompts, 128 new tokens, prefill step 2,048. The same M5, MLX 0.32.4 and mlx-lm 0.32.2. The Mac was on battery, which does not change peak memory.

Before the run we declared one sum: weights + N × (KV cache + buffers in flight). We said it would be within 10% at 2, 4 and 8 sessions. It was not: three of the six cells missed, all on the high side. A sum with the buffers counted once came in low. So `--sessions` prints both.

| Model | Sessions | Measured | Low figure | High figure |
|---|---|---|---|---|
| Qwen3-8B, 4-bit | 2 | 7.84 GB | 7.58 GB (3.4% low) | 8.07 GB (2.8% high) |
| Qwen3-8B, 4-bit | 4 | 10.53 GB | 10.07 GB (4.5% low) | 11.52 GB (8.6% high) |
| Qwen3-8B, 4-bit | 8 | 16.18 GB | 15.06 GB (7.5% low) | 18.44 GB (12.2% high) |
| Qwen3-4B, 4-bit | 2 | 5.33 GB | 5.24 GB (1.9% low) | 5.72 GB (6.7% high) |
| Qwen3-4B, 4-bit | 4 | 8.04 GB | 7.73 GB (4.1% low) | 9.18 GB (12.4% high) |
| Qwen3-4B, 4-bit | 8 | 13.52 GB | 12.71 GB (6.4% low) | 16.09 GB (16.0% high) |

GB here is 10^9 bytes, as mlx-lm prints it. Each added session cost 1.33 to 1.48 GB. That is the KV cache (1.25 GB) plus 0.1 to 0.2 GB. With one session, both models were within 0.3%.

## Against figures that Apple published

Apple has published peak memory for seven standard models. We ran macfit with the same prompt length and token count.

| Model | Apple | macfit | Difference |
|---|---|---|---|
| Qwen3-1.7B, bf16 | 4.40 GB | 4.42 GB | +0.5% |
| Qwen3-8B, bf16 | 17.46 GB | 17.51 GB | +0.3% |
| Qwen3-8B, 4-bit | 5.61 GB | 5.48 GB | −2.4% |
| Qwen3-14B, 4-bit | 9.16 GB | 9.04 GB | −1.3% |
| gpt-oss-20b, MXFP4 (sliding window) | 12.08 GB | 11.77 GB | −2.5% |
| Qwen3-30B-A3B, 4-bit | 17.31 GB | 17.13 GB | −1.0% |
| Mistral Medium 3.5 128B, FP8, 65,536 tokens | 158 GB | 157.8 GB | −0.1% |

- The first six rows: a 4,096-token prompt and 128 new tokens on a 24 GB M5 MacBook Pro. Source: [Exploring LLMs with MLX and the Neural Accelerators in the M5 GPU](https://machinelearning.apple.com/research/exploring-llms-mlx-m5).
- The last row: a Mac Studio with M5 Ultra. Source: [Local and Distributed Large LLM Inference with MLX and RDMA on Mac Studio](https://machinelearning.apple.com/research/mlx-distributed-inference).
- We read "GB" in both posts as 10^9 bytes.
- The second post also reports up to 68 GB for Qwen 3.8 27B. That model mixes attention with recurrent layers, so macfit gives no estimate for it.

## Not yet checked

- Sliding-window layers, by a measurement of our own.
- Any other Mac, or other MLX versions.
- Models from other families (Llama, Gemma), by any measurement. Mistral, by a measurement of our own.
- Sessions that arrive at different times, and the saved caches of `mlx_lm.server`.

## Data

`tests/validation_data.json` holds each model's `config.json`, its weight bytes, the declared estimates and the measured peaks. `tests/test_macfit.py` checks them.
