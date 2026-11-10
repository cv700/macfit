# macfit

Will this model fit on my Mac? `macfit` estimates the peak memory that [mlx-lm](https://github.com/ml-explore/mlx-lm) needs for a model and a prompt length. It reads only the model's `config.json` and file sizes, so you can check a model before you download it.

## Use it

You need only the Python that ships with macOS. There is nothing to install.

```bash
curl -O https://raw.githubusercontent.com/cv700/macfit/main/macfit.py
```

```bash
python3 macfit.py mlx-community/Qwen3-8B-4bit
```

More examples:

```bash
python3 macfit.py mlx-community/Qwen3-8B-4bit --context 8192,32768,131072
```

```bash
python3 macfit.py mlx-community/Qwen3-8B-4bit --kv-bits 4
```

```bash
python3 macfit.py ~/models/my-local-model --json
```

For a gated or private repo, set `HF_TOKEN`. macfit sends it only to huggingface.co.

## Sample output

From a 32 GB M5:

```
Model:  mlx-community/Qwen3-8B-4bit at 545dc425
        weights 4.29 GiB, 36 layers, 8 KV heads, head size 128
Mac:    GPU memory limit 24.96 GiB (from Metal)
Setup:  16-bit KV cache, prefill step 2048, 512 new tokens

   Prompt    KV cache        Peak  Fits?
    8,192    1.23 GiB    5.97 GiB  yes, 24% of the limit
  131,072   18.11 GiB   22.85 GiB  tight, 92% of the limit
  145,000   20.00 GiB   24.75 GiB  tight, 99% of the limit

Longest prompt with 10% to spare: about 128,448 tokens (at the limit: about 146,624).
The model itself is trained for up to 40,960 tokens.
Tight means within 10% of the limit. Estimates have run up to 4% low, so a tight fit may fail.
Other apps share this memory, so leave some room. This is an estimate, not a guarantee.
```

## What the answer means

- **The GPU memory limit** is the working-set size that Metal recommends for this Mac. MLX uses the same value. On a 32 GB M5, it is 24.96 GiB, or 78% of RAM.
- **"yes"** means the estimate is at most 90% of the limit.
- **"tight"** means 90% to 100%. Every measured estimate so far has been low, by up to 4% on models of 2 GB and up. So a tight fit may fail.
- **"NO"** means the estimate is over the limit.

## How accurate it is

We tested it on one M5 with 32 GB, MLX 0.32.4 and mlx-lm 0.32.2. We wrote down the estimates before we measured. We had never measured three of the four models on this Mac.

| Model | Size | Error at 2,048 tokens | Error at 8,192 tokens |
|---|---|---|---|
| Qwen3-0.6B, 4-bit | 0.3 GB | 12.4% low | 6.5% low |
| Qwen3-4B, 4-bit | 2.3 GB | 3.7% low | 0.2% low |
| Qwen3-8B, 4-bit | 4.6 GB | 1.2% low | 0.2% low |
| Qwen3-30B-A3B, 4-bit (MoE) | 17.2 GB | 0.8% low | 0.9% low |

Every estimate came in below the measurement, by 0.01 to 0.18 GiB. That matters only near the limit, which is why "tight" starts at 90%. [VALIDATION.md](VALIDATION.md) has the method and all the numbers.

## What it does not do

- **Some models get only a weights answer.** That covers latent attention (DeepSeek), hybrid and recurrent layers (Qwen3-Next, Mamba), and configs that leave out fields (Gemma 3). For these models, macfit tells you whether the weights fit and why it stops there.
- **Sliding-window models** (gpt-oss, Gemma 2) get an estimate, but we have not yet checked one against a measurement.
- **It estimates mlx-lm only.** llama.cpp, Ollama and LM Studio with GGUF files use memory in other ways.
- **One Mac so far.** Other chips and MLX versions may differ. If you measure your Mac, please open an issue with the numbers.
- **No speed estimate yet.**

## How it estimates

```
peak = weights            (sum of the model*.safetensors file sizes, the files mlx-lm loads)
     + KV cache           (layers × KV heads × head size × 2 × bytes × tokens, in steps of 256 tokens)
     + buffers in flight  (0.40 to 0.45 GiB at prefill chunks of 512 to 2,048 tokens)
     + score matrices     (only with a 4-bit or 8-bit KV cache, which writes them out)
```

## Tests

```bash
python3 -m unittest discover -s tests
```

The tests check that the estimator still gives the values we wrote down before the measurement. They also check the accuracy that this README states.

## License

No license yet. Until one is added, the default copyright rules apply.
