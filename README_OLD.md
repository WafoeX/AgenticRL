# RL-Trained Tool-Use Agent

A weekend experiment in training a 0.5B model to call APIs correctly using GRPO — the same algorithm behind DeepSeek-R1.

The training itself ran in 90 minutes. The other 47 hours were dependency and framework debugging. This repo holds the full code, the patches, the trained model, and a `RUNBOOK.md` of every error I hit so you don't have to.

- **Trained model:** [abhid1234/qwen-0.5b-tool-agent-grpo](https://huggingface.co/abhid1234/qwen-0.5b-tool-agent-grpo) on HuggingFace
- **Write-up:** [Substack post](https://abhid.substack.com/p/i-tried-to-train-a-05b-model-to-use)
- **Stack:** [OpenPipe ART](https://github.com/OpenPipe/ART) · GRPO · Qwen 2.5 0.5B Instruct · custom vLLM fork

---

## Results

Eval on 50 validation scenarios (8 rollouts each = 400 total) using the step-15 checkpoint, after the framework crashed for good at step 16 of a planned 50.

| Metric | Value |
|---|---|
| Overall accuracy | **4.5%** (18 of 400 rollouts) |
| Best scenario — "Convert 98 kg to lbs" | **87.5%** (7 of 8 rollouts) |
| Weather lookups | ~25% |
| Multi-step tool chains | ~10% |
| Average reward | -1.885 |
| Training steps completed | 16 of 50 planned |

The 4.5% headline is the wrong number to fixate on. The model learned one task pattern reliably in 16 GRPO steps and was beginning to pick up a few others. Multi-step chains needed more training than I could fit in — the reward function had enough signal granularity for single-tool calls but not for chained ones.

### Reward curve across training

![Reward curve](docs/images/reward_curve.svg)

Starting at -0.2, oscillating through -2 and +1.97, ending at +1.6 by step 16. Classic noisy early-training behavior — finding patterns, losing them, finding them again.

### Per-scenario breakdown

![Scenarios won](docs/images/scenarios_won.svg)

Of 50 validation scenarios, 9 had at least one successful rollout in 8 attempts. Unit conversion was the breakout pattern.

### Where the time actually went

![Pain timeline](docs/images/pain_timeline.svg)

48 hours, 11 milestones, 9 framework bugs before the first successful training step. The full breakdown of every error is in [RUNBOOK.md](RUNBOOK.md).

---

## How it works

![Architecture](docs/images/architecture.svg)

The training loop is roughly 340 lines of Python:

1. **Tool environment** (`src/tools.py`) — 5 deterministic APIs: calculator, weather lookup, search, unit converter, country database.
2. **Scenario generator** (`src/scenarios.py`) — 200 multi-step training scenarios from 12 templates, generated deterministically.
3. **Reward function** (`src/rewards.py`) — Decomposed scoring from the [ToolRL paper](https://arxiv.org/abs/2504.13958): format check + tool name match + argument correctness, range [-3, +4].
4. **Rollout** (`src/rollout.py`) — ART rollout loop with retry logic for vLLM's "Already borrowed" concurrency bug.
5. **Training** (`src/train.py`) — GRPO loop via ART's `LocalBackend` with a monkey-patched `_monitor_openai_server` to suppress crashes.

---

## Quick start

```bash
pip install uv
uv pip install --system 'openpipe-art[backend]==0.5.17'

# Apply the runtime patches needed by transformers 5.x and ART
python scripts/patch_art_tokenize.py

# Generate scenarios and run a 2-step smoke test
python scripts/generate_scenarios.py
python -m src.train --steps 2 --rollouts 32
```

The `[backend]` extra is non-negotiable. It's what pulls the custom vLLM fork (`vllm-0.17.0+art1`) from a GitHub release URL — without it, `pip install vllm` resolves to upstream vLLM, which is missing the modules ART's patches import. This was the single biggest debugging breakthrough of the project. Six hours of pinning before I finally read ART's `pyproject.toml` and saw it.

The full setup procedure for RunPod (recommended over Colab because of the torch 2.8 requirement) is in [`RUNPOD.md`](RUNPOD.md).

---

## What I'd do differently

1. **Start with RunPod, not Colab.** Colab's fixed PyTorch base image is a liability when the project requires torch 2.8.
2. **Read `pyproject.toml` before trusting any install command.** The optional extras and pinned URLs there are what the project actually tests against.
3. **Smoke-test at the real concurrency level.** My early smoke test was 2 steps × 8 rollouts — too small to catch the vLLM concurrency bug. 32 rollouts is the minimum that surfaces it.
4. **Spend more time on the reward function.** Once the pipeline ran, the reward function did most of the work. Shaping it would have moved the needle further than infrastructure tweaks ever did.
5. **Treat monkey-patches as first-class code.** ART's `_monitor_openai_server` patch and the tokenize.py normalization are load-bearing now and live in `scripts/`.

---

## Project structure

```
src/
  tools.py             # 5 deterministic tool definitions + executors
  scenarios.py         # Multi-step task generator (12 templates)
  rewards.py           # Decomposed reward function (ToolRL-inspired)
  rollout.py           # ART rollout loop + vLLM retry logic
  train.py             # GRPO training loop via ART LocalBackend
  evaluate.py          # Per-scenario evaluation harness
scripts/
  generate_scenarios.py # Build training + validation scenario JSONL
  patch_art_tokenize.py # Runtime patch for ART under transformers 5.x
  eval_checkpoint.py    # Evaluate a saved ART checkpoint
  export_model.py       # Merge LoRA adapter and push to HF Hub
configs/
  train_config.yaml     # GRPO hyperparameters
notebooks/
  train_colab.ipynb     # Colab fallback (RunPod is preferred)
docs/
  images/               # README chart assets
RUNBOOK.md              # 13 framework issues and their workarounds
RUNPOD.md               # Step-by-step RunPod deployment runbook
```

---

## Tech stack

- **[OpenPipe ART](https://github.com/OpenPipe/ART)** — RL training framework for agents (v0.5.17)
- **[vLLM custom fork](https://github.com/vivekkalyan/vllm/releases/tag/v0.17.0-art1)** — `vllm-0.17.0+art1`, distributed as a wheel URL via ART's `[backend]` extra
- **[HuggingFace TRL](https://github.com/huggingface/trl)** — `GRPOTrainer` (wrapped by ART)
- **[Unsloth](https://github.com/unslothai/unsloth)** — Training acceleration
- **[Qwen 2.5 0.5B Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)** — Base model (498M params, 4.4M trainable via LoRA)
- **[uv](https://github.com/astral-sh/uv)** — pip replacement (faster resolver, handles direct URL deps)
- **[RunPod](https://runpod.io)** — GPU host (A100 80GB on the PyTorch 2.8 template)

## Papers referenced

- **[ToolRL: Reward is All Tool Learning Needs](https://arxiv.org/abs/2504.13958)** — The reward function design
- **[GRPO (DeepSeek-Math)](https://arxiv.org/abs/2402.03300)** — Original GRPO formulation
- **[DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1)** — The model that made GRPO famous

---

## Artifacts

Every file from the actual training run is on HuggingFace alongside the trained model:

- **Trained model:** [huggingface.co/abhid1234/qwen-0.5b-tool-agent-grpo](https://huggingface.co/abhid1234/qwen-0.5b-tool-agent-grpo)
- **Training log** (every step, every reward): [training.log](https://huggingface.co/abhid1234/qwen-0.5b-tool-agent-grpo/blob/main/artifacts/training.log)
- **Eval results JSON** (per-scenario breakdown): [eval_results.json](https://huggingface.co/abhid1234/qwen-0.5b-tool-agent-grpo/blob/main/artifacts/eval_results.json)
- **Reward curve** (raw numbers): [reward_curve.txt](https://huggingface.co/abhid1234/qwen-0.5b-tool-agent-grpo/blob/main/artifacts/reward_curve.txt)
- **All artifacts:** [tree/main/artifacts](https://huggingface.co/abhid1234/qwen-0.5b-tool-agent-grpo/tree/main/artifacts)

If you try to reproduce this and hit an error I didn't cover, file an issue. The point of the project wasn't the trained model — it was understanding what actually breaks when you try to do this for real.
