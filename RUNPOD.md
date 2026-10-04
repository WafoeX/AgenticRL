# RunPod Setup Guide

The tried-and-tested way to run this project on RunPod. Follow these phases in order.

**Why RunPod over Colab:** ART requires torch >= 2.8.0, which Colab's default A100 runtime does not ship. Colab installs can work with in-place torch upgrades, but the upgrade path is fragile. RunPod has a dedicated PyTorch 2.8 image (`runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu`) that matches ART's requirement exactly — one less thing to break.

**Total cost:** ~$4-6 (A100 80GB at ~$1.89/hr × ~2-3 hours training + setup).
**Total time:** ~20 min setup + ~3 hours training.

---

## Phase 1 — Account + credit

1. Sign up at https://runpod.io (email + password, verify email)
2. Add $10 credit via Account → Billing. Covers this project with margin.

---

## Phase 2 — Deploy a pod

1. **Pods → Deploy**
2. **GPU:** Pick **A100-SXM4-80GB** (~$1.89/hr) or **A100 PCIe 40GB** (~$1.29/hr). 80GB recommended — more headroom, training finishes faster.
3. **Container Image:** Search the template gallery for **"PyTorch 2.8"** OR paste the image directly:
   ```
   runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu
   ```
4. **Storage:** 20 GB container disk, 20 GB volume (defaults are fine)
5. **Expose HTTP Ports:** `8888` (for optional Jupyter)
6. **Deploy On-Demand** (NOT Spot — spot pods can be evicted mid-training)

Pod boots in ~1 min.

---

## Phase 3 — Connect

**Pods → your pod → Connect → "Start Web Terminal"**

You're now root on a fresh Ubuntu 24.04 VM with PyTorch 2.8.0 preinstalled.

---

## Phase 4 — Sanity-check the starting environment (5 seconds)

```bash
nvidia-smi | head -15
python3 --version
python3 -c "import torch; print(f'torch: {torch.__version__}, cuda: {torch.version.cuda}')"
```

**Expected:**
- `NVIDIA A100-SXM4-80GB` or similar
- `Python 3.11.x`
- `torch: 2.8.0+cu128, cuda: 12.8`

❌ If torch is anything other than 2.8.0 — you picked the wrong template. Terminate and redeploy with `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu`.
✅ If all three look right, continue.

---

## Phase 5 — Install (the critical step)

```bash
pip install uv
uv pip install --system 'openpipe-art[backend]==0.5.17' 'pyyaml>=6.0' 'huggingface_hub'
```

**Why:**
- **`uv`** — Astral's pip replacement. Faster resolver, handles the custom vLLM wheel URL cleanly.
- **`--system`** — install into the system Python, no venv. Simpler for a single-task pod.
- **`openpipe-art[backend]==0.5.17`** — the `[backend]` extra pulls ART's custom vLLM fork (`vllm-0.17.0+art1`) from GitHub, plus pinned versions of trl, transformers, unsloth, accelerate, bitsandbytes, etc. Pinned to 0.5.17 for reproducibility.
- **`pyyaml` and `huggingface_hub`** — tools our scripts use directly but aren't in ART's backend extra.

**Expected runtime:** 5-10 min. Will download ~3 GB including the custom vLLM wheel.

**Expected exit:** no red traceback. Some yellow warnings about pre-existing Ubuntu packages are fine.

---

## Phase 6 — Verify imports (the gate)

```bash
python3 -c "import torch, art, vllm, trl, unsloth; print(f'torch={torch.__version__}, vllm={vllm.__version__}, trl={trl.__version__}, unsloth={unsloth.__version__}, all imports OK')"
```

**Expected output (this is the one line that matters):**
```
torch=2.8.0+cu128, vllm=0.17.0+art1, trl=0.20.0, unsloth=2026.3.3, all imports OK
```

**The critical check: `vllm=0.17.0+art1`.** If you see any other vllm version, you pulled upstream vLLM instead of the custom fork. Re-run with explicit wheel URL (see Debug section at the bottom).

---

## Phase 7 — Clone the repo

You need a fresh GitHub PAT. **Generate a new one** (not any you may have pasted in chat earlier):

1. https://github.com/settings/personal-access-tokens/new
2. **Fine-grained token**, expiration 1 day
3. **Repository access:** Only select repositories → `abhid1234/RL-Trained-Tool-Use-Agent`
4. **Repository permissions:** Contents → Read-only
5. Click Generate token
6. Copy the `github_pat_...` value (you only see it once)

**In the pod terminal:**

```bash
cd /workspace
read -s GH_TOKEN
# (terminal waits — paste the token, hit Enter. Invisible paste.)
git clone https://${GH_TOKEN}@github.com/abhid1234/RL-Trained-Tool-Use-Agent.git
cd RL-Trained-Tool-Use-Agent
unset GH_TOKEN
git log --oneline -3
```

Expected: latest commit on main is displayed. Should include the commit that added this RUNPOD.md file.

---

## Phase 8 — Run the test suite

```bash
python3 -m pytest tests/ -q
```

Expected: `32 passed`.

If a test fails because of an API mismatch with trl 0.20 / transformers 5.2 / ART 0.5.17, see the Debug section below.

---

## Phase 9 — Regenerate scenarios

```bash
python3 scripts/generate_scenarios.py
```

Expected:
```
Generated 200 training + 50 validation scenarios
```

---

## Phase 10 — Smoke test (the real moment of truth) ~4 min

```bash
rm -rf .art
python3 -m src.train --steps 2 --rollouts 32
```

**What to look for in the output, in order:**

1. `GRPO config: 4 scenarios/step × 8 rollouts/scenario = 32 rollouts/step`
2. vLLM engine init mentioning `vllm 0.17.0+art1` somewhere
3. `Starting from step 0`
4. Four per-group variance lines (one per group), each like `group N: mean=... std=X.XXX min=... max=... | task preview`
5. `groups with variance: N/4` where N ≥ 1 (ideally 4/4)
6. **THE CRITICAL LINE** — you want to see a `loss=X.XXX` or some non-SKIP training log, NOT `SKIP: all rewards equal`. This is the signal we've been chasing all day.
7. `Training complete! Final step: 2`

**If you see `SKIP: all rewards equal`** — the logprobs plumbing is still off. Check Debug issue 5 below.

**If you see `loss=X.XXX`** — we're clear to run full training. Proceed to Phase 11.

---

## Phase 11 — Full training (only after Phase 10 succeeds)

```bash
rm -rf .art
python3 -m src.train --config configs/train_config.yaml 2>&1 | tee training.log
```

**Runtime:** ~3 hours on A100 (depending on GPU tier).
**What to watch:** `Avg reward` should climb from ~-1.5 (step 1-5) → ~+1.5 (step 20) → ~+3 (step 50).

**Don't close the web terminal tab.** The training continues if you close it, but you lose live output. If you need to walk away, open a second terminal with `tmux attach` or just leave the tab open.

**If `Avg reward` stays flat at -3 for 10+ steps**, stop and re-check Cell 6 output for hints. Most likely suspect: `src/rollout.py` needs a small fix for the new ART API.

---

## Phase 12 — Evaluate and push

```bash
# Evaluate on validation set
python3 -m src.evaluate --scenarios data/scenarios_val.jsonl

# Push to HuggingFace
read -s HF_TOKEN
# (paste your HF WRITE token, hit Enter)
export HF_TOKEN
python3 scripts/export_model.py \
    --checkpoint-dir ./checkpoints/qwen-0.5b-tool-agent \
    --repo-id abhidaas/qwen-0.5b-tool-agent-grpo \
    --merge-adapter
unset HF_TOKEN
```

Your HF token needs **Write** scope (https://huggingface.co/settings/tokens → New → Write).

---

## Phase 13 — 🛑 STOP THE POD (critical for cost control)

**RunPod charges per minute as long as the pod is running.** Don't forget this step.

1. **Pods → your pod → Terminate** (NOT Stop — Terminate releases storage too)
2. Confirm the termination dialog
3. Check **Account → Billing** to verify total charges match expectations (~$4-6)

---

## Cost reality check

| Phase | Time | Cost (@$1.89/hr A100 80GB) |
|---|---|---|
| 1-3 (signup, deploy, connect) | 5 min | ~$0.16 |
| 4-6 (sanity + install + verify) | 10 min | ~$0.32 |
| 7-9 (clone + tests + scenarios) | 3 min | ~$0.09 |
| 10 (smoke test) | 5 min | ~$0.16 |
| 11 (full training) | 180 min | **~$5.67** |
| 12 (eval + export) | 5 min | ~$0.16 |
| **TOTAL** | **~3.5 hours** | **~$6.56** |

You bought $10 of credit → ~$3.50 buffer.

If you pick A100 PCIe 40GB ($1.29/hr) instead, total drops to ~$4.48.

---

## Debug Playbook

### Issue 1: Phase 6 shows `vllm=0.X.X` (NOT `0.17.0+art1`)

uv fell back to upstream vllm. Force-install the custom wheel:

```bash
uv pip install --system --reinstall \
  'https://github.com/vivekkalyan/vllm/releases/download/v0.17.0-art1/vllm-0.17.0%2Bart1-cp38-abi3-manylinux_2_31_x86_64.whl'
python3 -c "import vllm; print(vllm.__version__)"
```

### Issue 2: `ImportError` when running tests or smoke test

ART 0.5.17 may have renamed or moved something compared to our test stubs or `src/rollout.py`/`src/train.py`. Read the exact error, then:

- `tests/test_rollout.py` stubs `art` at module level — update the stub class to match the new API.
- `src/train.py` calls `backend.train(model, train_groups, learning_rate=...)` — if `learning_rate` was renamed, adjust the call.
- `src/rollout.py` uses `art.Trajectory(messages_and_choices=..., reward=..., metrics=...)` — if the kwarg names changed, adjust.

### Issue 3: Smoke test runs but prints `SKIP: all rewards equal`

The logprobs-extraction path in ART 0.5.17 may expect a different Choice object shape. Check:

```bash
grep -rn "allow_training_without_logprobs\|_get_packed_tensors" /usr/local/lib/python3.11/dist-packages/art/
```

Read the relevant section of backend.py to see what `_get_packed_tensors` looks for, then adjust `src/rollout.py` to produce that shape.

### Issue 4: CUDA out of memory during full training

Qwen 0.5B + custom vLLM on A100 80GB should have massive headroom. If you OOM:

- Reduce `num_generations` in `configs/train_config.yaml` from 8 → 4
- Reduce `rollouts_per_step` from 96 → 48
- Lower `max_completion_tokens` from 512 → 256

### Issue 5: Reward stays at -3 for many steps

The model isn't producing any tool calls. Check:

- `src/rollout.py` `SYSTEM_PROMPT` — is it clear enough?
- `temperature` and `top_p` in rollout — currently 1.2 / 0.95; try higher temperature
- The scenario task — are they solvable with the available tools?

### Issue 6: Training runs but reward doesn't climb past ~0

Learning rate too low or GRPO updates too small. Try:

```bash
python3 -m src.train --config configs/train_config.yaml --lr 1e-4
```

---

## Why this works (root cause explained)

Spent a full day fighting this because we assumed `openpipe-art` declared its deps the normal way. It doesn't.

1. `openpipe-art` (bare, no extras) pulls only client-side deps — no torch, no vllm, no unsloth. That's intentional: you can use the ART client to talk to a remote ART server without installing the training stack locally.
2. `openpipe-art[backend]` pulls the **training** stack. And critically, it pulls a **custom vLLM fork** from a direct GitHub URL. This fork has ART-specific patches — new API surface that upstream vLLM doesn't have.
3. ART's internal code in `art/vllm/patches.py` imports from `vllm.entrypoints.openai.chat_completion.protocol` — a path that only exists in the custom fork. Any upstream vLLM version (0.7 through 0.16) will fail with `ModuleNotFoundError` at import time.
4. The `[backend]` extra also pins **exact** versions of trl (0.20.0), transformers (5.2.0), unsloth (2026.3.3), etc. These are the versions ART's maintainers test against. Anything else is a gamble.

We tried every wrong combination imaginable before finding this. The definitive answer is in ART's `pyproject.toml` and nowhere else obvious in the docs.

---

## Sources

- [ART pyproject.toml (main branch)](https://raw.githubusercontent.com/OpenPipe/ART/main/pyproject.toml)
- [OpenPipe/ART GitHub](https://github.com/OpenPipe/ART)
- [openpipe-art on PyPI](https://pypi.org/project/openpipe-art/)
- [ART installation docs](https://art.openpipe.ai/getting-started/installation-setup)
- [RunPod PyTorch 2.8 template](https://www.runpod.io/articles/guides/pytorch-2-8-cuda-12-8)
- [Custom vLLM wheel URL](https://github.com/vivekkalyan/vllm/releases/download/v0.17.0-art1/vllm-0.17.0%2Bart1-cp38-abi3-manylinux_2_31_x86_64.whl)
