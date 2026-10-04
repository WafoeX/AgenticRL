# RUNBOOK — Operational Guide

Every gotcha we hit while trying to run this on Colab, and the fix. If you're debugging an unexpected error, search here first before anything else.

---

## 🚨 ROOT CAUSE DISCOVERED (2026-04-12)

After ~7 hours of dependency hell on both Colab and RunPod, the real root cause is:

> **ART uses a CUSTOM vLLM fork distributed as a direct wheel URL from GitHub, NOT upstream vLLM from PyPI.**

The custom fork is `vllm-0.17.0+art1` at https://github.com/vivekkalyan/vllm/releases — it has the `vllm.entrypoints.openai.chat_completion.protocol` module that ART's patches import. **Upstream vLLM (any version from PyPI) does not have this module.** That's why every upstream pin we tried failed — we were installing packages that ART's code cannot use.

**Every issue below is a consequence of the same root problem: we were pinning upstream packages when we should have been using ART's `[backend]` extra, which declares the custom vLLM wheel as a direct dependency.**

## ✅ The Correct Install (replaces everything below)

```bash
pip install uv
uv pip install --system 'openpipe-art[backend]==0.5.17' 'pyyaml>=6.0' 'huggingface_hub'

# Verify — vllm MUST be 0.17.0+art1 (the custom fork)
python3 -c "import torch, art, vllm, trl, unsloth; print(f'torch={torch.__version__}, vllm={vllm.__version__}, trl={trl.__version__}, unsloth={unsloth.__version__}, all imports OK')"
```

**Expected output:**
```
torch=2.8.0+cu128, vllm=0.17.0+art1, trl=0.20.0, unsloth=2026.3.3, all imports OK
```

### Host requirements for this install

- **Python 3.11+** (ART's pyproject requires `>=3.11`)
- **torch >= 2.8.0** preinstalled — Colab typically has older torch, and in-place upgrade is fragile. RunPod has a PyTorch 2.8 template (`runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu`) which is the known-good starting environment. See RUNPOD.md.
- **Linux x86_64** with glibc ≥ 2.31 (for the `manylinux_2_31_x86_64` vLLM wheel).

### What `[backend]` pulls (versions pinned by ART's pyproject.toml)

| Package | Pin |
|---|---|
| `torch` | `>=2.8.0` |
| `transformers` | `==5.2.0` |
| `trl` | `==0.20.0` |
| `peft` | `>=0.14.0` |
| `unsloth` | `==2026.3.3` |
| `unsloth-zoo` | `==2026.3.1` |
| `vllm` | `@ https://github.com/vivekkalyan/vllm/releases/download/v0.17.0-art1/vllm-0.17.0%2Bart1-cp38-abi3-manylinux_2_31_x86_64.whl` |
| `bitsandbytes` | `>=0.45.2` |
| `accelerate` | `==1.7.0` |

These pins come straight from [ART's pyproject.toml](https://raw.githubusercontent.com/OpenPipe/ART/main/pyproject.toml) `[project.optional-dependencies].backend`.

---

## Legacy Install Command (DEPRECATED — kept for context)

The pre-discovery command was:

```python
!pip install --quiet 'openpipe-art' 'trl>=0.15,<0.17' 'vllm>=0.10,<0.12' unsloth datasets peft pyyaml huggingface_hub
```

**This does not work.** It pulls upstream vllm which is missing the modules ART patches expect. Use the `[backend]` extra instead (see above).

---

## Known Issues and Fixes

### Issue 1: `ModuleNotFoundError: No module named 'unsloth'`

**Cause:** openpipe-art does `import unsloth` at module load time but doesn't declare unsloth as a hard pip dependency. Fresh installs miss it.

**Fix:** Install unsloth explicitly in Cell 2:

```python
!pip install unsloth
```

Already baked into the Golden Install Command above.

---

### Issue 2: `ImportError: cannot import name 'ConstantLengthDataset' from 'trl.trainer.utils'`

**Cause:** `trl>=0.17` moved or removed `ConstantLengthDataset` from `trl.trainer.utils`. `unsloth_zoo/dataset_utils.py` still imports from the old location.

**Fix:** Pin `trl<0.17`. The Golden Install Command does this.

```python
!pip install --force-reinstall --no-deps 'trl==0.15.2'
```

---

### Issue 3: `ModuleNotFoundError: No module named 'vllm.adapter_commons'`

**Cause:** `vllm>=0.12` (especially 0.16+) removed the `adapter_commons` module. `unsloth_zoo/vllm_lora_request.py` still imports from it.

**Fix:** Pin `vllm<0.12`. The Golden Install Command does this.

```python
!pip install --force-reinstall --no-deps 'vllm==0.10.2'
```

---

### Issue 4: `ResolutionImpossible` when pinning versions

**Cause:** pip's resolver couldn't find a combination of specific versions that all satisfy each other's requirements.

**Fix:** Use ranges (`>=0.15,<0.17`) instead of exact pins (`==0.15.2`) so pip has wiggle room to resolve. Drop the `transformers` pin entirely and let trl/vllm pull the version they need. Use `uv` instead of pip if pip takes >5 minutes resolving.

```python
!pip install uv
!uv pip install --system 'trl<0.17' 'vllm<0.12' 'openpipe-art' unsloth datasets peft pyyaml huggingface_hub
```

---

### Issue 5: `SKIP: all rewards equal` but per-group stats show variance

**Cause:** This message is misleading. The actual trigger is that `backend._get_packed_tensors()` returned `None`, which happens when:
1. **No per-token logprobs in trajectories** — the most common cause
2. All rewards in a group genuinely are identical (rare after fix #6 below)

**Fix:** Ensure `src/rollout.py`:
1. Requests `logprobs=True, top_logprobs=1` in every `chat.completions.create` call
2. Stores raw OpenAI `Choice` objects in `messages_and_choices`, not dict conversions

Already fixed in commit `ad11d51`. If you see this again, grep to confirm the fix is in place:

```python
!grep -c "logprobs=True" src/rollout.py  # should be >= 1
!grep -c "_build_api_messages" src/rollout.py  # should be >= 1
```

---

### Issue 6: "All rewards equal" because model outputs are deterministic

**Cause:** At `temperature=1.0`, the Qwen 0.5B base model produces near-identical outputs for the same prompt. GRPO's within-group variance collapses to zero.

**Fix:** Use `temperature=1.2, top_p=0.95` for exploration. Already in `src/rollout.py`.

---

### Issue 7: `Already borrowed` BadRequestError from vLLM

**Cause:** vLLM 0.16+ has a concurrency bug in guided tool-calling where multiple parallel requests race on the sampler state. Gets worse after LoRA swaps.

**Fix:** Pin `vllm<0.12` (the Golden Install Command does this). 0.10.2 doesn't have the bug.

**Workaround:** `src/rollout.py` has a defensive try/except that catches BadRequestError and returns a failed trajectory with reward=-3.0, so individual failures don't crash the whole training step.

---

### Issue 8: Training script exits immediately with "Starting from step N / Training complete"

**Cause:** ART's LocalBackend persists checkpoints in `.art/`. Each previous run (even failed ones) bumps the step counter. If you pass `--steps N` and the counter is already at N, the training loop runs zero iterations.

**Fix:** Clean `.art` before each smoke test:

```python
!rm -rf .art
!python3 -m src.train --steps 2 --rollouts 32
```

Already baked into Cell 6 and Cell 7.

---

### Issue 9: `GRPO config: 1 scenarios/step × 8 rollouts/scenario = 8 rollouts/step` — training gets skipped

**Cause:** With only 1 group per step, ART can't compute cross-group advantages. You need at least 2 groups, ideally 4+.

**Fix:** Use `--rollouts 32` or higher (so with `num_generations=8` you get ≥ 4 groups):

```python
!python3 -m src.train --steps 2 --rollouts 32
```

Already in Cell 6.

---

### Issue 10: `Trajectory.metrics` ValidationError ("Input should be a valid number")

**Cause:** ART's pydantic `Trajectory.metrics` is strictly typed `dict[str, float | int | bool]`. Old code tried to store `tools_called` (list[str]) and `task` (str) in metrics.

**Fix:** Only put numeric values in metrics. Already fixed in commit `7622f03`.

---

### Issue 11: `fatal: not a git repository`

**Cause:** Runtime restart (or timeout disconnect) wiped `/content/`. The repo directory is gone. `.art` and any local edits are gone. Installed packages may or may not have survived.

**Fix:** Re-clone from scratch.

```python
from google.colab import userdata
gh_token = userdata.get('GITHUB_TOKEN')
!rm -rf /content/RL-Trained-Tool-Use-Agent
!git clone https://{gh_token}@github.com/abhid1234/RL-Trained-Tool-Use-Agent.git /content/RL-Trained-Tool-Use-Agent
%cd /content/RL-Trained-Tool-Use-Agent
!git log --oneline -3
```

---

### Issue 12: `SecretNotFoundError: Secret GITHUB_TOKEN does not exist`

**Cause:** Either the secret isn't added in Colab's 🔑 sidebar, or "Notebook access" toggle is OFF, or the name is wrong (it's case-sensitive: `GITHUB_TOKEN` not `github_token`).

**Fix:** 🔑 sidebar → Add new secret → Name: `GITHUB_TOKEN` → Value: your PAT → toggle **Notebook access** ON.

---

### Issue 13: pip's `ResolutionTooDeep: 200000`

**Cause:** Pip's dependency resolver gave up after 200,000 rounds of backtracking. Happens when the dependency graph is too tangled.

**Fix:** Use `uv` instead of pip. It has a vastly better resolver.

```python
!pip install --quiet uv
!uv pip install --system 'openpipe-art' 'trl<0.17' 'vllm<0.12' unsloth datasets peft pyyaml huggingface_hub
```

---

## Compute Budget Management

### Starting budget

Colab Pro = **100 compute units / month** at $9.99/month.

### A100 burn rates

| State | Rate |
|---|---|
| A100 active (training) | ~11.77 units/hour |
| A100 idle + connected | ~3-5 units/hour |
| A100 disconnected | 0 units/hour |

### Budget math for this project

| Activity | Units |
|---|---|
| Smoke test (Cell 6, ~3 min) | ~1 |
| Full training (Cell 7, ~3 hours) | ~36 |
| Evaluation (Cell 8, ~5 min) | ~1 |
| HF export (Cell 9, ~2 min) | <1 |
| **Total needed** | **~40 units** |

Leaves ~60 units of headroom for retries and debugging. Budget is not the problem on this project.

### Hard tripwires

1. **Check the compute units counter** (top-right RAM/Disk dropdown) at the start of every session. If it's below 40, do NOT start a full training run.
2. **Disconnect the runtime when walking away** (Runtime → Disconnect runtime). Keeps files, stops idle burn.
3. **Manage Sessions** periodically (account menu bottom-left of colab.research.google.com) — terminate any ghost sessions from forgotten tabs. Ghost sessions DOUBLE your burn rate.
4. **Cancel Colab Pro within 28 days** if you don't need another month. Auto-renews.

---

## Runtime State Management

### Three kinds of "restart"

| Action | Python state | /content/ | Installed packages | Use when |
|---|---|---|---|---|
| **Restart session** | Wiped | Kept | Kept | Kernel is misbehaving but environment is fine |
| **Disconnect runtime** | Wiped | Kept | Kept (for a while) | Walking away, save idle compute |
| **Disconnect and delete runtime** | Wiped | Wiped | Wiped | Environment is broken, need fresh VM |

### When in doubt

**Disconnect and delete runtime → Reconnect → Run Cell 2 (install) → Cell 3 (clone).** This is the known-good reset sequence.

---

## Training Sanity Checks

### After Cell 6 (smoke test), verify before launching Cell 7

1. ✅ **GRPO config line** shows `4 scenarios/step × 8 rollouts/scenario = 32 rollouts/step`
2. ✅ **4 per-group variance lines** appear in the output, each with `std > 0`
3. ✅ **Non-SKIP message from ART** — look for `loss=X.XXX`, `train/reward`, or similar
4. ✅ **No import errors** in startup
5. ⚠️ Some `[rollout] caught BadRequestError` lines are OK if they're rare — if more than ~10% of rollouts fail, downgrade vllm more aggressively

### While Cell 7 runs, watch for

- `Avg reward` should climb step by step
- `groups with variance: N/12` should be mostly 12/12 or close
- `Avg tools/rollout` should stabilize around 1.5-2.5 (model calling appropriate number of tools)
- vLLM OOM → unlikely on A100 with Qwen 0.5B, but if it happens, reduce `num_generations` from 8 to 4

---

## Code Architecture Reminders

### `src/rollout.py` — the agent loop

Every rollout request MUST include:
- `logprobs=True` — ART reads these for GRPO policy gradient
- `top_logprobs=1` — at least the chosen token's logprob

Every trajectory MUST:
- Store raw `Choice` objects for assistant turns (not dict conversions)
- Use plain dicts only for system/user/tool turns
- Put only numeric values in `metrics` (no strings, no lists)

### `src/train.py` — GRPO loop

GRPO requires:
- `rollouts_per_step % num_generations == 0` (checked)
- Multiple groups per step (ideally `rollouts_per_step / num_generations >= 4`)
- Fresh `.art` state OR `num_steps > current_step`

### `src/rewards.py` — decomposed reward

Range: `[-3, +4]`. Components:
- `r_format` (0 or 1): did any tool get called?
- `r_correct` (−3 to +3): Jaccard on tool names + arg keys + arg values

If all rollouts in a group hit the same reward value, GRPO skips. Higher temperature in rollout (currently 1.2 + top_p 0.95) is the remedy.

---

## If All Else Fails

1. **Pin harder.** Try exact versions: `trl==0.15.2`, `vllm==0.10.2`, `unsloth==2025.3.19`, `openpipe-art==0.3.2`.
2. **Switch to RunPod.** ~$3-4 for 3 hours of A100, clean Ubuntu box, fresh Python environment. Not Colab-specific issues.
3. **Walk away. Try again tomorrow.** Sometimes the best debug move is sleep.

---

## Change Log

- **2026-04-12:** Initial runbook. Wrote after a full day of Colab debugging that uncovered every entry above.
