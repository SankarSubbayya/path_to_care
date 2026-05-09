"""GRPO/RLVR stretch goal — Group-Relative Policy Optimization with the
verifiable urgency reward (1.0 / 0.5 / 0.0 per docs/EVALUATION.md).

Status: SCAFFOLD ONLY. Not run for the 24-hour submission. Ships as an
artifact to preserve the RLVR narrative documented in docs/PROJECT.md and
docs/PLAN.md week 5.

If you want to actually run this, you need:
  - `trl` installed (>=0.8 has `GRPOTrainer`)
  - A working ROCm path for trl's reference-model rollouts
  - Enough labeled (case + ground_truth_urgency) data — we have 30; the
    brief calls for 50-100. Expect noisy training otherwise.

We deliberately do NOT add `trl` to the default pyproject deps because the
ROCm story is unverified. Add it locally if you want to try:
  uv add trl

Skeleton of the loop is below for reviewers.
"""
from __future__ import annotations

import json
import sys


def grpo_skeleton() -> None:
    """Outline of the loop — not executable as-is."""
    # 1. Load Gemma 4 + LoRA SFT-tuned adapter as the policy.
    # 2. Load Gemma 4 + frozen base as the reference model.
    # 3. For each batch of cases:
    #     - Sample K=4 responses per case at temperature > 0.
    #     - Score each response with harness.reward.reward(predicted, ground_truth).
    #     - Compute advantages via group-relative normalization.
    #     - Take a PPO-style step on the policy with KL penalty against the
    #       reference. trl.GRPOTrainer encapsulates this.
    # 4. Save the resulting adapter to adapters/triage-gemma4-grpo/.
    raise NotImplementedError(
        "GRPO/RLVR is a stretch goal documented in docs/PLAN.md week 5. "
        "Not part of the 24-hour submission. See docstring for the wiring."
    )


if __name__ == "__main__":
    print("GRPO is a stretch goal; not run for the 24-hour submission.")
    print("See training/grpo_stretch.py docstring for the wiring outline.")
    sys.exit(0)
