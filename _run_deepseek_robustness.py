"""Run a small DeepSeek V4 Flash robustness check across all 8 protocols.

This is intentionally smaller than the main experiment. It checks whether the
mechanism/format trends are directionally stable under a second model provider
without turning the project into a full two-model factorial experiment.

Design:
  8 protocols x 3 domains x 5 samples/domain x 2 reps = 240 runs

Requires:
  DEEPSEEK_API_KEY in the environment or project `.env`.

Outputs:
  - results/results_deepseek_v4_flash_8protocols.csv
  - results/results_deepseek_v4_flash_8protocols_messages.jsonl
"""

from __future__ import annotations

from experiment_utils import load_domain_samples, run_protocol_grid
from pipeline import FULL_FACTORIAL_PROTOCOLS


if __name__ == "__main__":
    samples = load_domain_samples(n_math=5, n_reading=5, n_news=5)
    run_protocol_grid(
        provider="deepseek",
        model="deepseek-v4-flash",
        protocols=FULL_FACTORIAL_PROTOCOLS,
        domain_samples=samples,
        n_reps=2,
        output_csv="results/results_deepseek_v4_flash_8protocols.csv",
        output_messages="results/results_deepseek_v4_flash_8protocols_messages.jsonl",
    )

