"""Backward-compatible one-off runner for the original SM_JSON ablation.

This keeps the 97d0040 behavior and output filenames:
  - results/results_ablation_sm_json.csv
  - results/results_ablation_sm_json_messages.jsonl

For the complete 2x4 protocol matrix, run `_run_full_ablation.py`.
"""

from __future__ import annotations

from experiment_utils import load_domain_samples, run_protocol_grid
from pipeline import Protocol


if __name__ == "__main__":
    samples = load_domain_samples(n_math=10, n_reading=10, n_news=10)
    run_protocol_grid(
        provider="openai",
        model="gpt-4o-mini",
        protocols=[Protocol.SHARED_MEMORY_JSON],
        domain_samples=samples,
        n_reps=3,
        output_csv="results/results_ablation_sm_json.csv",
        output_messages="results/results_ablation_sm_json_messages.jsonl",
    )

