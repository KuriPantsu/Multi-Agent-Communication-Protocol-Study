"""Run the complete supplemental 2x4 protocol ablation on gpt-4o-mini.

The main proposal experiment remains the original 360-run grid:
  NL / MARKDOWN / JSON / SHARED_MEMORY.

This script adds the four missing cells needed for the full mechanism x format
matrix without overwriting main results:
  RELAY_DEFAULT, SHARED_MEMORY_NL, SHARED_MEMORY_MARKDOWN, SHARED_MEMORY_JSON

Outputs:
  - results/results_ablation_full_factorial.csv
  - results/results_ablation_full_factorial_messages.jsonl
"""

from __future__ import annotations

from experiment_utils import load_domain_samples, run_protocol_grid
from pipeline import FULL_FACTORIAL_ABLATION_PROTOCOLS


if __name__ == "__main__":
    samples = load_domain_samples(n_math=10, n_reading=10, n_news=10)
    run_protocol_grid(
        provider="openai",
        model="gpt-4o-mini",
        protocols=FULL_FACTORIAL_ABLATION_PROTOCOLS,
        domain_samples=samples,
        n_reps=3,
        output_csv="results/results_ablation_full_factorial.csv",
        output_messages="results/results_ablation_full_factorial_messages.jsonl",
    )

