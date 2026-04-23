# Communication Protocol Effects on Multi-Agent System Efficiency and Task Completion

> STAT GR5293 (Generative AI using Large Language Models) — Columbia University, Spring 2026

**Team.** Yi Zhang (yz5104), Xian Zhang (xz3447), Tianyu Zhan (tz2704) — Department of Statistics, Columbia University.

## Summary

A controlled experimental study of how inter-agent communication protocol choice affects multi-agent LLM system performance. Using a fixed three-agent pipeline (**Planning → Execution → Integration**), we compare four communication protocols — Natural Language, Markdown, JSON, Shared Memory — across three task domains (GSM8K mathematical reasoning, SQuAD reading comprehension, curated news analysis) drawn from standard benchmarks. A full 4 × 3 factorial design with 10 samples per domain and 3 repetitions per cell yields 360 pipeline runs. We measure token consumption (input / output / total), wall-clock latency, and task completion quality, and analyze the results with two-way ANOVA, Tukey HSD post-hoc, Cohen's d effect sizes, and 2 000-resample bootstrap confidence intervals.

The repository includes the experiment notebook, a thin shared-helper module, a live Streamlit demo that auto-selects the empirically optimal protocol for a user task, and figures + tables for the final report.

## Repository layout

```
.
├── pipeline.py                                  Shared helpers (agents, logger, evaluators, runner)
├── Multi_Agent_Communication_Protocol_Study.ipynb   Experiment driver: data loading → 360-run grid → analysis → figures
├── app.py                                       Streamlit demo
├── requirements.txt                             Python dependencies
├── results/                                     Populated by the notebook
│   ├── results_raw.csv                          one row per pipeline run
│   ├── results_messages.jsonl                   one JSON line per inter-agent message
│   ├── results_summary.csv                      aggregated table (protocol × domain)
│   └── experiment_config.json                   run configuration snapshot
├── figures/                                     Populated by the notebook
│   ├── fig1_tokens_by_protocol_domain.png
│   ├── fig2_completion_by_cell.png
│   ├── fig3_interaction.png
│   ├── fig4_pareto.png
│   ├── fig5_input_vs_output.png
│   └── fig6_latency.png
├── report/
│   └── REPORT_OUTLINE.md                        Skeleton for the final report
├── Project_Rubric_STATGR5293_2026.pdf           Course grading rubric (authoritative)
├── STAT_GR5293_Proposal_yz5104_xz3447_tz2704.pdf    Submitted proposal (authoritative)
├── old_code/                                    Prior notebook iterations (not used)
└── old_proposal/                                Earlier proposal drafts (not used, except §10/§7 notes)
```

## Quickstart

### 1. Set up the environment

```bash
git clone https://github.com/KuriPantsu/Multi-Agent-Communication-Protocol-Study.git
cd Multi-Agent-Communication-Protocol-Study
python3 -m venv .venv && source .venv/bin/activate    # optional but recommended
pip install -r requirements.txt
```

Python 3.10+ is expected. The notebook and the Streamlit app share `pipeline.py`, so both must run from the project root.

### 2. Provide your OpenAI API key

Create a `.env` file in the project root (the `.gitignore` already protects it) — copy `.env.example` as a template:

```bash
cp .env.example .env
# then edit .env and paste your real key:
# OPENAI_API_KEY=sk-proj-...
```

Both the notebook (via `python-dotenv`) and the Streamlit app pick the key up from here. You can also simply `export OPENAI_API_KEY=sk-...` in the shell before launching, and skip the `.env` file.

### 3. Run the experiment end-to-end (~10 min, ~$0.40)

Open `Multi_Agent_Communication_Protocol_Study.ipynb` in Jupyter or VS Code and **Run All**, *or* execute headless:

```bash
jupyter nbconvert --to notebook --execute \
    Multi_Agent_Communication_Protocol_Study.ipynb \
    --output Multi_Agent_Communication_Protocol_Study.ipynb
```

Outputs:

- `results/results_raw.csv` — one row per pipeline run (360 rows after a full grid)
- `results/results_messages.jsonl` — one JSON line per inter-agent message (1 080 lines)
- `results/results_summary.csv` — aggregated table (protocol × domain)
- `results/experiment_config.json` — configuration snapshot
- `figures/fig1 … fig6.png` — 150 DPI report figures

### 4. Launch the Streamlit demo

```bash
streamlit run app.py
```

Then open the printed `Local URL` (default http://localhost:8501) in your browser.

**Demo workflow (5 steps, all in one page):**

1. **Enter a task** in the text area — free-form English. Three example tasks are shown in the placeholder.
2. The app runs an LLM **domain classifier** and displays the predicted domain (MATH / READING / NEWS / OTHER) with a confidence score.
3. It looks up the **empirically-best protocol** for that domain from `results/results_summary.csv` and shows the rationale (completion rate, mean tokens).
4. The **three-agent pipeline runs**, rendering each agent's output (Planner → Executor → Integrator) with per-agent token and latency counts.
5. A **metrics dashboard** summarizes protocol, total tokens (prompt / completion split), wall-clock latency, and estimated USD cost. The **Final answer** is highlighted; an expandable **Raw message log** pane shows every inter-agent message for debugging.

The "Auto-select protocol" sidebar toggle lets you override the recommendation and pick a protocol manually. If `results/results_summary.csv` is missing (e.g. before step 3 is run), the app falls back to a reasonable default mapping.

**Try these to verify each route works:**

- *Math → recommends Markdown.* `Janet's ducks lay 16 eggs per day. She eats 3 for breakfast and bakes muffins with 4. She sells the rest at $2 per egg. How much does she make?` (expected answer: 18)
- *Reading → recommends Shared Memory.* Paste a short passage, then end with `… Question: <your question>?`. The app splits on the last `?` — passage before, question after.
- *News → recommends Shared Memory.* Paste a paragraph-length article and end with `Summarize the key facts.`

## Experimental design

| | |
|---|---|
| **Agent pipeline** | Planning → Execution → Integration (fixed roles, fixed system prompts) |
| **Independent variable** | Inter-agent communication protocol (4 levels: NL, Markdown, JSON, Shared Memory) |
| **Task domains** | GSM8K (math), SQuAD (reading), 10 curated news articles |
| **Samples × reps × protocols × domains** | 10 × 3 × 4 × 3 = 360 pipeline runs |
| **Model** | OpenAI `gpt-4o-mini`, temperature 0.3, per-rep `seed` for best-effort determinism |
| **Max tokens** | 256 (math, reading), 512 (news) |
| **Evaluation** | Numeric exact match (math) · SQuAD token-F1 (reading) · mean of ROUGE-2 F1 and ROUGE-L F1 (news) |
| **Statistics** | Two-way ANOVA with η² · per-domain one-way ANOVA · Tukey HSD · Cohen's d · 2000-resample bootstrap 95% CIs |

### Protocol implementations

- **NL** — explicit `"plain English prose, no markdown / bullets / JSON"` instruction suffix to prevent the model from defaulting to markdown-style output.
- **Markdown** — headings, bullet points, numbered lists.
- **JSON** — OpenAI `response_format={"type": "json_object"}` + descriptive field names. A single parse-retry is attempted on `JSONDecodeError`; remaining failures are logged as `json_parse_error=True`.
- **Shared Memory** — a true blackboard: every agent is injected the full JSON snapshot of the shared state, so downstream agents' input tokens grow as the state accumulates. This is what makes H1 ("Shared Memory incurs higher overhead") actually testable.

### Evaluators

Chosen to avoid false positives from negation or keyword matching:

- **Math.** Numeric exact match against the GSM8K gold answer (`####` marker + last-number fallback).
- **Reading.** SQuAD-style token-level F1 against the gold answer set. Catches cases like *"The building is not 187 feet tall"* that substring match scores as correct.
- **News.** Mean of ROUGE-2 F1 (bigram overlap) and ROUGE-L F1 (longest common subsequence) against the concatenated `key_facts` reference. Catches cases like *"S&P 500 did not rise"* that keyword coverage scores as correct.

All three evaluators are self-tested at the bottom of `pipeline.py` — run `python pipeline.py` to check.

## Reproducibility

- `OpenAI.chat.completions.create(..., seed=rep)` gives best-effort determinism per repetition.
- `random.seed(rep)` and `np.random.seed(rep)` cover any downstream sampling.
- All prompts, system messages, and configuration are defined as module-level constants in `pipeline.py` and dumped to `results/experiment_config.json` after each run.
- `results/results_messages.jsonl` records every inter-agent message (sender, receiver, content, token breakdown, latency, finish_reason, timestamp) for post-hoc case-study and error analysis.

## Rubric alignment

| Component | Weight | Deliverable |
|---|---|---|
| Proposal | 10% | `STAT_GR5293_Proposal_yz5104_xz3447_tz2704.pdf` (submitted) |
| Final Presentation | 30% | slides (see `report/REPORT_OUTLINE.md` for the narrative) |
| Final Report | 30% | see `report/REPORT_OUTLINE.md` |
| Project Demo | 20% | `app.py` (Streamlit) — auto-classifies domain, selects empirically-best protocol, streams agent messages, shows cost dashboard |
| GitHub Repo | 10% | this repository |

## Related files worth reading

- **Proposal (submitted, authoritative)** — `STAT_GR5293_Proposal_yz5104_xz3447_tz2704.pdf`
- **Rubric** — `Project_Rubric_STATGR5293_2026.pdf`
- **Limitations & Future Work notes** — `old_proposal/STAT_5293_Project_Proposal （Limitations and Future Extensions added）.pdf` §10, which is the source for the Limitations section of the final report.

## License

Academic project, not licensed for redistribution beyond course submission.
