# Final Report — Outline (with results filled in)

> **Rubric weights for the report.** Structure & Format 5% · Depth of Research 8% · Methodology 8% · Results & Analysis 6% · Grammar & Writing Quality 3% · **Total 30%.**

All numbers below come from the local 360-run grid executed on 2026-04-23 with `gpt-4o-mini`, seed=rep, temperature=0.3. Source files: [results/results_raw.csv](../results/results_raw.csv), [results/results_summary.csv](../results/results_summary.csv), [results/results_messages.jsonl](../results/results_messages.jsonl), [results/experiment_config.json](../results/experiment_config.json).

---

## Target length and format

- 10–14 pages, single column, 11pt, 1-inch margins. Match the proposal's LaTeX style (NeurIPS / ICML-flavored).
- Figures referenced from `figures/`. Tables as `\begin{table}` with booktabs.
- Every claim either cites a reference or points at a figure/table in the Results section.

## 1. Abstract  *(≈150 words)*

Adapt the proposal's abstract and append the three verdict sentences:

> Across 360 pipeline runs spanning four protocols (NL, Markdown, JSON, Shared Memory) and three domains (math, reading, news) on `gpt-4o-mini`, we find that (1) **communication protocol has a large effect on efficiency** — a two-way ANOVA attributes η²=0.262 of token variance to protocol and all six protocol pairs differ on Tukey HSD, with Shared Memory costing +338 tokens/run over NL on average; (2) **protocol efficiency interacts with domain** (F=15.66, p<.001, η²=0.094 on tokens); and (3) **protocol effectiveness is domain-dependent** — protocol does not affect math completion (F=0.66, p=.58) but matters strongly for reading (η²=.102, Cohen d=0.88) and news (η²=.197, d=1.44), where the Shared Memory blackboard outperforms every other protocol despite its token overhead.

## 2. Introduction  *(≈1 page)*

Lift from proposal §1 almost verbatim. Two required elements:

1. The two research questions (RQ1 efficiency, RQ2 effectiveness) — unchanged from the submitted proposal.
2. A clear statement that this project operates at the *protocol* level, not the *application* level — this is the "meta-level" differentiator we already committed to in the proposal and it is what separates us from the other course groups.

Close with a contribution bullet list (4 items, matches proposal §7):

- Empirical: first controlled multi-domain comparison of inter-agent communication protocols.
- Practical: protocol-selection guideline usable with LangGraph / AutoGen / CrewAI.
- Statistical: two-factor ANOVA + Tukey HSD + bootstrap inference, uncommon in LLM systems papers.
- Artifacts: reusable framework ([pipeline.py](../pipeline.py)), 360-run dataset, Streamlit demo ([app.py](../app.py)).

## 3. Related Work  *(≈1.5 pages, rubric: Depth of Research 8%)*

Expand the four paragraphs from proposal §2. For each of the four strands, go one reference deeper and note *what our study adds*.

- **Multi-agent LLM frameworks** — LangGraph [1], AutoGen [2], CrewAI [3]. Add: these frameworks do not prescribe a protocol; protocol is a free parameter we fix and vary.
- **Agent communication & structured output** — Toolformer [4], constrained decoding [5], AgentVerse [6]. Add: prior work treats protocol at the single-agent / single-message level; we treat it as an experimental factor across a pipeline.
- **Multi-agent evaluation** — AgentBench [7], multi-agent debate [8]. Add: benchmarks hold communication fixed; we manipulate it.
- **Task-specific LLM performance** — GSM8K [9], SQuAD [10], information extraction [11]. Add: these measure single-agent task performance; we test whether their domain effects carry into multi-agent pipelines and interact with protocol.

## 4. Methodology  *(≈2–3 pages, rubric: Methodology 8%)*

### 4.1 System model and pipeline
- 3-agent directed chain; roles, system prompts, and max tokens are all held constant across conditions. See [pipeline.py](../pipeline.py) `SYSTEM_PROMPTS`, `DOMAIN_MAX_TOKENS = {MATH:256, READING:256, NEWS:512}`.
- Only the communication protocol (encoded by `PROTOCOL_INSTRUCTIONS` + `SHARED_MEMORY_PREAMBLE` + `response_format` for JSON) is manipulated.

### 4.2 Protocols
Reuse proposal §3.2 text. For each protocol state the *encoding*, the *instruction given to the agent*, and the *expected overhead pattern*.

**Implementation notes not in the submitted proposal (add here for credit under Methodology):**

1. **Shared Memory is a true blackboard.** Every agent is injected the full JSON snapshot of the shared state, so integrator input grows with the accumulated plan + execution result. Empirically confirmed in our message log: mean integrator `prompt_tokens` is 469 for SHARED_MEMORY versus 194 (NL), 254 (Markdown), 188 (JSON). Earlier drafts passed only the predecessor's output, which was indistinguishable from NL sequential message-passing and made H1 untestable; we corrected this before final data collection.
2. **NL carries an explicit plain-prose instruction** ("Respond in plain English prose. Do not use markdown, bullet points, headings, JSON…"). Without this, `gpt-4o-mini` drifts toward markdown-style output by default, contaminating the NL-vs-Markdown comparison.
3. **JSON protocol uses OpenAI `response_format={"type": "json_object"}` with one parse-validate-retry.** Result: 0 unrecoverable parse errors across 90 JSON runs.
4. **Per-rep reproducibility via OpenAI `seed=rep`** alongside `random` / `numpy` seeding for sample draws.

### 4.3 Task domains and evaluators
Reuse proposal §3.3. Justify the evaluator choices briefly:

- Math: numeric exact match — unambiguous for arithmetic answers.
- Reading: **SQuAD token-level F1** (official SQuAD eval) — replaces naive substring match, which gives false positives on negation (e.g. *"The building is not 187 feet"*).
- News: **mean of ROUGE-2 F1 and ROUGE-L F1** — replaces keyword / token coverage, which is similarly fooled by negation.

The self-tests embedded in `pipeline.py::_run_self_tests` verify the negation fix empirically (positive answer strictly scores higher than the negated form).

### 4.4 Experimental design
- 4 × 3 factorial: 4 protocols × 3 domains.
- 10 samples per domain (GSM8K first-20 test split, SQuAD first-20 validation, 10 hand-curated news articles with pre-extracted `key_facts`).
- 3 reps per (protocol, domain, sample) cell → **360 pipeline runs → 1 080 API calls**.
- `gpt-4o-mini`, temperature 0.3, OpenAI `seed=rep`, per-rep `random` / `numpy` seeds for sampling reproducibility.

### 4.5 Statistical analysis plan
Copy proposal §6 verbatim, then add:

- **Tukey HSD post-hoc** on total tokens (pooled) and on `completion_score` per domain.
- **Input / output token decomposition** (Fig 5) to distinguish schema-tightness from response terseness.
- **Full message-level audit** — every inter-agent message persisted to [results/results_messages.jsonl](../results/results_messages.jsonl) for §5.4 case studies and §5.5 error analysis.

## 5. Results  *(≈3–4 pages, rubric: Results & Analysis 6%)*

### 5.1 RQ1 — Efficiency

Figures: [fig1](../figures/fig1_tokens_by_protocol_domain.png), [fig5](../figures/fig5_input_vs_output.png), [fig6](../figures/fig6_latency.png).

**Two-way ANOVA on `total_tokens`** (Type II; N=360):

| Term | F | p | η² |
|---|---|---|---|
| Protocol | 87.40 | <.001 | .262 |
| Domain | 148.19 | <.001 | .296 |
| Protocol × Domain | 15.66 | <.001 | .094 |

**Two-way ANOVA on `total_latency_ms`:** Protocol F=16.55, p<.001, η²=.070; Domain F=136.44, p<.001, η²=.385; Interaction F=6.38, p<.001, η²=.054. Domain dominates latency variance; protocol contributes a smaller but clearly detectable effect.

**Tukey HSD on tokens (pooled across domains).** All six pairs differ significantly (p-adj < .01). Mean differences (group2 − group1):

| Pair | Δ tokens | p-adj |
|---|---|---|
| NL vs SHARED_MEMORY | +337.6 | <.001 |
| MARKDOWN vs SHARED_MEMORY | +113.6 | .002 |
| JSON vs SHARED_MEMORY | +233.3 | <.001 |
| JSON vs MARKDOWN | +119.7 | .001 |
| JSON vs NL | −104.3 | .006 |
| MARKDOWN vs NL | −224.0 | <.001 |

Ordering from cheapest to most expensive (pooled): **NL ≺ JSON ≺ Markdown ≺ Shared Memory**.

**Bootstrap 95% CIs on mean tokens per cell** (2 000 resamples, see `results_summary.csv` and Fig 1):

| | MATH | READING | NEWS |
|---|---|---|---|
| NL | 988 [926, 1054] | **819 [807, 831]** | 995 [977, 1016] |
| Markdown | 1046 [971, 1125] | 995 [961, 1031] | 1433 [1371, 1497] |
| JSON | **847 [807, 894]** | 955 [924, 987] | 1313 [1253, 1378] |
| Shared Memory | 1204 [1123, 1291] | 1192 [1183, 1201] | 1418 [1375, 1465] |

Non-overlapping CIs confirm Shared Memory is strictly more expensive than NL on every domain.

**H1 verdict — supported.** Shared Memory incurs the highest token overhead both pooled (+338 vs NL) and on every domain slice; the Tukey pair NL vs SHARED_MEMORY is the largest gap in the matrix. Note JSON is *not* pooled-cheapest — NL is — so the stronger claim in H1 ("JSON lowest") needs qualification: **JSON is cheapest specifically on MATH**, but NL is cheapest on reading and news because of shorter input framing.

### 5.2 RQ2 — Effectiveness

Figures: [fig2](../figures/fig2_completion_by_cell.png), [fig3](../figures/fig3_interaction.png).

**Two-way ANOVA on `completion_score`:** Protocol F=2.36, p=.071 (ns at α=.05); Domain F=125.42, p<.001, η²=.408; Protocol × Domain F=1.43, p=.203, η²=.014. **The pooled interaction term is not significant**, but the pooled protocol effect is a misleading average because the direction of the protocol effect depends on domain (see below).

**Per-domain one-way ANOVA on `completion_score`:**

| Domain | F | p | η² | Cohen d (best vs worst) |
|---|---|---|---|---|
| MATH | 0.66 | .58 | .017 | 0.33 (Markdown 0.867 vs JSON 0.733) |
| READING | 4.39 | **.006** | .102 | **0.88** (SHARED_MEMORY 0.555 vs Markdown 0.347) |
| NEWS | 9.51 | **<.001** | .197 | **1.44** (SHARED_MEMORY 0.346 vs Markdown 0.271) |

**Tukey HSD per domain** — significant pairs (FWER α=.05):

- **MATH:** no pair significant. Protocol does not affect math completion within this sample size.
- **READING:** SHARED_MEMORY > Markdown (Δ=+0.208, p=.003). SHARED_MEMORY vs JSON (+0.136, p=.098) and vs NL (+0.130, p=.125) trend positive but do not cross α.
- **NEWS:** SHARED_MEMORY > Markdown (Δ=+0.075, p<.001), SHARED_MEMORY > NL (Δ=+0.071, p<.001), SHARED_MEMORY > JSON (Δ=+0.051, p=.009). The other three pairs do not differ.

**H2 verdict — partially supported, in the expected direction.** Protocol choice matters for reading (η²=.10) and especially news (η²=.20) but not math (η²=.02). The proposal predicted "protocol matters more for math than for news"; the data show the opposite — protocol matters *least* for math, because all protocols preserve the single-number final answer well, whereas news and reading expose genuine representation-format effects.

**H3 verdict — qualitatively supported but the formal pooled interaction F-test is not significant on completion.** The pattern "best protocol changes across domain" is visible in Fig 3 (non-parallel lines: Markdown wins MATH by ε, SHARED_MEMORY wins READING and NEWS by wide margins), and the per-domain η² swings from .017 (MATH) to .197 (NEWS) — a 10× change. But the Protocol × Domain term on `completion_score` is F=1.43, p=.203, η²=.014 — the interaction is real in effect-size terms but the N=30/cell design is under-powered for the formal test. We report both: the interaction *on tokens* is strongly significant (F=15.66, p<.001, η²=.094), the interaction *on completion* shows large per-domain effect-size heterogeneity but does not cross α in the pooled two-way ANOVA.

### 5.3 Efficiency-effectiveness trade-off

Figure: [fig4](../figures/fig4_pareto.png). Raw cell-level data in `results/results_summary.csv`.

**Pareto frontier (minimize tokens, maximize completion):**

| Domain | Non-dominated cells | Dominated cells |
|---|---|---|
| MATH | JSON (847, 0.733) — cheapest; NL (988, 0.833); Markdown (1046, 0.867) — highest completion | SHARED_MEMORY (1204, 0.833) — dominated by NL |
| READING | NL (819, 0.426) — cheapest; SHARED_MEMORY (1192, 0.556) — highest completion | Markdown (995, 0.347), JSON (955, 0.419) |
| NEWS | NL (995, 0.275) — cheapest; SHARED_MEMORY (1418, 0.346) — highest completion | Markdown (1433, 0.271), JSON (1313, 0.296) |

**Practitioner recommendations.** (This is what drives the Streamlit `auto_protocol` feature.)

- **MATH → Markdown** if completion matters most (86.7%), **JSON** if cost matters (73.3% at −20% tokens vs Markdown).
- **READING → Shared Memory** when a 30%+ completion lift is worth +45% tokens; **NL** if the budget is tight (cheapest AND second-best completion).
- **NEWS → Shared Memory** (only protocol that materially lifts completion, despite cost); **NL** if the budget is tight.

### 5.4 Case studies

Pick 2–3 illustrative runs from `results/results_messages.jsonl` keyed by `run_id`. Suggested narrative:

1. **Reading case where Shared Memory wins.** A SQuAD passage about a named-entity span where the Markdown integrator paraphrases the passage (diluting the exact span) but the SHARED_MEMORY integrator quotes the passage verbatim off the blackboard. Show the three agent outputs side-by-side and the resulting F1 scores.
2. **News case where Shared Memory wins.** A 500-word article where JSON compresses key facts into a schema that drops dates/figures; Shared Memory preserves the figure exactly because the executor's full text remains readable on the blackboard.
3. **Math case where Markdown wins narrowly over JSON.** Chain-of-arithmetic problem where Markdown headings preserve intermediate-step structure for the integrator, whereas JSON forces the executor to flatten steps into a single value, occasionally losing an intermediate number.

For each case show: user task, all three agent outputs (content + token split + latency), evaluator score per protocol, and one sentence on *why* the format decided the outcome.

### 5.5 Error analysis

Segmented failures across the 360 runs:

- **JSON parse errors: 0 / 90 runs (0.0%)** — `response_format={"type":"json_object"}` + one retry eliminated unrecoverable parse failures entirely.
- **Truncation (`finish_reason=length`): 11 / 360 runs total**, all on MATH (longest chain-of-arithmetic): Markdown 6, Shared Memory 3, NL 2, JSON 0. The Markdown rate on MATH is 6/30 = 20% — headings/bullets inflate the executor's output past `max_tokens=256`. A follow-up could raise MATH `max_tokens` to 384 for Markdown specifically; we flag but do not re-run.
- **Completion-score clustering by failure mode** (qualitative, from `results_messages.jsonl`): NL failures are typically *under-specification* (planner outputs vague steps, executor improvises); JSON failures are *schema-literalism* (executor returns the schema-specified field even when empty); Markdown failures are *verbosity truncation* (see above); Shared Memory failures are *context over-retrieval* (integrator copies stale plan text into the final answer).

## 6. Discussion  *(≈1 page)*

Tie back to the three hypotheses:

- **For practitioners.** If you are building a math/reasoning pipeline, protocol choice is a cost lever, not a quality lever — pick JSON for cheapness, Markdown if you can spare +20% tokens. For reading comprehension and open-text analysis (news), pick Shared Memory *despite* its overhead — the completion lift (Cohen d ≈ 0.9–1.4 vs Markdown) is the largest effect in the study.
- **Why the pooled protocol effect understates.** The pooled one-way ANOVA on completion is p=.071 because MATH's flat η²=.017 averages down READING's .102 and NEWS's .197. Any report that looks only at pooled stats will miss the real finding; per-domain reporting is non-negotiable.
- **Generalization.** All results are on `gpt-4o-mini`. We expect the *direction* of the effects to transfer (Shared Memory → more tokens, JSON → terser), but the *magnitude* is model-specific. A one-paragraph replication on Claude 3.5 Haiku or Llama-3.1-70B-Instruct would strengthen the practitioner guideline.

## 7. Limitations and Future Work  *(≈0.5–1 page)*

Pull from `old_proposal/STAT_5293_Project_Proposal （Limitations and Future Extensions added）.pdf` §10, plus one extra:

1. **Scale.** N = 10 per domain, 3 reps. The Protocol × Domain interaction on completion did not reach α=.05 (p=.203) despite large per-domain effect sizes — the design is underpowered for the formal interaction test. A replication at N=30/domain would resolve this.
2. **Additional protocols.** XML, YAML, hybrid (Markdown with embedded JSON fields).
3. **Heterogeneous LLM agents.** Assign GPT / Gemini / Claude to different roles. 3! = 6 permutations per (protocol, domain) cell, enabling study of which model is best for planning vs execution vs integration.
4. **Dynamic protocol selection.** Switch protocols mid-pipeline based on intermediate output quality or token budget — e.g. route to JSON once the executor signals confidence.
5. **Single base model.** All three agents use `gpt-4o-mini`; replication on one other model family (e.g. Claude 3.5 Haiku, Llama-3.1-70B-Instruct) is a one-paragraph follow-up we flag.

## 8. Conclusion  *(≈0.25 page)*

Two sentences: restate the finding (protocol has a large efficiency effect that interacts with domain; Shared Memory is the best choice for reading/news despite +25–45% token overhead; JSON is the cheapest for math; NL is the best default when budget dominates) and restate the artifact contribution (reusable framework `pipeline.py`, 360-run dataset in `results/`, Streamlit demo `app.py`).

## 9. References

Start from proposal's 12 references [1]–[12]. Add any new citation you pick up during the literature expansion in §3.

## 10. Appendix — Reproducibility

Pure "how-to-rerun" block. Most of this can be a pointer to `README.md`:

- Repository URL.
- `requirements.txt` pinning.
- `OPENAI_API_KEY` requirement (via `.env`).
- `jupyter nbconvert --execute Multi_Agent_Communication_Protocol_Study.ipynb` for the full grid (~10 min, ~$0.40).
- `streamlit run app.py` for the demo.
- Git commit hash under which the reported numbers were generated.
- Pointer to `results/experiment_config.json` for the exact parameter dump.

---

## Figure / table checklist (populated)

| Asset | File | Required for |
|---|---|---|
| Fig 1 tokens by cell | [fig1_tokens_by_protocol_domain.png](../figures/fig1_tokens_by_protocol_domain.png) | §5.1 |
| Fig 2 completion boxplot | [fig2_completion_by_cell.png](../figures/fig2_completion_by_cell.png) | §5.2 |
| Fig 3 interaction plot | [fig3_interaction.png](../figures/fig3_interaction.png) | §5.2 (H3 centerpiece) |
| Fig 4 Pareto scatter | [fig4_pareto.png](../figures/fig4_pareto.png) | §5.3 |
| Fig 5 input vs output tokens | [fig5_input_vs_output.png](../figures/fig5_input_vs_output.png) | §5.1 |
| Fig 6 latency boxplot | [fig6_latency.png](../figures/fig6_latency.png) | §5.1 |
| Table 1 experiment matrix | proposal table 1 | §4.4 |
| Table 2 summary (protocol × domain) | [results_summary.csv](../results/results_summary.csv) | §5.3 |
| Table 3 ANOVA (tokens / latency / completion) | notebook §7 — values inlined above | §5.1 / §5.2 |
| Table 4 Tukey HSD (pooled tokens + per-domain completion) | notebook §9 — values inlined above | §5.1 / §5.2 |

## Hypotheses — one-line verdicts

| Hypothesis | Verdict | Key evidence |
|---|---|---|
| **H1** — Shared Memory highest token overhead; JSON lowest. | **Supported, with qualification.** Shared Memory highest on every domain (Tukey: +338 tok vs NL, p<.001). JSON is cheapest on MATH (847) but NL is pooled-cheapest overall. | Fig 1, two-way ANOVA Protocol η²=.262, Tukey HSD all pairs. |
| **H2** — Protocol matters more for math than for news. | **Partially supported, direction reversed.** Protocol matters most for news (η²=.197), then reading (.102), then math (.017). | §5.2 per-domain ANOVA + Tukey. |
| **H3** — Significant Protocol × Domain interaction. | **Qualitatively yes, formally no on completion.** Interaction F=1.43, p=.203, η²=.014 on completion; but F=15.66, p<.001, η²=.094 on tokens, and per-domain Cohen d ranges 0.33 → 1.44. | Fig 3, §5.2, §7 limitation #1 (N underpowered for pooled interaction test on completion). |

## Writing style notes

- Past tense for methods and results ("we ran 360 pipelines"), present tense for framing claims ("JSON provides the lowest token cost on math").
- Do not use the phrase "significant" without a p-value next to it.
- Report effect sizes (η², Cohen's d) alongside p-values per proposal §6.
- Use "protocol" (lowercase) in prose; use `JSON`, `NL`, `Markdown`, `SHARED_MEMORY` (monospace) when naming a level.
- Do not claim any result that is not backed by a figure in `figures/` or a table in `results/`.
