"""
Streamlit dashboard — Multi-Agent Communication Protocol Study.

This app is both an experiment walkthrough and a live demo:
  1. Explain the research question and fixed three-agent system.
  2. Show the 4 x 3 x 10 x 3 experimental design.
  3. Compare protocols using saved experiment results and figures.
  4. Browse real inter-agent messages from the 360-run log.
  5. Run a new user task through Planner -> Executor -> Integrator.

Run: `streamlit run app.py`
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI
from scipy import stats

from experiment_utils import MODEL_PRICING_PER_1M
from pipeline import (
    FULL_FACTORIAL_PROTOCOLS,
    MAIN_PROTOCOLS,
    PROTOCOL_FORMAT,
    PROTOCOL_MECHANISM,
    Protocol,
    TaskDomain,
    run_pipeline,
)


RESULTS_SUMMARY = Path("results/results_summary.csv")
RESULTS_RAW = Path("results/results_raw.csv")
RESULTS_MESSAGES = Path("results/results_messages.jsonl")
EXPERIMENT_CONFIG = Path("results/experiment_config.json")
RESULTS_ABLATION_SM_JSON = Path("results/results_ablation_sm_json.csv")
RESULTS_ABLATION_FULL = Path("results/results_ablation_full_factorial.csv")
RESULTS_DEEPSEEK = Path("results/results_deepseek_v4_flash_8protocols.csv")
FIGURES_DIR = Path("figures")
COST_PER_1M = MODEL_PRICING_PER_1M


PROTOCOL_DETAILS = {
    "RELAY_DEFAULT": {
        "label": "Relay + Default",
        "short": "Sequential handoff with no explicit format suffix.",
        "implementation": "Planner output is passed to Executor, then Executor output to Integrator; no NL/Markdown/JSON instruction is appended.",
        "strength": "Completes the default-format relay cell for the 2x4 ablation matrix.",
        "risk": "LLMs may drift toward their own default markdown-ish formatting.",
    },
    "NL": {
        "label": "Natural Language",
        "short": "Plain prose between agents.",
        "implementation": "Adds an explicit plain-English instruction and forbids bullets, Markdown, JSON, and headings.",
        "strength": "Low framing overhead; good budget baseline.",
        "risk": "Can under-specify intermediate state and leaves structure implicit.",
    },
    "MARKDOWN": {
        "label": "Markdown",
        "short": "Headings and bullet lists.",
        "implementation": "Asks agents to use headings, bullets, and numbered lists for intermediate outputs.",
        "strength": "Readable, good for math work traces, easy to inspect in demos.",
        "risk": "More verbose completion tokens; can hit max-token limits on longer reasoning.",
    },
    "JSON": {
        "label": "JSON",
        "short": "Validated structured objects.",
        "implementation": 'Uses OpenAI response_format={"type": "json_object"} and validates parseability with one retry.',
        "strength": "Compact and machine-readable; cheapest protocol on math in this run.",
        "risk": "Schema-like outputs can compress away useful nuance in reading/news tasks.",
    },
    "SHARED_MEMORY": {
        "label": "Shared Memory",
        "short": "Blackboard state shared across agents with default output format.",
        "implementation": "Injects the full accumulated JSON blackboard snapshot into downstream agents.",
        "strength": "Best quality on reading and news because context persists across the whole chain.",
        "risk": "Highest prompt-token overhead because state grows at each step.",
    },
    "SHARED_MEMORY_NL": {
        "label": "Shared Memory + NL",
        "short": "Blackboard mechanism with explicit plain-English output.",
        "implementation": "Injects the full blackboard snapshot and appends the same NL format instruction used by the relay NL condition.",
        "strength": "Completes the NL-format mechanism comparison: NL vs SHARED_MEMORY_NL.",
        "risk": "Plain prose may make intermediate state less machine-readable.",
    },
    "SHARED_MEMORY_MARKDOWN": {
        "label": "Shared Memory + Markdown",
        "short": "Blackboard mechanism with Markdown output.",
        "implementation": "Injects the full blackboard snapshot and appends the same Markdown format instruction used by the relay Markdown condition.",
        "strength": "Completes the Markdown-format mechanism comparison: MARKDOWN vs SHARED_MEMORY_MARKDOWN.",
        "risk": "Combines state serialization with verbose formatting.",
    },
    "SHARED_MEMORY_JSON": {
        "label": "Shared Memory + JSON (ablation)",
        "short": "Blackboard mechanism with forced JSON output; the original H1 ablation cell.",
        "implementation": "Combines the blackboard preamble with response_format=json_object enforcement.",
        "strength": "Cleanly isolates the cost of the blackboard mechanism alone (vs. JSON-relay) for H1 testing.",
        "risk": "Pure ablation protocol — not intended for general use; layers JSON verbosity on top of state injection.",
    },
}


DOMAIN_LABELS = {
    "MATH": "GSM8K math reasoning",
    "READING": "SQuAD reading comprehension",
    "NEWS": "Curated news analysis",
}


RECOMMENDATIONS = pd.DataFrame(
    [
        {"Domain": "MATH", "Quality first": "MARKDOWN", "Cost first": "JSON", "Why": "Markdown preserves arithmetic steps; JSON is the cheapest math cell."},
        {"Domain": "READING", "Quality first": "SHARED_MEMORY", "Cost first": "NL", "Why": "Shared Memory keeps passage evidence available; NL is cheapest and second-best."},
        {"Domain": "NEWS", "Quality first": "SHARED_MEMORY", "Cost first": "NL", "Why": "The blackboard preserves facts and figures; NL is the budget choice."},
    ]
)


st.set_page_config(
    page_title="Multi-Agent Protocol Study",
    page_icon="MA",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
      .main .block-container { padding-top: 1.6rem; max-width: 1280px; }
      div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 0.8rem 0.9rem;
      }
      .hero {
        background: linear-gradient(135deg, #f8f7f4 0%, #eef5f0 100%);
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 1.25rem 1.4rem;
        margin-bottom: 1rem;
      }
      .section-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 1rem 1.1rem;
        height: 100%;
      }
      .muted { color: #6b7280; }
      .smallcaps {
        color: #6b7280;
        font-size: 0.76rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
      }
      .flow-node {
        background: #ffffff;
        border: 1px solid #d1d5db;
        border-radius: 8px;
        padding: 0.75rem;
        text-align: center;
        min-height: 92px;
      }
      .flow-arrow {
        text-align: center;
        color: #6b7280;
        font-size: 1.3rem;
        padding-top: 1.6rem;
      }
      code { white-space: pre-wrap; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_summary() -> Optional[pd.DataFrame]:
    if not RESULTS_SUMMARY.exists():
        return None
    return pd.read_csv(RESULTS_SUMMARY)


@st.cache_data
def load_raw() -> Optional[pd.DataFrame]:
    if not RESULTS_RAW.exists():
        return None
    return pd.read_csv(RESULTS_RAW)


@st.cache_data
def load_messages() -> pd.DataFrame:
    if not RESULTS_MESSAGES.exists():
        return pd.DataFrame()
    records = []
    with RESULTS_MESSAGES.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return pd.DataFrame(records)


@st.cache_data
def load_config() -> dict:
    if not EXPERIMENT_CONFIG.exists():
        return {}
    with EXPERIMENT_CONFIG.open("r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_optional_csv(path: str) -> Optional[pd.DataFrame]:
    csv_path = Path(path)
    if not csv_path.exists():
        return None
    return pd.read_csv(csv_path)


def add_protocol_axes(df: pd.DataFrame, protocol_col: str = "protocol") -> pd.DataFrame:
    out = df.copy()
    out["Mechanism"] = out[protocol_col].map(
        {p.value: PROTOCOL_MECHANISM[p] for p in FULL_FACTORIAL_PROTOCOLS}
    )
    out["Format"] = out[protocol_col].map(
        {p.value: PROTOCOL_FORMAT[p] for p in FULL_FACTORIAL_PROTOCOLS}
    )
    return out


def combined_full_factorial(raw_df: pd.DataFrame, ablation_df: pd.DataFrame) -> pd.DataFrame:
    main = raw_df[raw_df["protocol"].isin([p.value for p in MAIN_PROTOCOLS])][
        ["protocol", "task_domain", "total_tokens", "completion_score"]
    ]
    supplemental = ablation_df[
        ["protocol", "task_domain", "total_tokens", "completion_score"]
    ]
    return add_protocol_axes(pd.concat([main, supplemental], ignore_index=True))


def mechanism_effect_rows(df: pd.DataFrame, domain: str | None = None) -> pd.DataFrame:
    rows = []
    sub_df = df if domain is None else df[df["task_domain"] == domain]
    for fmt in ["Default", "NL", "Markdown", "JSON"]:
        relay = sub_df[(sub_df["Mechanism"] == "Relay") & (sub_df["Format"] == fmt)]["total_tokens"].values
        shared = sub_df[
            (sub_df["Mechanism"] == "Shared Memory") & (sub_df["Format"] == fmt)
        ]["total_tokens"].values
        if len(relay) == 0 or len(shared) == 0:
            continue
        diff = shared.mean() - relay.mean()
        pooled = np.sqrt((relay.var() + shared.var()) / 2)
        d = diff / pooled if pooled > 0 else 0
        _, p_val = stats.ttest_ind(shared, relay, equal_var=False)
        rows.append({
            "Domain": domain or "ALL",
            "Format": fmt,
            "Relay tokens": relay.mean(),
            "Shared Memory tokens": shared.mean(),
            "Mechanism Δ tokens": diff,
            "Cohen's d": d,
            "p": p_val,
        })
    return pd.DataFrame(rows)


def best_protocol(summary: pd.DataFrame, domain: str) -> tuple[str, float, float]:
    """Return (best_protocol, completion_rate, mean_tokens) for a domain."""
    sub = summary[summary["Domain"] == domain]
    row = sub.sort_values(["Completion Rate", "Mean Tokens"], ascending=[False, True]).iloc[0]
    return row["Protocol"], row["Completion Rate"], row["Mean Tokens"]


def cheapest_protocol(summary: pd.DataFrame, domain: str) -> tuple[str, float, float]:
    sub = summary[summary["Domain"] == domain]
    row = sub.sort_values(["Mean Tokens", "Completion Rate"], ascending=[True, False]).iloc[0]
    return row["Protocol"], row["Completion Rate"], row["Mean Tokens"]


def classify_domain(client: OpenAI, model: str, task_text: str) -> tuple[str, float]:
    """Ask the LLM to classify the task. Returns (domain, confidence 0-1)."""
    classifier_prompt = (
        "Classify the following user task into exactly one category:\n"
        "  MATH     - numerical / arithmetic / word problems with a number answer\n"
        "  READING  - question answering where the answer is a short span from a passage\n"
        "  NEWS     - article summarization or factual analysis of a news-style passage\n"
        "  OTHER    - anything that clearly does not fit the above\n\n"
        'Respond with JSON: {"domain": "...", "confidence": 0.0-1.0}.\n\n'
        f"TASK:\n{task_text}"
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a concise task classifier."},
            {"role": "user", "content": classifier_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
        max_tokens=60,
    )
    try:
        parsed = json.loads(resp.choices[0].message.content)
        domain = parsed.get("domain", "OTHER").upper()
        conf = float(parsed.get("confidence", 0.5))
    except (json.JSONDecodeError, ValueError, TypeError):
        domain, conf = "OTHER", 0.5
    if domain not in {"MATH", "READING", "NEWS", "OTHER"}:
        domain = "OTHER"
    return domain, conf


def build_sample(domain: str, task_text: str) -> tuple[TaskDomain, dict]:
    """Coerce free text into the sample shape expected by each evaluator."""
    if domain == "MATH":
        return TaskDomain.MATH, {"question": task_text, "answer": ""}
    if domain == "READING":
        marker = "Question:"
        if marker in task_text:
            ctx, q = task_text.rsplit(marker, 1)
            q = q.strip() or "Answer the question."
        elif "?" in task_text:
            ctx, q = task_text.rsplit("?", 1)
            q = (q.strip() + "?") if q.strip() else "Answer the question."
        else:
            ctx, q = task_text, "Answer based on the passage."
        return TaskDomain.READING, {"context": ctx.strip(), "question": q.strip(), "answers": []}
    return TaskDomain.NEWS, {"title": "User input", "content": task_text, "key_facts": []}


def metric_row(summary: Optional[pd.DataFrame], raw: Optional[pd.DataFrame]) -> None:
    total_runs = int(raw.shape[0]) if raw is not None else 360
    protocols = int(summary["Protocol"].nunique()) if summary is not None else 4
    domains = int(summary["Domain"].nunique()) if summary is not None else 3
    total_messages = int(total_runs * 3)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pipeline runs", f"{total_runs:,}")
    c2.metric("Protocols", protocols)
    c3.metric("Task domains", domains)
    c4.metric("Message logs", f"{total_messages:,}")


def show_image(path: Path, caption: str) -> None:
    if path.exists():
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.info(f"Missing figure: `{path}`")


def render_flow() -> None:
    cols = st.columns([1.2, 0.25, 1.1, 0.25, 1.25, 0.25, 1.1, 0.25, 1.2])
    nodes = [
        ("Task data", "GSM8K, SQuAD, curated news"),
        ("Protocol", "NL, Markdown, JSON, Shared Memory"),
        ("Agents", "Planner -> Executor -> Integrator"),
        ("Logs", "Tokens, latency, final answer, message content"),
        ("Analysis", "ANOVA, Tukey HSD, effect sizes, bootstrap CIs"),
    ]
    for i, col in enumerate(cols):
        if i % 2 == 0:
            title, body = nodes[i // 2]
            col.markdown(
                f"<div class='flow-node'><div class='smallcaps'>{title}</div><strong>{body}</strong></div>",
                unsafe_allow_html=True,
            )
        else:
            col.markdown("<div class='flow-arrow'>-></div>", unsafe_allow_html=True)


def render_protocol_card(key: str, detail: dict) -> None:
    st.markdown(
        f"""
        <div class="section-card">
          <div class="smallcaps">{key}</div>
          <h4 style="margin: 0.2rem 0 0.35rem 0;">{detail["label"]}</h4>
          <p class="muted">{detail["short"]}</p>
          <p><strong>Implementation:</strong> {detail["implementation"]}</p>
          <p><strong>Best use:</strong> {detail["strength"]}</p>
          <p><strong>Watch out:</strong> {detail["risk"]}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_live_demo(summary: Optional[pd.DataFrame]) -> None:
    st.subheader("Run the Pipeline on a New Task")
    st.caption("This is the original demo runner, now embedded inside the full experiment dashboard.")

    api_key = st.session_state.get("api_key", "")
    model = st.session_state.get("model", "gpt-4o-mini")
    auto_protocol = st.checkbox("Auto-select protocol from experiment results", value=True)
    manual_protocol = st.selectbox(
        "Manual protocol override",
        [p.value for p in Protocol],
        index=[p.value for p in Protocol].index("JSON"),
        disabled=auto_protocol,
    )

    examples = {
        "Math": "Janet has 24 apples. She gives 6 to her friend, buys 12 more, and then sells half of the total for $3 each. How much money does she make?",
        "Reading": "The Eiffel Tower is a wrought-iron lattice tower in Paris, France. It was named after engineer Gustave Eiffel, whose company designed and built the tower. Question: Who was the Eiffel Tower named after?",
        "News": "A city council approved a $42 million transit plan on Monday, adding 18 electric buses and three new rapid routes by 2027. Summarize the key facts.",
    }
    example_name = st.selectbox("Load an example", ["Custom"] + list(examples))
    default_text = examples.get(example_name, "")
    task_text = st.text_area("Enter your task", value=default_text, height=150)

    col_run, col_note = st.columns([1, 4])
    run_clicked = col_run.button("Run pipeline", type="primary", disabled=not (api_key and task_text.strip()))
    col_note.caption("The run will classify the domain, select a protocol, run three agents, and report cost/latency.")

    if not run_clicked:
        if not api_key:
            st.info("Add your OpenAI API key in the sidebar to enable live runs.")
        return

    client = OpenAI(api_key=api_key)

    with st.status("Step 1 - classifying task domain", expanded=False) as status:
        try:
            domain_str, conf = classify_domain(client, model, task_text)
        except Exception as exc:
            st.error(f"Classification failed: {exc}")
            st.stop()
        status.update(label=f"Step 1 complete - domain = {domain_str} (confidence {conf:.2f})", state="complete")

    fallback_recs = {
        "MATH": ("JSON", None, None),
        "READING": ("JSON", None, None),
        "NEWS": ("NL", None, None),
        "OTHER": ("NL", None, None),
    }

    if auto_protocol:
        if summary is not None and domain_str in summary["Domain"].values:
            proto_name, completion_rate, mean_tokens = best_protocol(summary, domain_str)
            rationale = (
                f"{proto_name} recommended for {domain_str}: mean completion "
                f"{completion_rate:.3f}, about {mean_tokens:.0f} tokens/run."
            )
        else:
            proto_name, _, _ = fallback_recs.get(domain_str, fallback_recs["OTHER"])
            rationale = f"{proto_name} fallback recommendation because summary data is unavailable."
    else:
        proto_name = manual_protocol
        rationale = f"{proto_name} selected manually."

    st.info(f"Step 2 - protocol: **{proto_name}**. {rationale}")

    protocol = Protocol(proto_name)
    task_domain, sample = build_sample(domain_str, task_text)

    t_start = time.time()
    try:
        with st.spinner("Running Planner -> Executor -> Integrator"):
            result, msgs = run_pipeline(
                protocol,
                task_domain,
                sample,
                sample_idx=0,
                client=client,
                model=model,
                seed=0,
            )
    except Exception as exc:
        st.error(f"Pipeline failed: {exc}")
        st.stop()
    elapsed = time.time() - t_start

    by_sender = {m.sender: m for m in msgs}
    st.subheader("Agent Messages")
    agent_cols = st.columns(3)
    for col, sender, label in zip(agent_cols, ["Planner", "Executor", "Integrator"], ["Planner", "Executor", "Integrator"]):
        if sender in by_sender:
            msg = by_sender[sender]
            with col:
                st.markdown(f"**{label}**")
                st.caption(f"{msg.total_tokens} tokens | {msg.latency_ms:.0f} ms")
                st.code(str(msg.content), language="markdown")

    prices = COST_PER_1M.get(model, COST_PER_1M["gpt-4o-mini"])
    cost_usd = (
        result.total_prompt_tokens / 1e6 * prices["prompt"]
        + result.total_completion_tokens / 1e6 * prices["completion"]
    )
    st.subheader("Run Metrics")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Protocol", proto_name)
    c2.metric("Total tokens", f"{result.total_tokens:,}", f"{result.total_prompt_tokens}p + {result.total_completion_tokens}c")
    c3.metric("Latency", f"{result.total_latency_ms:,.0f} ms", f"{elapsed:.1f}s wall")
    c4.metric("Est. cost", f"${cost_usd:.5f}")
    c5.metric("Domain confidence", f"{conf:.2f}")

    if result.any_truncation:
        st.warning("At least one agent hit finish_reason=length; output may be truncated.")
    if result.any_json_parse_error:
        st.warning("JSON protocol output failed to parse even after retry.")

    st.subheader("Final Answer")
    st.success(str(by_sender["Integrator"].content) if "Integrator" in by_sender else "(no integrator output)")

    with st.expander("Raw message log"):
        for m in msgs:
            st.markdown(
                f"**{m.sender} -> {m.receiver}** | {m.total_tokens} tokens "
                f"({m.prompt_tokens}p + {m.completion_tokens}c) | "
                f"{m.latency_ms:.0f} ms | finish_reason={m.finish_reason}"
            )
            st.text(str(m.content))
            st.markdown("---")


summary = load_summary()
raw = load_raw()
messages = load_messages()
config = load_config()
ablation_sm_json = load_optional_csv(str(RESULTS_ABLATION_SM_JSON))
ablation_full = load_optional_csv(str(RESULTS_ABLATION_FULL))
deepseek_results = load_optional_csv(str(RESULTS_DEEPSEEK))

with st.sidebar:
    st.title("Dashboard Controls")
    st.session_state["api_key"] = st.text_input(
        "OpenAI API key",
        value=os.environ.get("OPENAI_API_KEY", ""),
        type="password",
        help="Stored only in this Streamlit session. You can also set OPENAI_API_KEY in your shell.",
    )
    st.session_state["model"] = st.selectbox("Live demo model", ["gpt-4o-mini", "gpt-4o"], index=0)
    st.markdown("---")
    page = st.radio(
        "Sections",
        [
            "Overview",
            "Experiment Design",
            "Protocol Explorer",
            "Results Dashboard",
            "Message Log",
            "Live Demo",
            "Recommendations",
        ],
    )
    st.markdown("---")
    st.caption("Data source: saved results in `results/` plus figures in `figures/`.")


st.markdown(
    """
    <div class="hero">
      <div class="smallcaps">STAT GR5293 | Multi-Agent Communication Protocol Study</div>
      <h1 style="margin: 0.25rem 0 0.4rem 0;">Communication Protocols Change Multi-Agent Cost and Quality</h1>
      <p class="muted" style="font-size: 1.05rem; margin-bottom: 0;">
        A full experiment dashboard for the fixed Planner -> Executor -> Integrator pipeline.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

if summary is None:
    st.warning("`results/results_summary.csv` is missing. Run the notebook first to enable full dashboard results.")


if page == "Overview":
    metric_row(summary, raw)
    st.subheader("Research Question")
    st.markdown(
        """
        Multi-agent LLM systems often let developers choose how agents communicate, but that choice is usually treated as an implementation detail.
        This project turns the communication protocol into the experimental factor and asks:

        **How do communication mechanism and output format affect token cost, latency, and task completion quality?**
        """
    )
    render_flow()

    st.subheader("Project Artifacts")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### Core code")
        st.write("`pipeline.py` defines agents, protocol instructions, shared memory, evaluators, and the pipeline runner.")
    with c2:
        st.markdown("#### Experiment")
        st.write("The notebook executes the 360-run grid and writes results, message logs, summary tables, and figures.")
    with c3:
        st.markdown("#### Demo")
        st.write("This Streamlit app converts the offline experiment into a protocol recommendation and live pipeline runner.")

    st.subheader("What the Study Found")
    st.markdown(
        """
        - Protocol choice has a large effect on token cost.
        - Shared Memory is the most expensive protocol, but wins on reading and news quality.
        - Markdown is best for math completion in this run, while JSON is the cheapest math option.
        - The right protocol depends on the domain and whether the priority is quality or cost.
        """
    )


elif page == "Experiment Design":
    st.subheader("4 x 3 Factorial Experiment")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Main protocols", "4", "NL, Markdown, JSON, Shared Memory")
    c2.metric("Domains", "3", "Math, Reading, News")
    c3.metric("Samples/domain", "10")
    c4.metric("Repetitions/cell", "3")

    st.subheader("Fixed Three-Agent Pipeline")
    render_flow()

    st.subheader("Experimental Matrix")
    if summary is not None:
        matrix = summary.pivot(index="Protocol", columns="Domain", values="Completion Rate")
        st.dataframe(matrix.style.format("{:.3f}"), width="stretch")
    else:
        st.info("Summary data is not available.")

    st.subheader("Evaluation Metrics")
    metrics = pd.DataFrame(
        [
            {"Domain": "MATH", "Completion metric": "Numeric exact match", "Efficiency metrics": "Prompt, completion, total tokens; latency"},
            {"Domain": "READING", "Completion metric": "SQuAD-style token F1", "Efficiency metrics": "Prompt, completion, total tokens; latency"},
            {"Domain": "NEWS", "Completion metric": "Mean of ROUGE-2 F1 and ROUGE-L F1", "Efficiency metrics": "Prompt, completion, total tokens; latency"},
        ]
    )
    st.dataframe(metrics, width="stretch", hide_index=True)

    if config:
        with st.expander("Experiment configuration snapshot"):
            st.json(config)


elif page == "Protocol Explorer":
    st.subheader("Communication Protocol Conditions")
    cols = st.columns(2)
    for idx, (key, detail) in enumerate(PROTOCOL_DETAILS.items()):
        with cols[idx % 2]:
            render_protocol_card(key, detail)

    st.subheader("Protocol Trade-offs From the Experiment")
    if summary is not None:
        selected_domain = st.selectbox("Domain", ["MATH", "READING", "NEWS"])
        sub = summary[summary["Domain"] == selected_domain].copy()
        sub["Tokens per completion point"] = sub["Mean Tokens"] / sub["Completion Rate"].replace(0, pd.NA)
        st.dataframe(
            sub[["Protocol", "Mean Tokens", "Mean Prompt Tok", "Mean Compl Tok", "Mean Latency (ms)", "Completion Rate", "Tokens per completion point"]]
            .sort_values("Completion Rate", ascending=False)
            .style.format({
                "Mean Tokens": "{:.0f}",
                "Mean Prompt Tok": "{:.0f}",
                "Mean Compl Tok": "{:.0f}",
                "Mean Latency (ms)": "{:.0f}",
                "Completion Rate": "{:.3f}",
                "Tokens per completion point": "{:.0f}",
            }),
            width="stretch",
            hide_index=True,
        )


elif page == "Results Dashboard":
    st.subheader("Headline Results")
    if summary is not None:
        for domain in ["MATH", "READING", "NEWS"]:
            q_proto, q_score, q_tokens = best_protocol(summary, domain)
            c_proto, c_score, c_tokens = cheapest_protocol(summary, domain)
            col1, col2 = st.columns(2)
            with col1:
                st.metric(f"{domain} quality winner", q_proto, f"score {q_score:.3f}, {q_tokens:.0f} tokens")
            with col2:
                st.metric(f"{domain} cost winner", c_proto, f"score {c_score:.3f}, {c_tokens:.0f} tokens")

        st.subheader("Summary Table")
        st.dataframe(
            summary.style.format({
                "Mean Tokens": "{:.0f}",
                "Mean Prompt Tok": "{:.0f}",
                "Mean Compl Tok": "{:.0f}",
                "Mean Latency (ms)": "{:.0f}",
                "Completion Rate": "{:.3f}",
            }),
            width="stretch",
            hide_index=True,
        )

        st.subheader("Interactive Charts")
        chart_df = summary.copy()
        chart_df["Protocol / Domain"] = chart_df["Protocol"] + " / " + chart_df["Domain"]
        left, right = st.columns(2)
        with left:
            st.markdown("#### Mean tokens by cell")
            st.bar_chart(chart_df, x="Protocol / Domain", y="Mean Tokens", color="Domain", width="stretch")
        with right:
            st.markdown("#### Completion rate by cell")
            st.bar_chart(chart_df, x="Protocol / Domain", y="Completion Rate", color="Domain", width="stretch")

    st.subheader("Report Figures")
    fig_cols = st.columns(2)
    figure_specs = [
        ("fig1_tokens_by_protocol_domain.png", "Token cost by protocol and domain"),
        ("fig2_completion_by_cell.png", "Completion quality by protocol and domain"),
        ("fig4_pareto.png", "Efficiency-effectiveness Pareto trade-off"),
        ("fig6_latency.png", "Latency distribution by protocol"),
    ]
    for i, (filename, caption) in enumerate(figure_specs):
        with fig_cols[i % 2]:
            show_image(FIGURES_DIR / filename, caption)

    # ── Ablation Study: Mechanism vs Format ─────────────────────────────────
    st.markdown("---")
    st.subheader("Ablation Study: Mechanism vs Format")

    if raw is None:
        st.info("Main experiment data not found — cannot compute decomposition.")
    elif ablation_full is not None:
        st.markdown(
            "The supplemental 2x4 ablation completes the mechanism x format matrix: "
            "relay vs shared-memory crossed with default, NL, Markdown, and JSON output."
        )
        full_compare = combined_full_factorial(raw, ablation_full)
        full_means = (
            full_compare.groupby(["task_domain", "Mechanism", "Format"])["total_tokens"]
            .mean().round(0).reset_index()
        )
        full_means["Cell"] = (
            full_means["Mechanism"] + " / " + full_means["Format"] + " / " + full_means["task_domain"]
        )

        st.markdown("#### Mean tokens across the full 2x4 matrix")
        st.bar_chart(
            full_means,
            x="Cell", y="total_tokens", color="Mechanism",
            width="stretch",
        )

        domain_rows = []
        for dom in ["MATH", "READING", "NEWS"]:
            domain_rows.append(mechanism_effect_rows(full_compare, dom))
        effect_table = pd.concat(domain_rows, ignore_index=True)
        st.markdown("#### Mechanism effect by held-constant format")
        st.dataframe(
            effect_table.style.format({
                "Relay tokens": "{:.0f}",
                "Shared Memory tokens": "{:.0f}",
                "Mechanism Δ tokens": "{:+.0f}",
                "Cohen's d": "{:+.2f}",
                "p": "{:.4f}",
            }),
            width="stretch",
            hide_index=True,
        )
        st.caption(
            "Each row compares Relay vs Shared Memory while holding output format fixed. "
            "The JSON row is the original JSON vs SHARED_MEMORY_JSON comparison."
        )
    elif ablation_sm_json is not None:
        st.info(
            "Only the original SHARED_MEMORY_JSON ablation file is present. "
            "Run `python _run_full_ablation.py` to generate the complete 2x4 matrix."
        )
        st.markdown(
            "The main 4-protocol grid confounds **mechanism** (blackboard injection) and "
            "**default output format** in `SHARED_MEMORY`. The existing `SHARED_MEMORY_JSON` "
            "file still isolates the JSON-format mechanism effect."
        )

        df_abl = ablation_sm_json

        compare = pd.concat([
            raw[raw["protocol"].isin(["JSON", "SHARED_MEMORY"])][
                ["protocol", "task_domain", "total_tokens", "completion_score"]
            ],
            df_abl[["protocol", "task_domain", "total_tokens", "completion_score"]],
        ], ignore_index=True)
        token_means = (
            compare.groupby(["task_domain", "protocol"])["total_tokens"]
            .mean().round(0).reset_index()
        )
        token_means["Protocol / Domain"] = (
            token_means["protocol"] + " / " + token_means["task_domain"]
        )

        st.markdown("#### Mean tokens — JSON vs SHARED_MEMORY vs SHARED_MEMORY_JSON")
        st.bar_chart(
            token_means,
            x="Protocol / Domain", y="total_tokens", color="protocol",
            width="stretch",
        )

        rows = []
        for dom in ["MATH", "READING", "NEWS"]:
            j = raw[(raw["protocol"] == "JSON") & (raw["task_domain"] == dom)]["total_tokens"].values
            sm = raw[(raw["protocol"] == "SHARED_MEMORY") & (raw["task_domain"] == dom)]["total_tokens"].values
            sj = df_abl[(df_abl["protocol"] == "SHARED_MEMORY_JSON") & (df_abl["task_domain"] == dom)]["total_tokens"].values
            d_mech = (sj.mean() - j.mean()) / np.sqrt((j.var() + sj.var()) / 2) if (j.var() + sj.var()) > 0 else 0
            _, p_mech = stats.ttest_ind(sj, j, equal_var=False)
            d_fmt = (sj.mean() - sm.mean()) / np.sqrt((sm.var() + sj.var()) / 2) if (sm.var() + sj.var()) > 0 else 0
            _, p_fmt = stats.ttest_ind(sj, sm, equal_var=False)
            rows.append({
                "Domain": dom,
                "Mechanism Δ tokens": f"{sj.mean() - j.mean():+.0f}",
                "Mechanism Cohen's d": f"{d_mech:+.2f}",
                "Mechanism p": f"{p_mech:.4f}",
                "Format Δ tokens": f"{sj.mean() - sm.mean():+.0f}",
                "Format Cohen's d": f"{d_fmt:+.2f}",
                "Format p": f"{p_fmt:.4f}",
            })
        st.markdown("#### Statistical comparison (Welch's t-test, Cohen's d)")
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

        dom = "READING"
        j_mean = raw[(raw["protocol"] == "JSON") & (raw["task_domain"] == dom)]["total_tokens"].mean()
        sm_mean = raw[(raw["protocol"] == "SHARED_MEMORY") & (raw["task_domain"] == dom)]["total_tokens"].mean()
        sj_mean = df_abl[(df_abl["protocol"] == "SHARED_MEMORY_JSON") & (df_abl["task_domain"] == dom)]["total_tokens"].mean()
        st.markdown("#### Decomposition for READING (largest effect)")
        st.markdown(
            f"- **JSON (relay)** = {j_mean:.0f} tokens *(baseline)*\n"
            f"- **SHARED_MEMORY_JSON** = {sj_mean:.0f} tokens -> **mechanism alone adds +{sj_mean - j_mean:.0f}** (format held = JSON)\n"
            f"- **SHARED_MEMORY (default fmt)** = {sm_mean:.0f} tokens -> format change saves {sj_mean - sm_mean:+.0f} (mechanism held = blackboard)\n"
            f"- **Net SM vs JSON in main experiment** = {sm_mean - j_mean:+.0f} tokens "
            f"= mechanism (+{sj_mean - j_mean:.0f}) + format ({sm_mean - sj_mean:+.0f})"
        )
    else:
        st.info(
            "Ablation results not found. Run `python _run_full_ablation.py` "
            "to generate `results/results_ablation_full_factorial.csv`."
        )

    st.markdown("---")
    st.subheader("DeepSeek V4 Flash Robustness Check")
    if deepseek_results is None:
        st.info(
            "DeepSeek robustness results not found. Run `python _run_deepseek_robustness.py` "
            "after setting `DEEPSEEK_API_KEY` to generate the 8-protocol small-sample check."
        )
    else:
        ds = add_protocol_axes(deepseek_results)
        ds_means = (
            ds.groupby(["task_domain", "Mechanism", "Format"])["total_tokens"]
            .mean().round(0).reset_index()
        )
        ds_means["Cell"] = ds_means["Mechanism"] + " / " + ds_means["Format"] + " / " + ds_means["task_domain"]
        st.bar_chart(ds_means, x="Cell", y="total_tokens", color="Mechanism", width="stretch")

        ds_rows = []
        for dom in ["MATH", "READING", "NEWS"]:
            ds_rows.append(mechanism_effect_rows(ds, dom))
        ds_effects = pd.concat(ds_rows, ignore_index=True)
        st.dataframe(
            ds_effects.style.format({
                "Relay tokens": "{:.0f}",
                "Shared Memory tokens": "{:.0f}",
                "Mechanism Δ tokens": "{:+.0f}",
                "Cohen's d": "{:+.2f}",
                "p": "{:.4f}",
            }),
            width="stretch",
            hide_index=True,
        )


elif page == "Message Log":
    st.subheader("Message-Level Audit Browser")
    st.caption("Each saved run has three messages: Planner -> Executor, Executor -> Integrator, and Integrator -> Output.")
    if messages.empty:
        st.info("No message log found at `results/results_messages.jsonl`.")
    else:
        col1, col2, col3 = st.columns(3)
        protocols = ["All"] + sorted(messages["protocol"].dropna().unique().tolist())
        domains = ["All"] + sorted(messages["task_domain"].dropna().unique().tolist())
        senders = ["All"] + sorted(messages["sender"].dropna().unique().tolist())
        protocol_filter = col1.selectbox("Protocol", protocols)
        domain_filter = col2.selectbox("Domain", domains)
        sender_filter = col3.selectbox("Sender", senders)

        filtered = messages.copy()
        if protocol_filter != "All":
            filtered = filtered[filtered["protocol"] == protocol_filter]
        if domain_filter != "All":
            filtered = filtered[filtered["task_domain"] == domain_filter]
        if sender_filter != "All":
            filtered = filtered[filtered["sender"] == sender_filter]

        st.metric("Matching messages", f"{len(filtered):,}")
        run_ids = filtered["run_id"].drop_duplicates().head(50).tolist()
        selected_run = st.selectbox("Run ID", run_ids)
        run_messages = messages[messages["run_id"] == selected_run].sort_values("timestamp")

        if raw is not None:
            run_row = raw[raw["run_id"] == selected_run]
            if not run_row.empty:
                st.dataframe(run_row, width="stretch", hide_index=True)

        for _, m in run_messages.iterrows():
            with st.expander(f'{m["sender"]} -> {m["receiver"]} | {m["total_tokens"]} tokens | {m["latency_ms"]:.0f} ms', expanded=True):
                st.code(str(m["content"]), language="markdown")


elif page == "Live Demo":
    render_live_demo(summary)


elif page == "Recommendations":
    st.subheader("Protocol Selection Matrix")
    st.dataframe(RECOMMENDATIONS, width="stretch", hide_index=True)

    st.subheader("How to Present This Project")
    st.markdown(
        """
        1. Start with the protocol-format gap: frameworks expose communication format, but rarely quantify its cost or quality impact.
        2. Show the controlled design: same model, same agents, same tasks, only protocol changes.
        3. Walk through one live task so the audience sees Planner -> Executor -> Integrator in action.
        4. Use the results dashboard to show the cost/quality trade-off.
        5. Close on the recommendation matrix: choose protocol by domain and priority.
        """
    )

    st.subheader("Final Takeaway")
    st.info(
        "Do not choose one communication protocol globally. Choose by domain and by whether your priority is cost, latency, or completion quality."
    )
