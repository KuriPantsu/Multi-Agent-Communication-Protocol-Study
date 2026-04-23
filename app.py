"""
Streamlit demo — Multi-Agent Communication Protocol Study.

Workflow (matches old_proposal §7 Frontend Demo Design):
  1. User enters a free-text task.
  2. A small LLM call classifies the task domain (MATH / READING / NEWS / OTHER).
  3. The app looks up the best-performing protocol for that domain from the
     experiment summary (`results/results_summary.csv`) and displays the rationale.
  4. The three-agent pipeline runs; each agent's output is rendered as it arrives.
  5. A metrics dashboard shows tokens, latency, cost, and the protocol used.

Run: `streamlit run app.py`
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st
from openai import OpenAI

from pipeline import (
    DOMAIN_MAX_TOKENS,
    EVALUATORS,
    Protocol,
    TaskDomain,
    run_pipeline,
)

RESULTS_SUMMARY = Path('results/results_summary.csv')
COST_PER_1M = {'prompt': 0.15, 'completion': 0.60}  # gpt-4o-mini pricing


# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='Multi-Agent Protocol Study — Demo',
    page_icon='🔀',
    layout='wide',
)


@st.cache_data
def load_summary() -> Optional[pd.DataFrame]:
    if not RESULTS_SUMMARY.exists():
        return None
    return pd.read_csv(RESULTS_SUMMARY)


def best_protocol(summary: pd.DataFrame, domain: str) -> tuple[str, float, float]:
    """Return (best_protocol, completion_rate, mean_tokens) for a domain."""
    sub = summary[summary['Domain'] == domain]
    row = sub.sort_values(
        ['Completion Rate', 'Mean Tokens'], ascending=[False, True]
    ).iloc[0]
    return row['Protocol'], row['Completion Rate'], row['Mean Tokens']


def classify_domain(client: OpenAI, model: str, task_text: str) -> tuple[str, float]:
    """Ask the LLM to classify the task. Returns (domain, confidence 0-1)."""
    classifier_prompt = (
        'Classify the following user task into exactly one category:\n'
        '  MATH     — numerical / arithmetic / word problems with a number answer\n'
        '  READING  — question answering where the answer is a short span from a passage\n'
        '  NEWS     — article summarization or factual analysis of a news-style passage\n'
        '  OTHER    — anything that clearly does not fit the above\n\n'
        'Respond with JSON: {"domain": "...", "confidence": 0.0-1.0}.\n\n'
        f'TASK:\n{task_text}'
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {'role': 'system', 'content': 'You are a concise task classifier.'},
            {'role': 'user', 'content': classifier_prompt},
        ],
        response_format={'type': 'json_object'},
        temperature=0.0,
        max_tokens=60,
    )
    try:
        parsed = json.loads(resp.choices[0].message.content)
        domain = parsed.get('domain', 'OTHER').upper()
        conf = float(parsed.get('confidence', 0.5))
    except (json.JSONDecodeError, ValueError, TypeError):
        domain, conf = 'OTHER', 0.5
    if domain not in {'MATH', 'READING', 'NEWS', 'OTHER'}:
        domain = 'OTHER'
    return domain, conf


def build_sample(domain: str, task_text: str) -> tuple[TaskDomain, dict]:
    """Coerce the user's free text into the dict shape each TaskDomain expects."""
    if domain == 'MATH':
        return TaskDomain.MATH, {'question': task_text, 'answer': ''}
    if domain == 'READING':
        parts = task_text.split('?', 1)
        if len(parts) == 2:
            ctx, q = parts[0].strip(), parts[1].strip() or 'Answer the question.'
        else:
            ctx, q = task_text, 'Answer based on the passage.'
        return TaskDomain.READING, {'context': ctx, 'question': q, 'answers': []}
    # NEWS and OTHER both route to NEWS (most general analysis flow)
    return TaskDomain.NEWS, {'title': 'User input', 'content': task_text, 'key_facts': []}


# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title('⚙️ Settings')
api_key = st.sidebar.text_input(
    'OpenAI API key',
    value=os.environ.get('OPENAI_API_KEY', ''),
    type='password',
    help='Stored in session only. Defaults to OPENAI_API_KEY env var.',
)
model = st.sidebar.selectbox('Model', ['gpt-4o-mini', 'gpt-4o'], index=0)
auto_protocol = st.sidebar.checkbox(
    'Auto-select protocol (recommended)', value=True,
    help='When off, you choose the protocol manually.',
)
manual_protocol = st.sidebar.selectbox(
    'Manual protocol override',
    [p.value for p in Protocol],
    index=2,
    disabled=auto_protocol,
)
st.sidebar.markdown('---')
st.sidebar.caption(
    'Protocol recommendations are derived from the 360-run experiment '
    '(results/results_summary.csv). If the summary file is missing the app '
    'falls back to JSON for MATH, NL for NEWS/OTHER, JSON for READING.'
)


# ── Header ───────────────────────────────────────────────────────────────────
st.title('Multi-Agent Communication Protocol Study')
st.markdown(
    'A three-agent pipeline **Planning → Execution → Integration** routes a free-text '
    'task through an empirically selected inter-agent communication protocol.'
)

summary = load_summary()
if summary is None:
    st.warning(
        '`results/results_summary.csv` not found. Run the notebook end-to-end first '
        '(`jupyter nbconvert --execute Multi_Agent_Communication_Protocol_Study.ipynb`). '
        'The demo will use fallback recommendations until then.',
        icon='⚠️',
    )

FALLBACK_RECS = {
    'MATH': ('JSON', None, None),
    'READING': ('JSON', None, None),
    'NEWS': ('NL', None, None),
    'OTHER': ('NL', None, None),
}


# ── User input ───────────────────────────────────────────────────────────────
task_text = st.text_area(
    'Enter your task',
    height=160,
    placeholder=(
        'Example (MATH): Janet has 16 eggs, eats 3, bakes 4 muffins, sells the rest at $2/egg. '
        'How much does she make?\n\n'
        'Example (READING): [passage]… Question: Who invented X?\n\n'
        'Example (NEWS): [news article text]'
    ),
)

col_run, col_clear = st.columns([1, 4])
run_clicked = col_run.button('Run pipeline', type='primary', disabled=not (api_key and task_text.strip()))
col_clear.caption('Protocol, tokens, latency, and cost update live below.')


if run_clicked:
    client = OpenAI(api_key=api_key)

    # Step 1 — classification
    with st.status('Step 1 — classifying task domain…', expanded=False) as status:
        try:
            domain_str, conf = classify_domain(client, model, task_text)
        except Exception as exc:
            st.error(f'Classification failed: {exc}')
            st.stop()
        status.update(label=f'Step 1 ✓ — domain = **{domain_str}**  (conf {conf:.2f})', state='complete')

    # Step 2 — protocol recommendation
    if auto_protocol:
        if summary is not None and domain_str in summary['Domain'].values:
            proto_name, completion_rate, mean_tokens = best_protocol(summary, domain_str)
            rationale = (
                f'{proto_name} recommended for {domain_str}: mean completion '
                f'{completion_rate:.3f}, ~{mean_tokens:.0f} tokens/run '
                '(based on our 360-run experiment).'
            )
        else:
            proto_name, _, _ = FALLBACK_RECS.get(domain_str, FALLBACK_RECS['OTHER'])
            rationale = f'{proto_name} (fallback — experiment summary unavailable).'
    else:
        proto_name = manual_protocol
        rationale = f'{proto_name} (manual override).'

    st.info(f'**Step 2 — protocol: {proto_name}**  ·  {rationale}', icon='🔀')

    protocol = Protocol(proto_name)
    task_domain, sample = build_sample(domain_str, task_text)

    # Step 3 — run pipeline with live streaming
    st.markdown('### Step 3 — Agent messages')
    planner_box = st.empty()
    executor_box = st.empty()
    integrator_box = st.empty()

    t_start = time.time()
    try:
        with st.spinner('Running Planner → Executor → Integrator…'):
            result, msgs = run_pipeline(
                protocol, task_domain, sample, sample_idx=0,
                client=client, model=model, seed=0,
            )
    except Exception as exc:
        st.error(f'Pipeline failed: {exc}')
        st.stop()
    total_time = time.time() - t_start

    # Render each agent's contribution
    by_sender = {m.sender: m for m in msgs}
    if 'Planner' in by_sender:
        with planner_box.container():
            st.markdown(f"**🧭 Planner** · {by_sender['Planner'].total_tokens} tok · {by_sender['Planner'].latency_ms:.0f} ms")
            st.code(str(by_sender['Planner'].content), language='markdown')
    if 'Executor' in by_sender:
        with executor_box.container():
            st.markdown(f"**⚙️ Executor** · {by_sender['Executor'].total_tokens} tok · {by_sender['Executor'].latency_ms:.0f} ms")
            st.code(str(by_sender['Executor'].content), language='markdown')
    if 'Integrator' in by_sender:
        with integrator_box.container():
            st.markdown(f"**🧩 Integrator** · {by_sender['Integrator'].total_tokens} tok · {by_sender['Integrator'].latency_ms:.0f} ms")
            st.code(str(by_sender['Integrator'].content), language='markdown')

    # Step 4 — dashboard
    cost_usd = (
        result.total_prompt_tokens / 1e6 * COST_PER_1M['prompt']
        + result.total_completion_tokens / 1e6 * COST_PER_1M['completion']
    )
    st.markdown('### Step 4 — Run metrics')
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric('Protocol', proto_name)
    c2.metric('Total tokens', f'{result.total_tokens:,}',
              f'{result.total_prompt_tokens}p + {result.total_completion_tokens}c')
    c3.metric('Latency (ms)', f'{result.total_latency_ms:,.0f}')
    c4.metric('Est. cost (USD)', f'${cost_usd:.5f}')
    c5.metric('Domain conf.', f'{conf:.2f}')

    if result.any_truncation:
        st.warning('⚠️ At least one agent hit `finish_reason=length` — output may be truncated.')
    if result.any_json_parse_error:
        st.warning('⚠️ JSON protocol output failed to parse even after retry.')

    # Final answer
    st.markdown('### Final answer')
    st.success(str(by_sender.get('Integrator').content) if 'Integrator' in by_sender else '(no integrator output)')

    # Debug pane
    with st.expander('🔍 Raw message log'):
        for m in msgs:
            st.markdown(
                f"**{m.sender} → {m.receiver}** · {m.total_tokens} tok "
                f"({m.prompt_tokens}p + {m.completion_tokens}c) · "
                f"{m.latency_ms:.0f} ms · finish_reason={m.finish_reason}"
            )
            st.text(str(m.content))
            st.markdown('---')
