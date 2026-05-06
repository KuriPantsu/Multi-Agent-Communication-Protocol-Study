"""
FastAPI dashboard for the Multi-Agent Communication Protocol Study.

Run:
    uvicorn fastapi_app:app --reload --port 8000

The OpenAI key is intentionally local-only:
    1. `OPENAI_API_KEY` from the shell, or
    2. `OPENAI_API_KEY=...` in a project-root `.env` file.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel, Field

from pipeline import Protocol, TaskDomain, run_pipeline


ROOT = Path(__file__).resolve().parent
DASHBOARD_DIR = ROOT / "dashboard"
RESULTS_SUMMARY = ROOT / "results" / "results_summary.csv"
RESULTS_RAW = ROOT / "results" / "results_raw.csv"
RESULTS_MESSAGES = ROOT / "results" / "results_messages.jsonl"
EXPERIMENT_CONFIG = ROOT / "results" / "experiment_config.json"
FIGURES_DIR = ROOT / "figures"
COST_PER_1M = {"prompt": 0.15, "completion": 0.60}

load_dotenv(ROOT / ".env")

app = FastAPI(
    title="Multi-Agent Protocol Study Dashboard",
    description="Experiment dashboard and live multi-agent pipeline runner.",
)
app.mount("/static", StaticFiles(directory=DASHBOARD_DIR), name="static")
app.mount("/figures", StaticFiles(directory=FIGURES_DIR), name="figures")


class RunRequest(BaseModel):
    task: str = Field(..., min_length=1)
    model: str = "gpt-4o-mini"
    auto_protocol: bool = True
    protocol: Optional[str] = None


class CompareRequest(BaseModel):
    task: str = Field(..., min_length=1)
    model: str = "gpt-4o-mini"
    protocols: list[str] = Field(default_factory=lambda: ["NL", "MARKDOWN", "JSON", "SHARED_MEMORY"])


class ClassifyRequest(BaseModel):
    task: str = Field(..., min_length=1)
    model: str = "gpt-4o-mini"


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Missing file: {path.relative_to(ROOT)}")
    return pd.read_csv(path)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def client_from_local_key() -> OpenAI:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key or "YOUR-KEY-HERE" in key:
        raise HTTPException(
            status_code=400,
            detail="OPENAI_API_KEY is not configured. Add it to your shell or project-root .env file.",
        )
    return OpenAI(api_key=key)


def dataframe_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(df.to_json(orient="records"))


def best_protocol(summary: pd.DataFrame, domain: str) -> tuple[str, float, float]:
    sub = summary[summary["Domain"] == domain]
    if sub.empty:
        fallback = {"MATH": "JSON", "READING": "JSON", "NEWS": "NL", "OTHER": "NL"}
        return fallback.get(domain, "NL"), 0.0, 0.0
    row = sub.sort_values(["Completion Rate", "Mean Tokens"], ascending=[False, True]).iloc[0]
    return row["Protocol"], float(row["Completion Rate"]), float(row["Mean Tokens"])


def cheapest_protocol(summary: pd.DataFrame, domain: str) -> tuple[str, float, float]:
    sub = summary[summary["Domain"] == domain]
    row = sub.sort_values(["Mean Tokens", "Completion Rate"], ascending=[True, False]).iloc[0]
    return row["Protocol"], float(row["Completion Rate"]), float(row["Mean Tokens"])


def classify_domain(client: OpenAI, model: str, task_text: str) -> tuple[str, float]:
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
        parsed = json.loads(resp.choices[0].message.content or "{}")
        domain = str(parsed.get("domain", "OTHER")).upper()
        confidence = float(parsed.get("confidence", 0.5))
    except (json.JSONDecodeError, ValueError, TypeError):
        domain, confidence = "OTHER", 0.5
    if domain not in {"MATH", "READING", "NEWS", "OTHER"}:
        domain = "OTHER"
    return domain, confidence


def build_sample(domain: str, task_text: str) -> tuple[TaskDomain, dict[str, Any]]:
    if domain == "MATH":
        return TaskDomain.MATH, {"question": task_text, "answer": ""}
    if domain == "READING":
        marker = "Question:"
        if marker in task_text:
            context, question = task_text.rsplit(marker, 1)
            question = question.strip() or "Answer the question."
        elif "?" in task_text:
            context, question = task_text.rsplit("?", 1)
            question = (question.strip() + "?") if question.strip() else "Answer the question."
        else:
            context, question = task_text, "Answer based on the passage."
        return TaskDomain.READING, {"context": context.strip(), "question": question.strip(), "answers": []}
    return TaskDomain.NEWS, {"title": "User input", "content": task_text, "key_facts": []}


def protocol_recommendation(domain: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Create a practical recommendation from this specific live comparison."""
    if not rows:
        return {
            "recommended_protocol": None,
            "priority": "unavailable",
            "reason": "No protocol runs were completed.",
            "alternatives": [],
        }

    fastest = min(rows, key=lambda r: r["latency_ms"])
    cheapest = min(rows, key=lambda r: r["estimated_cost_usd"])
    leanest = min(rows, key=lambda r: r["total_tokens"])
    by_protocol = {r["protocol"]: r for r in rows}

    quality_preference = {
        "MATH": "MARKDOWN",
        "READING": "SHARED_MEMORY",
        "NEWS": "SHARED_MEMORY",
        "OTHER": "NL",
    }.get(domain, "NL")

    if quality_preference in by_protocol:
        recommended = by_protocol[quality_preference]
        priority = "quality"
        reason = (
            f"For {domain}, the offline experiment suggests {quality_preference} when answer quality is the priority. "
            f"In this live run it used {recommended['total_tokens']} tokens and {recommended['latency_ms']:.0f} ms."
        )
    else:
        recommended = leanest
        priority = "cost"
        reason = (
            f"The preferred quality protocol for {domain} was not selected, so the recommendation falls back to "
            f"the lowest-token option in this live run: {leanest['protocol']}."
        )

    if recommended["protocol"] != leanest["protocol"]:
        token_delta = recommended["total_tokens"] - leanest["total_tokens"]
        pct = token_delta / max(leanest["total_tokens"], 1) * 100
        reason += f" Compared with the lowest-token option ({leanest['protocol']}), this costs {token_delta:+.0f} tokens ({pct:+.1f}%)."
    else:
        reason += " It is also the lowest-token option among the selected protocols."

    if recommended["protocol"] != fastest["protocol"]:
        latency_delta = recommended["latency_ms"] - fastest["latency_ms"]
        reason += f" The fastest option is {fastest['protocol']} by {abs(latency_delta):.0f} ms."
    else:
        reason += " It is also the fastest option in this live run."

    alternatives = [
        {
            "label": "Lowest token cost",
            "protocol": leanest["protocol"],
            "tokens": leanest["total_tokens"],
            "latency_ms": leanest["latency_ms"],
            "estimated_cost_usd": leanest["estimated_cost_usd"],
        },
        {
            "label": "Fastest latency",
            "protocol": fastest["protocol"],
            "tokens": fastest["total_tokens"],
            "latency_ms": fastest["latency_ms"],
            "estimated_cost_usd": fastest["estimated_cost_usd"],
        },
        {
            "label": "Lowest dollar cost",
            "protocol": cheapest["protocol"],
            "tokens": cheapest["total_tokens"],
            "latency_ms": cheapest["latency_ms"],
            "estimated_cost_usd": cheapest["estimated_cost_usd"],
        },
    ]
    return {
        "recommended_protocol": recommended["protocol"],
        "priority": priority,
        "reason": reason,
        "alternatives": alternatives,
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(DASHBOARD_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "has_openai_key": bool(os.environ.get("OPENAI_API_KEY", "").strip()),
        "summary_exists": RESULTS_SUMMARY.exists(),
        "raw_exists": RESULTS_RAW.exists(),
        "messages_exists": RESULTS_MESSAGES.exists(),
    }


@app.get("/api/summary")
def summary() -> dict[str, Any]:
    df = read_csv(RESULTS_SUMMARY)
    raw = read_csv(RESULTS_RAW) if RESULTS_RAW.exists() else pd.DataFrame()
    recommendations = []
    for domain in ["MATH", "READING", "NEWS"]:
        q_protocol, q_score, q_tokens = best_protocol(df, domain)
        c_protocol, c_score, c_tokens = cheapest_protocol(df, domain)
        recommendations.append(
            {
                "domain": domain,
                "qualityProtocol": q_protocol,
                "qualityScore": q_score,
                "qualityTokens": q_tokens,
                "costProtocol": c_protocol,
                "costScore": c_score,
                "costTokens": c_tokens,
            }
        )
    return {
        "summary": dataframe_records(df),
        "runCount": int(raw.shape[0]) if not raw.empty else 0,
        "messageCount": int(raw.shape[0] * 3) if not raw.empty else 0,
        "protocols": sorted(df["Protocol"].dropna().unique().tolist()),
        "domains": sorted(df["Domain"].dropna().unique().tolist()),
        "recommendations": recommendations,
    }


@app.get("/api/config")
def config() -> dict[str, Any]:
    return read_json(EXPERIMENT_CONFIG)


@app.get("/api/messages")
def messages(
    protocol: str = Query("All"),
    domain: str = Query("All"),
    sender: str = Query("All"),
    run_id: str = Query(""),
    limit: int = Query(60, ge=1, le=300),
) -> dict[str, Any]:
    if not RESULTS_MESSAGES.exists():
        return {"messages": [], "runIds": []}
    records = []
    with RESULTS_MESSAGES.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                if protocol != "All" and record.get("protocol") != protocol:
                    continue
                if domain != "All" and record.get("task_domain") != domain:
                    continue
                if sender != "All" and record.get("sender") != sender:
                    continue
                if run_id and record.get("run_id") != run_id:
                    continue
                records.append(record)
    run_ids = []
    seen = set()
    for record in records:
        rid = record.get("run_id")
        if rid and rid not in seen:
            seen.add(rid)
            run_ids.append(rid)
    if run_id:
        shown = sorted(records, key=lambda r: r.get("timestamp", 0))
    else:
        shown = records[:limit]
    return {"messages": shown, "runIds": run_ids[:100], "count": len(records)}


@app.post("/api/classify")
def classify(req: ClassifyRequest) -> dict[str, Any]:
    client = client_from_local_key()
    domain, confidence = classify_domain(client, req.model, req.task)
    return {"domain": domain, "confidence": confidence}


@app.post("/api/run")
def run(req: RunRequest) -> dict[str, Any]:
    client = client_from_local_key()
    domain, confidence = classify_domain(client, req.model, req.task)
    summary_df = read_csv(RESULTS_SUMMARY) if RESULTS_SUMMARY.exists() else pd.DataFrame()

    if req.auto_protocol:
        protocol_name, completion_rate, mean_tokens = best_protocol(summary_df, domain) if not summary_df.empty else ("NL", 0.0, 0.0)
        rationale = {
            "mode": "auto",
            "completionRate": completion_rate,
            "meanTokens": mean_tokens,
        }
    else:
        protocol_name = req.protocol or "NL"
        rationale = {"mode": "manual", "completionRate": None, "meanTokens": None}

    try:
        protocol = Protocol(protocol_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Unknown protocol: {protocol_name}") from exc

    task_domain, sample = build_sample(domain, req.task)
    result, log = run_pipeline(
        protocol=protocol,
        domain=task_domain,
        sample=sample,
        sample_idx=0,
        client=client,
        model=req.model,
        seed=0,
    )
    cost_usd = (
        result.total_prompt_tokens / 1e6 * COST_PER_1M["prompt"]
        + result.total_completion_tokens / 1e6 * COST_PER_1M["completion"]
    )
    return {
        "domain": domain,
        "confidence": confidence,
        "protocol": protocol_name,
        "rationale": rationale,
        "result": {
            **result.__dict__,
            "estimated_cost_usd": cost_usd,
        },
        "messages": [m.__dict__ for m in log],
    }


@app.post("/api/compare")
def compare(req: CompareRequest) -> dict[str, Any]:
    client = client_from_local_key()
    domain, confidence = classify_domain(client, req.model, req.task)
    task_domain, sample = build_sample(domain, req.task)

    selected_protocols = req.protocols or ["NL", "MARKDOWN", "JSON", "SHARED_MEMORY"]
    rows = []
    for protocol_name in selected_protocols:
        try:
            protocol = Protocol(protocol_name)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"Unknown protocol: {protocol_name}") from exc

        result, log = run_pipeline(
            protocol=protocol,
            domain=task_domain,
            sample=sample,
            sample_idx=0,
            client=client,
            model=req.model,
            seed=0,
        )
        cost_usd = (
            result.total_prompt_tokens / 1e6 * COST_PER_1M["prompt"]
            + result.total_completion_tokens / 1e6 * COST_PER_1M["completion"]
        )
        rows.append(
            {
                "protocol": protocol_name,
                "domain": domain,
                "confidence": confidence,
                "prompt_tokens": result.total_prompt_tokens,
                "completion_tokens": result.total_completion_tokens,
                "total_tokens": result.total_tokens,
                "latency_ms": result.total_latency_ms,
                "estimated_cost_usd": cost_usd,
                "messages": result.total_messages,
                "tokens_per_second": round(result.total_tokens / (result.total_latency_ms / 1000), 2)
                if result.total_latency_ms
                else None,
                "any_truncation": result.any_truncation,
                "any_json_parse_error": result.any_json_parse_error,
                "finish_reasons": [m.finish_reason for m in log],
            }
        )

    fastest = min(rows, key=lambda r: r["latency_ms"]) if rows else None
    cheapest = min(rows, key=lambda r: r["estimated_cost_usd"]) if rows else None
    leanest = min(rows, key=lambda r: r["total_tokens"]) if rows else None
    return {
        "domain": domain,
        "confidence": confidence,
        "rows": rows,
        "recommendation": protocol_recommendation(domain, rows),
        "winners": {
            "fastest": fastest["protocol"] if fastest else None,
            "cheapest": cheapest["protocol"] if cheapest else None,
            "fewest_tokens": leanest["protocol"] if leanest else None,
        },
    }
