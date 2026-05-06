"""Shared experiment helpers for supplemental protocol runs.

The main 360-run experiment remains anchored in the notebook and existing
`results/results_raw.csv`. These helpers are for additive ablations and
robustness checks that write separate result files.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from pipeline import Protocol, TaskDomain, run_pipeline


PROVIDER_CONFIG = {
    "openai": {
        "api_key_env": "OPENAI_API_KEY",
        "base_url": None,
        "default_model": "gpt-4o-mini",
        "send_seed": True,
        "extra_body": None,
    },
    "deepseek": {
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "default_model": "deepseek-v4-flash",
        "send_seed": False,
        "extra_body": {"thinking": {"type": "disabled"}},
    },
}


MODEL_PRICING_PER_1M = {
    "gpt-4o-mini": {"prompt": 0.15, "completion": 0.60},
    "gpt-4o": {"prompt": 2.50, "completion": 10.00},
    "gpt-4.1": {"prompt": 2.00, "completion": 8.00},
    # DeepSeek V4 Flash cache-miss input price. Cached input is cheaper, but
    # cache misses are the conservative default for experiment estimates.
    "deepseek-v4-flash": {"prompt": 0.14, "completion": 0.28},
}


NEWS_SAMPLES = [
    {
        "title": "Fed Holds Interest Rates Steady in March 2025",
        "content": (
            "The Federal Reserve held its benchmark interest rate unchanged at 4.25%-4.50% "
            "at its March 2025 meeting. Chair Jerome Powell stated that inflation remains "
            "above the 2% target at 2.8%, and the labor market shows continued strength with "
            "unemployment at 3.9%. The Fed signaled two potential rate cuts later in 2025 if "
            "inflation data cooperates. Markets reacted positively, with the S&P 500 rising 0.5%."
        ),
        "key_facts": [
            "Fed held rate at 4.25%-4.50%",
            "inflation at 2.8%",
            "unemployment at 3.9%",
            "two potential rate cuts in 2025",
            "S&P 500 rose 0.5%",
        ],
    },
    {
        "title": "Apple Reports Record Q1 FY2025 Earnings",
        "content": (
            "Apple reported Q1 FY2025 revenue of $124.3 billion, up 4% year-over-year. "
            "iPhone revenue reached $69.1 billion. Services revenue hit a record $26.3 billion. "
            "EPS was $2.40, beating analyst estimates of $2.35. Gross margin improved to 46.9%. "
            "CEO Tim Cook highlighted strong growth in emerging markets, particularly India."
        ),
        "key_facts": [
            "revenue $124.3 billion",
            "iPhone revenue $69.1 billion",
            "Services revenue $26.3 billion",
            "EPS $2.40",
            "gross margin 46.9%",
            "growth in India",
        ],
    },
    {
        "title": "Tesla Announces New Affordable EV Model",
        "content": (
            "Tesla unveiled its new affordable electric vehicle, the Model Q, priced at $25,000. "
            "The Model Q targets mass-market adoption with a 280-mile range and 0-60 mph in 5.8 seconds. "
            "Production is slated to begin at Gigafactory Texas in Q3 2025. Tesla stock rose 8% on the "
            "announcement. Analysts project annual sales of 500,000 units by 2027."
        ),
        "key_facts": [
            "Model Q priced at $25,000",
            "280-mile range",
            "production Q3 2025 at Gigafactory Texas",
            "stock rose 8%",
            "projected 500,000 units by 2027",
        ],
    },
    {
        "title": "Global Chip Shortage Eases as TSMC Expands",
        "content": (
            "TSMC reported that its Arizona fab is now producing 4nm chips at 90% yield rates. "
            "Global semiconductor lead times have dropped from 26 weeks to 14 weeks. "
            "TSMC revenue grew 25% YoY to $22.5 billion in Q1 2025. The company plans $40 billion "
            "in capex for 2025. Intel and Samsung are also expanding capacity but lag behind."
        ),
        "key_facts": [
            "Arizona fab producing 4nm chips at 90% yield",
            "lead times dropped from 26 to 14 weeks",
            "TSMC revenue $22.5 billion up 25% YoY",
            "$40 billion capex for 2025",
            "Intel and Samsung lag behind",
        ],
    },
    {
        "title": "EU Passes Comprehensive AI Regulation Act",
        "content": (
            "The European Union officially enacted the AI Act, the world's first comprehensive AI law. "
            "High-risk AI systems must undergo conformity assessments. Companies face fines up to 7% of "
            "global revenue for violations. The law bans social scoring and real-time biometric surveillance. "
            "Tech companies have 24 months to comply. The regulation affects all companies serving EU citizens."
        ),
        "key_facts": [
            "first comprehensive AI law",
            "fines up to 7% of global revenue",
            "bans social scoring",
            "bans real-time biometric surveillance",
            "24 months to comply",
        ],
    },
    {
        "title": "SpaceX Starship Completes First Orbital Flight",
        "content": (
            "SpaceX's Starship completed its first successful orbital flight, spending 90 minutes in orbit "
            "before splashing down in the Indian Ocean. The Super Heavy booster was caught by the launch tower. "
            "Starship reached an altitude of 250 km. NASA confirmed Starship as the lunar lander for Artemis III. "
            "SpaceX plans monthly launches starting Q2 2025. The vehicle carried a 100-ton test payload."
        ),
        "key_facts": [
            "90 minutes in orbit",
            "splashdown in Indian Ocean",
            "booster caught by launch tower",
            "altitude 250 km",
            "lunar lander for Artemis III",
            "100-ton test payload",
        ],
    },
    {
        "title": "Amazon Launches Drone Delivery in 10 US Cities",
        "content": (
            "Amazon expanded its Prime Air drone delivery service to 10 major US cities, including "
            "New York, Los Angeles, and Chicago. Deliveries under 5 pounds arrive within 30 minutes. "
            "The MK30 drone has a 15-mile delivery radius and operates in light rain. FAA granted "
            "beyond-visual-line-of-sight approval. Amazon targets 500 million drone deliveries annually by 2028."
        ),
        "key_facts": [
            "10 US cities including NY LA Chicago",
            "under 5 pounds within 30 minutes",
            "MK30 drone 15-mile radius",
            "FAA beyond-visual-line-of-sight approval",
            "500 million deliveries by 2028",
        ],
    },
    {
        "title": "OpenAI Releases GPT-5 with Reasoning Capabilities",
        "content": (
            "OpenAI released GPT-5, claiming 40% improvement on graduate-level reasoning benchmarks. "
            "The model scores 92% on the MATH benchmark and 88% on GPQA. API pricing is $15 per million "
            "input tokens and $60 per million output tokens. GPT-5 supports 256K context window. "
            "Early users report significant improvements in code generation and scientific reasoning."
        ),
        "key_facts": [
            "40% improvement on reasoning benchmarks",
            "92% on MATH benchmark",
            "88% on GPQA",
            "$15/1M input $60/1M output tokens",
            "256K context window",
        ],
    },
    {
        "title": "Japan Earthquake Triggers Tsunami Warning",
        "content": (
            "A 7.4 magnitude earthquake struck off the coast of Hokkaido, Japan at 3:42 AM local time. "
            "A tsunami warning was issued for coastal areas within 300 km. Waves up to 3 meters were "
            "observed. 14 people were injured and approximately 50,000 residents were evacuated. "
            "The Shinkansen bullet train service was suspended for 6 hours. No nuclear plant damage was reported."
        ),
        "key_facts": [
            "7.4 magnitude earthquake",
            "off coast of Hokkaido",
            "tsunami warning 300 km radius",
            "waves up to 3 meters",
            "14 injured 50000 evacuated",
            "Shinkansen suspended 6 hours",
            "no nuclear plant damage",
        ],
    },
    {
        "title": "Global Carbon Emissions Decline for First Time",
        "content": (
            "Global CO2 emissions fell 2.1% in 2024, the first annual decline not caused by economic recession. "
            "Renewable energy now accounts for 35% of global electricity generation. Solar capacity additions "
            "reached 450 GW, a record. China led with 180 GW of new solar. Coal power generation dropped 5% "
            "globally. The IEA projects emissions could fall another 3% in 2025 if trends continue."
        ),
        "key_facts": [
            "CO2 emissions fell 2.1% in 2024",
            "renewables 35% of electricity",
            "solar capacity 450 GW record",
            "China 180 GW new solar",
            "coal power dropped 5%",
            "IEA projects 3% further decline in 2025",
        ],
    },
]


def make_client(provider: str) -> OpenAI:
    load_dotenv()
    config = PROVIDER_CONFIG[provider]
    api_key = os.environ.get(config["api_key_env"], "").strip()
    if not api_key or "YOUR-KEY-HERE" in api_key:
        raise RuntimeError(f'{config["api_key_env"]} not set in environment or .env')
    if config["base_url"]:
        return OpenAI(api_key=api_key, base_url=config["base_url"])
    return OpenAI(api_key=api_key)


def load_domain_samples(
    n_math: int,
    n_reading: int,
    n_news: int,
) -> dict[TaskDomain, list[dict]]:
    from datasets import load_dataset

    gsm8k = load_dataset("openai/gsm8k", "main", split="test")
    math_samples = [
        {
            "question": item["question"],
            "answer": item["answer"].split("####")[-1].strip().replace(",", ""),
        }
        for item in list(gsm8k)[:n_math]
    ]

    squad = load_dataset("rajpurkar/squad", split="validation")
    reading_samples = [
        {
            "context": item["context"],
            "question": item["question"],
            "answers": item["answers"]["text"],
        }
        for item in list(squad)[:n_reading]
    ]

    return {
        TaskDomain.MATH: math_samples,
        TaskDomain.READING: reading_samples,
        TaskDomain.NEWS: NEWS_SAMPLES[:n_news],
    }


def _message_record(message, provider: str, model: str) -> dict:
    return {
        "provider": provider,
        "model": model,
        "run_id": message.run_id,
        "protocol": message.protocol,
        "task_domain": message.task_domain,
        "sender": message.sender,
        "receiver": message.receiver,
        "content": message.content,
        "prompt_tokens": message.prompt_tokens,
        "completion_tokens": message.completion_tokens,
        "total_tokens": message.total_tokens,
        "latency_ms": message.latency_ms,
        "finish_reason": message.finish_reason,
        "json_parse_error": message.json_parse_error,
        "timestamp": message.timestamp,
    }


def run_protocol_grid(
    *,
    provider: str,
    model: str,
    protocols: Iterable[Protocol],
    domain_samples: dict[TaskDomain, list[dict]],
    n_reps: int,
    output_csv: str,
    output_messages: str,
    sleep_s: float = 0.3,
) -> pd.DataFrame:
    config = PROVIDER_CONFIG[provider]
    client = make_client(provider)
    protocols = list(protocols)
    total = sum(len(samples) for samples in domain_samples.values()) * len(protocols) * n_reps

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(output_messages).parent.mkdir(parents=True, exist_ok=True)

    rows = []
    done = 0
    errors = 0
    started = time.time()
    print(
        f"Starting {provider}/{model}: {len(protocols)} protocols x "
        f"{sum(len(v) for v in domain_samples.values())} samples x {n_reps} reps = {total} runs"
    )

    with open(output_messages, "w", encoding="utf-8") as fp:
        for protocol in protocols:
            for domain in TaskDomain:
                for idx, sample in enumerate(domain_samples[domain]):
                    for rep in range(n_reps):
                        try:
                            result, messages = run_pipeline(
                                protocol,
                                domain,
                                sample,
                                idx,
                                client=client,
                                model=model,
                                seed=rep,
                                send_seed=config["send_seed"],
                                extra_body=config["extra_body"],
                            )
                            row = asdict(result)
                            row["provider"] = provider
                            row["model"] = model
                            rows.append(row)
                            for message in messages:
                                fp.write(
                                    json.dumps(
                                        _message_record(message, provider, model),
                                        ensure_ascii=False,
                                    )
                                    + "\n"
                                )
                            fp.flush()
                        except Exception as exc:
                            errors += 1
                            print(f"  ERROR {protocol.value}/{domain.value}/s{idx}/r{rep}: {exc}")

                        done += 1
                        if done % 10 == 0:
                            elapsed = time.time() - started
                            eta = elapsed / done * (total - done)
                            print(
                                f"  {done}/{total} done "
                                f"({errors} errors, elapsed {elapsed:.0f}s, ETA {eta:.0f}s)"
                            )
                        time.sleep(sleep_s)

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    elapsed = time.time() - started
    print(f"Complete: {len(df)} rows in {elapsed:.0f}s ({errors} errors)")
    print(f"Saved CSV:      {output_csv}")
    print(f"Saved messages: {output_messages}")
    if not df.empty:
        print(f"Total tokens:   {df['total_tokens'].sum():,}")
    return df

