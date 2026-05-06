"""One-off ablation runner for SHARED_MEMORY_JSON protocol.

Replicates notebook cell §5b standalone so we don't re-run the 360-run main
grid. Produces:
  - results/results_ablation_sm_json.csv          (90 rows)
  - results/results_ablation_sm_json_messages.jsonl (270 rows)

Run: .venv/bin/python _run_ablation.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, '.')
from openai import OpenAI
from pipeline import Protocol, TaskDomain, run_pipeline

# ── Setup ────────────────────────────────────────────────────────────────────
load_dotenv()
api_key = os.environ.get('OPENAI_API_KEY')
assert api_key and api_key != 'sk-proj-YOUR-KEY-HERE', 'OPENAI_API_KEY not set in .env'

client = OpenAI(api_key=api_key)
MODEL = 'gpt-4o-mini'
N_MATH = N_READING = N_NEWS = 10
N_REPS = 3

# ── Pull samples from the notebook cells (single source of truth) ────────────
print('Loading datasets...')
from datasets import load_dataset
gsm8k = load_dataset('openai/gsm8k', 'main', split='test')
GSM8K_SAMPLES = [
    {'question': item['question'],
     'answer': item['answer'].split('####')[-1].strip().replace(',', '')}
    for item in list(gsm8k)[:N_MATH]
]
print(f'  GSM8K: {len(GSM8K_SAMPLES)} samples')

squad = load_dataset('rajpurkar/squad', split='validation')
SQUAD_SAMPLES = [
    {'context': item['context'], 'question': item['question'],
     'answers': item['answers']['text']}
    for item in list(squad)[:N_READING]
]
print(f'  SQuAD: {len(SQUAD_SAMPLES)} samples')

# News samples — extracted from notebook cell 8b6e88d6 to keep parity
NEWS_SAMPLES = [
    {'title': 'Fed Holds Interest Rates Steady in March 2025',
     'content': ('The Federal Reserve held its benchmark interest rate unchanged at 4.25%-4.50% '
                 'at its March 2025 meeting. Chair Jerome Powell stated that inflation remains '
                 'above the 2% target at 2.8%, and the labor market shows continued strength with '
                 'unemployment at 3.9%. The Fed signaled two potential rate cuts later in 2025 if '
                 'inflation data cooperates. Markets reacted positively, with the S&P 500 rising 0.5%.'),
     'key_facts': ['Fed held rate at 4.25%-4.50%', 'inflation at 2.8%', 'unemployment at 3.9%',
                   'two potential rate cuts in 2025', 'S&P 500 rose 0.5%']},
    {'title': 'Apple Reports Record Q1 FY2025 Earnings',
     'content': ('Apple reported Q1 FY2025 revenue of $124.3 billion, up 4% year-over-year. '
                 'iPhone revenue reached $69.1 billion. Services revenue hit a record $26.3 billion. '
                 'EPS was $2.40, beating analyst estimates of $2.35. Gross margin improved to 46.9%. '
                 'CEO Tim Cook highlighted strong growth in emerging markets, particularly India.'),
     'key_facts': ['revenue $124.3 billion', 'iPhone revenue $69.1 billion',
                   'Services revenue $26.3 billion', 'EPS $2.40', 'gross margin 46.9%', 'growth in India']},
    {'title': 'Tesla Announces New Affordable EV Model',
     'content': ('Tesla unveiled its new affordable electric vehicle, the Model Q, priced at $25,000. '
                 'The Model Q targets mass-market adoption with a 280-mile range and 0-60 mph in 5.8 seconds. '
                 'Production is slated to begin at Gigafactory Texas in Q3 2025. Tesla stock rose 8% on the '
                 'announcement. Analysts project annual sales of 500,000 units by 2027.'),
     'key_facts': ['Model Q priced at $25,000', '280-mile range', 'production Q3 2025 at Gigafactory Texas',
                   'stock rose 8%', 'projected 500,000 units by 2027']},
    {'title': 'Global Chip Shortage Eases as TSMC Expands',
     'content': ('TSMC reported that its Arizona fab is now producing 4nm chips at 90% yield rates. '
                 'Global semiconductor lead times have dropped from 26 weeks to 14 weeks. '
                 'TSMC revenue grew 25% YoY to $22.5 billion in Q1 2025. The company plans $40 billion '
                 'in capex for 2025. Intel and Samsung are also expanding capacity but lag behind.'),
     'key_facts': ['Arizona fab producing 4nm chips at 90% yield', 'lead times dropped from 26 to 14 weeks',
                   'TSMC revenue $22.5 billion up 25% YoY', '$40 billion capex for 2025',
                   'Intel and Samsung lag behind']},
    {'title': 'EU Passes Comprehensive AI Regulation Act',
     'content': ("The European Union officially enacted the AI Act, the world's first comprehensive AI law. "
                 'High-risk AI systems must undergo conformity assessments. Companies face fines up to 7% of '
                 'global revenue for violations. The law bans social scoring and real-time biometric surveillance. '
                 'Tech companies have 24 months to comply. The regulation affects all companies serving EU citizens.'),
     'key_facts': ['first comprehensive AI law', 'fines up to 7% of global revenue', 'bans social scoring',
                   'bans real-time biometric surveillance', '24 months to comply']},
    {'title': 'SpaceX Starship Completes First Orbital Flight',
     'content': ("SpaceX's Starship completed its first successful orbital flight, spending 90 minutes in orbit "
                 'before splashing down in the Indian Ocean. The Super Heavy booster was caught by the launch tower. '
                 'Starship reached an altitude of 250 km. NASA confirmed Starship as the lunar lander for Artemis III. '
                 'SpaceX plans monthly launches starting Q2 2025. The vehicle carried a 100-ton test payload.'),
     'key_facts': ['90 minutes in orbit', 'splashdown in Indian Ocean', 'booster caught by launch tower',
                   'altitude 250 km', 'lunar lander for Artemis III', '100-ton test payload']},
    {'title': 'Amazon Launches Drone Delivery in 10 US Cities',
     'content': ('Amazon expanded its Prime Air drone delivery service to 10 major US cities, including '
                 'New York, Los Angeles, and Chicago. Deliveries under 5 pounds arrive within 30 minutes. '
                 'The MK30 drone has a 15-mile delivery radius and operates in light rain. FAA granted '
                 'beyond-visual-line-of-sight approval. Amazon targets 500 million drone deliveries annually by 2028.'),
     'key_facts': ['10 US cities including NY LA Chicago', 'under 5 pounds within 30 minutes',
                   'MK30 drone 15-mile radius', 'FAA beyond-visual-line-of-sight approval',
                   '500 million deliveries by 2028']},
    {'title': 'OpenAI Releases GPT-5 with Reasoning Capabilities',
     'content': ('OpenAI released GPT-5, claiming 40% improvement on graduate-level reasoning benchmarks. '
                 'The model scores 92% on the MATH benchmark and 88% on GPQA. API pricing is $15 per million '
                 'input tokens and $60 per million output tokens. GPT-5 supports 256K context window. '
                 'Early users report significant improvements in code generation and scientific reasoning.'),
     'key_facts': ['40% improvement on reasoning benchmarks', '92% on MATH benchmark', '88% on GPQA',
                   '$15/1M input $60/1M output tokens', '256K context window']},
    {'title': 'Japan Earthquake Triggers Tsunami Warning',
     'content': ('A 7.4 magnitude earthquake struck off the coast of Hokkaido, Japan at 3:42 AM local time. '
                 'A tsunami warning was issued for coastal areas within 300 km. Waves up to 3 meters were '
                 'observed. 14 people were injured and approximately 50,000 residents were evacuated. '
                 'The Shinkansen bullet train service was suspended for 6 hours. No nuclear plant damage was reported.'),
     'key_facts': ['7.4 magnitude earthquake', 'off coast of Hokkaido', 'tsunami warning 300 km radius',
                   'waves up to 3 meters', '14 injured 50000 evacuated',
                   'Shinkansen suspended 6 hours', 'no nuclear plant damage']},
    {'title': 'Global Carbon Emissions Decline for First Time',
     'content': ('Global CO2 emissions fell 2.1% in 2024, the first annual decline not caused by economic recession. '
                 'Renewable energy now accounts for 35% of global electricity generation. Solar capacity additions '
                 'reached 450 GW, a record. China led with 180 GW of new solar. Coal power generation dropped 5% '
                 'globally. The IEA projects emissions could fall another 3% in 2025 if trends continue.'),
     'key_facts': ['CO2 emissions fell 2.1% in 2024', 'renewables 35% of electricity',
                   'solar capacity 450 GW record', 'China 180 GW new solar', 'coal power dropped 5%',
                   'IEA projects 3% further decline in 2025']},
]
print(f'  News:  {len(NEWS_SAMPLES)} samples')

DOMAIN_SAMPLES = {
    TaskDomain.MATH:    GSM8K_SAMPLES,
    TaskDomain.READING: SQUAD_SAMPLES,
    TaskDomain.NEWS:    NEWS_SAMPLES,
}

# ── Run ablation loop ────────────────────────────────────────────────────────
os.makedirs('results', exist_ok=True)
ABLATION_TOTAL = (N_MATH + N_READING + N_NEWS) * N_REPS  # = 90

ablation_results = []
ablation_messages_path = 'results/results_ablation_sm_json_messages.jsonl'
fp = open(ablation_messages_path, 'w', encoding='utf-8')
done = 0
errors = 0

print(f'\nStarting ablation: {ABLATION_TOTAL} runs (SHARED_MEMORY_JSON × 3 domains × 10 samples × 3 reps)')
t_start = time.time()

for domain in TaskDomain:
    for idx, sample in enumerate(DOMAIN_SAMPLES[domain]):
        for rep in range(N_REPS):
            try:
                result, msgs = run_pipeline(
                    Protocol.SHARED_MEMORY_JSON, domain, sample, idx,
                    client=client, model=MODEL, seed=rep,
                )
                ablation_results.append(asdict(result))
                for m in msgs:
                    fp.write(json.dumps({
                        'run_id': m.run_id, 'protocol': m.protocol,
                        'task_domain': m.task_domain, 'sender': m.sender,
                        'receiver': m.receiver, 'content': m.content,
                        'prompt_tokens': m.prompt_tokens,
                        'completion_tokens': m.completion_tokens,
                        'total_tokens': m.total_tokens,
                        'latency_ms': m.latency_ms,
                        'finish_reason': m.finish_reason,
                        'json_parse_error': m.json_parse_error,
                        'timestamp': m.timestamp,
                    }, ensure_ascii=False) + '\n')
                fp.flush()
            except Exception as e:
                errors += 1
                print(f'  ERROR SHARED_MEMORY_JSON/{domain.value}/s{idx}/r{rep}: {e}')
            done += 1
            if done % 10 == 0:
                elapsed = time.time() - t_start
                eta = elapsed / done * (ABLATION_TOTAL - done)
                print(f'  {done}/{ABLATION_TOTAL} done ({errors} errors, elapsed {elapsed:.0f}s, ETA {eta:.0f}s)')
            time.sleep(0.3)

fp.close()
df = pd.DataFrame(ablation_results)
df.to_csv('results/results_ablation_sm_json.csv', index=False)

elapsed = time.time() - t_start
print(f'\nAblation complete: {len(df)} runs in {elapsed:.0f}s ({errors} errors)')
print(f'Total tokens: {df["total_tokens"].sum():,}')
print(f'Saved CSV:      results/results_ablation_sm_json.csv  ({len(df)} rows)')
print(f'Saved messages: {ablation_messages_path}')
