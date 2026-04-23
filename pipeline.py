"""
Shared helpers for the multi-agent communication protocol study.

Imported by both `Multi_Agent_Communication_Protocol_Study.ipynb` (experiment
driver) and `app.py` (Streamlit demo). The notebook handles data loading, grid
execution, and analysis; this module owns the agent/pipeline primitives.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import time
from collections import Counter
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from rouge_score import rouge_scorer as _rouge_scorer
    _ROUGE = _rouge_scorer.RougeScorer(['rouge2', 'rougeL'], use_stemmer=True)
except ImportError:
    _ROUGE = None


# ── Enums ────────────────────────────────────────────────────────────────────
class Protocol(str, Enum):
    NL = 'NL'
    MARKDOWN = 'MARKDOWN'
    JSON = 'JSON'
    SHARED_MEMORY = 'SHARED_MEMORY'


class TaskDomain(str, Enum):
    MATH = 'MATH'
    READING = 'READING'
    NEWS = 'NEWS'


# ── Communication log ────────────────────────────────────────────────────────
@dataclass
class Message:
    run_id: str
    protocol: str
    task_domain: str
    sender: str
    receiver: str
    content: Any
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    finish_reason: str = ''
    json_parse_error: bool = False
    timestamp: float = field(default_factory=time.time)


class CommunicationLogger:
    def __init__(self):
        self.messages: List[Message] = []

    def log(self, msg: Message):
        self.messages.append(msg)

    def summary(self) -> Dict:
        if not self.messages:
            return {}
        return {
            'total_messages': len(self.messages),
            'total_prompt_tokens': sum(m.prompt_tokens for m in self.messages),
            'total_completion_tokens': sum(m.completion_tokens for m in self.messages),
            'total_tokens': sum(m.total_tokens for m in self.messages),
            'total_latency_ms': round(sum(m.latency_ms for m in self.messages), 2),
            'any_truncation': any(m.finish_reason == 'length' for m in self.messages),
            'any_json_parse_error': any(m.json_parse_error for m in self.messages),
        }

    def clear(self):
        self.messages = []


# ── Shared memory blackboard ─────────────────────────────────────────────────
class SharedMemory:
    """
    A true blackboard: every agent can read the entire global state.
    `snapshot()` serializes the full state so downstream agents see everything
    prior agents wrote — this is what makes SHARED_MEMORY incur real overhead.
    """

    def __init__(self):
        self._state: Dict[str, Any] = {}

    def write(self, agent: str, key: str, value: Any):
        self._state[key] = value

    def read(self, key: str, default=None) -> Any:
        return self._state.get(key, default)

    def snapshot(self) -> str:
        return json.dumps(self._state, default=str, ensure_ascii=False, indent=2)

    def clear(self):
        self._state = {}


# ── Agent prompts and protocol format instructions ────────────────────────────
SYSTEM_PROMPTS = {
    'planner': (
        'You are a Planning Agent. Decompose the given task into 3-5 clear subtasks. '
        'Output ONLY the subtask list, no extra commentary.'
    ),
    'executor': (
        'You are an Execution Agent. Complete each subtask precisely based on the '
        'provided information. Output ONLY your results, no extra commentary.'
    ),
    'integrator': (
        'You are an Integration Agent. Synthesize all subtask results into a single '
        'coherent final answer. Output ONLY the final answer, no extra commentary.'
    ),
}


# NL is now explicit — otherwise the model drifts toward markdown-by-default,
# which muddles the NL-vs-Markdown comparison.
PROTOCOL_INSTRUCTIONS = {
    Protocol.NL: (
        '\n\nRespond in plain English prose. Do not use markdown, bullet points, '
        'headings, JSON, or any structured formatting.'
    ),
    Protocol.MARKDOWN: (
        '\n\nFormat your response using Markdown with clear headings (##), '
        'bullet points (-), and numbered lists where appropriate.'
    ),
    Protocol.JSON: (
        '\n\nFormat your response as a valid JSON object with descriptive field '
        'names. Output ONLY the JSON, no markdown fences or extra text.'
    ),
    # Shared-memory agents get a blackboard preamble instead of a format suffix.
    Protocol.SHARED_MEMORY: '',
}


SHARED_MEMORY_PREAMBLE = (
    'You share a global blackboard with other agents. Below is the current '
    'blackboard state as JSON. Read what you need to complete your role.\n\n'
    'BLACKBOARD STATE:\n{state}\n\nYOUR TASK:\n'
)


DOMAIN_MAX_TOKENS = {
    TaskDomain.MATH: 256,
    TaskDomain.READING: 256,
    TaskDomain.NEWS: 512,
}


# ── LLM wrapper ──────────────────────────────────────────────────────────────
def llm_call(
    client,
    model: str,
    role: str,
    prompt: str,
    protocol: Protocol,
    domain: TaskDomain = TaskDomain.MATH,
    temperature: float = 0.3,
    seed: Optional[int] = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    One LLM call. Returns (response_text, meta) where meta contains
    prompt_tokens, completion_tokens, total_tokens, latency_ms, finish_reason,
    json_parse_error.

    Enforces JSON output for Protocol.JSON via response_format, with one retry
    on parse failure. Falls back to raw text if both attempts fail.
    """
    max_tokens = DOMAIN_MAX_TOKENS.get(domain, 256)
    full_prompt = prompt + PROTOCOL_INSTRUCTIONS[protocol]

    kwargs: Dict[str, Any] = {
        'model': model,
        'messages': [
            {'role': 'system', 'content': SYSTEM_PROMPTS[role]},
            {'role': 'user', 'content': full_prompt},
        ],
        'temperature': temperature,
        'max_tokens': max_tokens,
    }
    if seed is not None:
        kwargs['seed'] = seed
    if protocol == Protocol.JSON:
        kwargs['response_format'] = {'type': 'json_object'}

    def _invoke() -> Tuple[Any, float]:
        t0 = time.time()
        for attempt in range(5):
            try:
                r = client.chat.completions.create(**kwargs)
                return r, round((time.time() - t0) * 1000, 2)
            except Exception as exc:
                msg = str(exc).lower()
                if 'rate' in msg or '429' in msg:
                    wait = 2 ** attempt
                    print(f'    Rate limited, retrying in {wait}s...')
                    time.sleep(wait)
                else:
                    raise
        raise RuntimeError('Exhausted rate-limit retries')

    resp, latency_ms = _invoke()
    text = resp.choices[0].message.content.strip()
    finish_reason = resp.choices[0].finish_reason
    usage = resp.usage
    json_parse_error = False

    # For JSON protocol, validate parseability; one retry if it fails.
    if protocol == Protocol.JSON:
        try:
            json.loads(text)
        except (json.JSONDecodeError, ValueError):
            resp2, latency_ms2 = _invoke()
            text2 = resp2.choices[0].message.content.strip()
            try:
                json.loads(text2)
                text = text2
                finish_reason = resp2.choices[0].finish_reason
                latency_ms = latency_ms + latency_ms2
                usage = resp2.usage
            except (json.JSONDecodeError, ValueError):
                json_parse_error = True
                latency_ms = latency_ms + latency_ms2

    meta = {
        'prompt_tokens': usage.prompt_tokens,
        'completion_tokens': usage.completion_tokens,
        'total_tokens': usage.total_tokens,
        'latency_ms': latency_ms,
        'finish_reason': finish_reason,
        'json_parse_error': json_parse_error,
    }
    return text, meta


# ── Task prompt builders ─────────────────────────────────────────────────────
def build_math_prompt(sample: dict) -> str:
    return (
        f'Solve this math problem step by step and give the final numeric answer.\n\n'
        f'Problem: {sample["question"]}'
    )


def build_reading_prompt(sample: dict) -> str:
    return (
        f'Read the following passage and answer the question. '
        f'Give a short, precise answer.\n\n'
        f'Passage: {sample["context"]}\n\n'
        f'Question: {sample["question"]}'
    )


def build_news_prompt(sample: dict) -> str:
    return (
        f'Analyze the following news article. Extract and summarize all key facts, '
        f'figures, and important details.\n\n'
        f'Title: {sample["title"]}\n\n'
        f'Article: {sample["content"]}'
    )


TASK_BUILDERS = {
    TaskDomain.MATH: build_math_prompt,
    TaskDomain.READING: build_reading_prompt,
    TaskDomain.NEWS: build_news_prompt,
}


# ── Evaluators ───────────────────────────────────────────────────────────────
def extract_number(text: str) -> Optional[str]:
    m = re.search(r'####\s*([\-\d,\.]+)', text)
    if m:
        return m.group(1).replace(',', '')
    numbers = re.findall(r'[\-]?[\d,]+\.?\d*', text)
    if numbers:
        return numbers[-1].replace(',', '')
    return None


def evaluate_math(response: str, gold_answer: str) -> float:
    extracted = extract_number(response)
    if extracted is None:
        return 0.0
    try:
        return 1.0 if abs(float(extracted) - float(gold_answer)) < 0.01 else 0.0
    except ValueError:
        return 1.0 if extracted.strip() == gold_answer.strip() else 0.0


def _normalize_tokens(text: str) -> list:
    text = text.lower()
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return text.split()


def _token_f1(prediction: str, gold: str) -> float:
    pred_toks = Counter(_normalize_tokens(prediction))
    gold_toks = Counter(_normalize_tokens(gold))
    common = sum((pred_toks & gold_toks).values())
    if common == 0:
        return 0.0
    precision = common / sum(pred_toks.values())
    recall = common / sum(gold_toks.values())
    return 2 * precision * recall / (precision + recall)


def evaluate_reading(response: str, gold_answers: List[str]) -> float:
    if not gold_answers:
        return 0.0
    return round(max(_token_f1(response, g) for g in gold_answers), 3)


def evaluate_news(response: str, key_facts: List[str]) -> float:
    if not key_facts or _ROUGE is None:
        return 0.0
    reference = ' . '.join(key_facts)
    scores = _ROUGE.score(reference, response)
    return round((scores['rouge2'].fmeasure + scores['rougeL'].fmeasure) / 2, 3)


EVALUATORS = {
    TaskDomain.MATH: lambda resp, sample: evaluate_math(resp, sample['answer']),
    TaskDomain.READING: lambda resp, sample: evaluate_reading(resp, sample['answers']),
    TaskDomain.NEWS: lambda resp, sample: evaluate_news(resp, sample['key_facts']),
}


# ── Agents ───────────────────────────────────────────────────────────────────
def _wrap_blackboard(state_json: str, task: str) -> str:
    return SHARED_MEMORY_PREAMBLE.format(state=state_json) + task


def agent_planner(
    task_prompt: str,
    protocol: Protocol,
    domain: TaskDomain,
    logger: CommunicationLogger,
    run_id: str,
    client,
    model: str,
    seed: Optional[int] = None,
    shared: Optional[SharedMemory] = None,
) -> str:
    if protocol == Protocol.SHARED_MEMORY and shared is not None:
        prompt = _wrap_blackboard(
            shared.snapshot(),
            f'Decompose the task stored in the blackboard into 3-5 subtasks:'
        )
    else:
        prompt = f'Decompose this task into 3-5 subtasks:\n\n{task_prompt}'

    text, meta = llm_call(client, model, 'planner', prompt, protocol, domain, seed=seed)
    logger.log(Message(
        run_id=run_id, protocol=protocol.value, task_domain=domain.value,
        sender='Planner', receiver='Executor', content=text, **meta,
    ))
    return text


def agent_executor(
    plan: str,
    task_prompt: str,
    protocol: Protocol,
    domain: TaskDomain,
    logger: CommunicationLogger,
    run_id: str,
    client,
    model: str,
    seed: Optional[int] = None,
    shared: Optional[SharedMemory] = None,
) -> str:
    if protocol == Protocol.SHARED_MEMORY and shared is not None:
        prompt = _wrap_blackboard(
            shared.snapshot(),
            'Execute the plan stored in the blackboard using the task information '
            'also available there. Report your results.'
        )
    else:
        prompt = (
            f'Execute the following subtasks based on the information provided.\n\n'
            f'Plan:\n{plan}\n\n'
            f'Task Information:\n{task_prompt}'
        )

    text, meta = llm_call(client, model, 'executor', prompt, protocol, domain, seed=seed)
    logger.log(Message(
        run_id=run_id, protocol=protocol.value, task_domain=domain.value,
        sender='Executor', receiver='Integrator', content=text, **meta,
    ))
    return text


def agent_integrator(
    execution_result: str,
    protocol: Protocol,
    domain: TaskDomain,
    logger: CommunicationLogger,
    run_id: str,
    client,
    model: str,
    seed: Optional[int] = None,
    shared: Optional[SharedMemory] = None,
) -> str:
    if domain == TaskDomain.MATH:
        extra = ' Give the final numeric answer clearly at the end in the format: #### <number>'
    elif domain == TaskDomain.READING:
        extra = ' Give the final answer as a short, direct response.'
    else:
        extra = ' Summarize all key facts and figures from the analysis.'

    if protocol == Protocol.SHARED_MEMORY and shared is not None:
        prompt = _wrap_blackboard(
            shared.snapshot(),
            f'Synthesize the execution results stored in the blackboard into a '
            f'final coherent answer.{extra}'
        )
    else:
        prompt = (
            f'Synthesize the following execution results into a final coherent '
            f'answer.{extra}\n\nExecution Results:\n{execution_result}'
        )

    text, meta = llm_call(client, model, 'integrator', prompt, protocol, domain, seed=seed)
    logger.log(Message(
        run_id=run_id, protocol=protocol.value, task_domain=domain.value,
        sender='Integrator', receiver='Output', content=text, **meta,
    ))
    return text


# ── Pipeline runner ──────────────────────────────────────────────────────────
@dataclass
class RunResult:
    run_id: str
    protocol: str
    task_domain: str
    sample_index: int
    repetition: int
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int
    total_latency_ms: float
    total_messages: int
    any_truncation: bool
    any_json_parse_error: bool
    completion_score: float
    final_answer: str  # truncated to 200 chars


def run_pipeline(
    protocol: Protocol,
    domain: TaskDomain,
    sample: dict,
    sample_idx: int,
    client,
    model: str = 'gpt-4o-mini',
    seed: int = 0,
) -> Tuple[RunResult, List[Message]]:
    """Execute one pipeline run and return (summary, full message log)."""
    random.seed(seed)
    np.random.seed(seed)

    run_id = hashlib.md5(
        f'{protocol}{domain}{sample_idx}{seed}'.encode()
    ).hexdigest()[:8]
    logger = CommunicationLogger()
    shared = SharedMemory() if protocol == Protocol.SHARED_MEMORY else None

    task_prompt = TASK_BUILDERS[domain](sample)

    if protocol == Protocol.SHARED_MEMORY:
        shared.write('System', 'task_prompt', task_prompt)
        plan = agent_planner(task_prompt, protocol, domain, logger, run_id,
                             client, model, seed=seed, shared=shared)
        shared.write('Planner', 'plan', plan)

        exec_out = agent_executor(plan, task_prompt, protocol, domain, logger,
                                   run_id, client, model, seed=seed, shared=shared)
        shared.write('Executor', 'execution_result', exec_out)

        final = agent_integrator(exec_out, protocol, domain, logger, run_id,
                                  client, model, seed=seed, shared=shared)
        shared.write('Integrator', 'final_answer', final)
    else:
        plan = agent_planner(task_prompt, protocol, domain, logger, run_id,
                             client, model, seed=seed)
        exec_out = agent_executor(plan, task_prompt, protocol, domain, logger,
                                   run_id, client, model, seed=seed)
        final = agent_integrator(exec_out, protocol, domain, logger, run_id,
                                  client, model, seed=seed)

    score = EVALUATORS[domain](final, sample)
    s = logger.summary()

    result = RunResult(
        run_id=run_id,
        protocol=protocol.value,
        task_domain=domain.value,
        sample_index=sample_idx,
        repetition=seed,
        total_prompt_tokens=s.get('total_prompt_tokens', 0),
        total_completion_tokens=s.get('total_completion_tokens', 0),
        total_tokens=s.get('total_tokens', 0),
        total_latency_ms=s.get('total_latency_ms', 0.0),
        total_messages=s.get('total_messages', 0),
        any_truncation=s.get('any_truncation', False),
        any_json_parse_error=s.get('any_json_parse_error', False),
        completion_score=score,
        final_answer=final[:200],
    )
    return result, list(logger.messages)


# ── Self-tests (run when executed directly) ──────────────────────────────────
def _run_self_tests() -> None:
    assert evaluate_math('The answer is 18 dollars.', '18') == 1.0
    assert evaluate_math('The answer is 20.', '18') == 0.0

    neg = evaluate_reading('The building is not 187 feet tall.', ['187 feet'])
    pos = evaluate_reading('The building is 187 feet tall.', ['187 feet'])
    assert neg < 1.0 and pos > neg, f'reading: pos={pos} neg={neg}'
    assert evaluate_reading('715,522', ['715,522', '715522']) == 1.0

    if _ROUGE is not None:
        neg_n = evaluate_news(
            'S&P 500 did not rise. The Fed did not cut rates.',
            ['S&P 500 rose 0.5%', 'two potential rate cuts in 2025'],
        )
        pos_n = evaluate_news(
            'The S&P 500 rose 0.5%. The Fed signaled two potential rate cuts in 2025.',
            ['S&P 500 rose 0.5%', 'two potential rate cuts in 2025'],
        )
        assert pos_n > neg_n, f'news: pos={pos_n} neg={neg_n}'

    print('pipeline self-tests passed')


if __name__ == '__main__':
    _run_self_tests()
