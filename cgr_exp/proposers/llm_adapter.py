from __future__ import annotations

from dataclasses import dataclass
import http.client
import json
import os
import threading
import time
from typing import Sequence, Protocol
from urllib import error
from urllib import request

from cgr_exp.datasets.level1_scheduling import SchedulingTask
from cgr_exp.schemas import action_from_dict


class LLMClient(Protocol):
    def generate_typed_actions(self, *args, **kwargs):
        ...

    def generate_scheduling_solution(self, *args, **kwargs):
        ...

    def score_state(self, *args, **kwargs):
        ...

    def score_frontier_states(self, *args, **kwargs):
        ...


@dataclass
class MockLLMClient:
    raw_output: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency: float = 0.0

    def generate_typed_actions(self, *args, **kwargs):
        del args, kwargs
        return {
            "raw_output": self.raw_output,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency": self.latency,
        }

    def generate_scheduling_solution(self, *args, **kwargs):
        return self.generate_typed_actions(*args, **kwargs)

    def score_state(self, *args, **kwargs):
        del args, kwargs
        return {"score": 0.0, "input_tokens": 0, "output_tokens": 0, "latency": 0.0}

    def score_frontier_states(self, *args, **kwargs):
        nodes = kwargs.get("nodes", [])
        return {"scores": [0.0 for _ in nodes], "input_tokens": 0, "output_tokens": 0, "latency": 0.0}


@dataclass(frozen=True)
class EnvConfiguredLLMClient:
    provider: str
    base_url: str | None
    api_key_env: str
    api_key_present: bool

    @classmethod
    def from_env(cls, env=None, provider: str = "openai-compatible", api_key_env: str = "CGR_LLM_API_KEY") -> "EnvConfiguredLLMClient":
        source = os.environ if env is None else env
        return cls(
            provider=provider,
            base_url=source.get("CGR_LLM_BASE_URL"),
            api_key_env=api_key_env,
            api_key_present=bool(source.get(api_key_env)),
        )

    def to_safe_metadata(self):
        return {
            "provider": self.provider,
            "base_url": self.base_url,
            "api_key_env": self.api_key_env,
            "api_key_present": self.api_key_present,
        }

    def generate_typed_actions(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError("EnvConfiguredLLMClient is a provider-neutral configuration stub; use a concrete network client outside smoke/CI")

    def generate_scheduling_solution(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError("EnvConfiguredLLMClient is a provider-neutral configuration stub; use a concrete network client outside smoke/CI")

    def score_state(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError("EnvConfiguredLLMClient is a provider-neutral configuration stub; use a concrete network client outside smoke/CI")

    def score_frontier_states(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError("EnvConfiguredLLMClient is a provider-neutral configuration stub; use a concrete network client outside smoke/CI")


class OpenAICompatibleHTTPClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 120.0,
        max_retries: int = 3,
        retry_sleep_seconds: float = 2.0,
        urlopen=request.urlopen,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_sleep_seconds = retry_sleep_seconds
        self.urlopen = urlopen

    @classmethod
    def from_env(
        cls,
        env=None,
        base_url_env: str = "CGR_LLM_BASE_URL",
        api_key_env: str = "CGR_LLM_API_KEY",
        model_env: str = "CGR_LLM_MODEL",
    ) -> "OpenAICompatibleHTTPClient":
        source = os.environ if env is None else env
        api_key = source.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"missing LLM API key in {api_key_env}")
        return cls(
            base_url=source.get(base_url_env, "https://api.openai.com/v1"),
            api_key=api_key,
            model=source.get(model_env, "gpt-4.1-mini"),
        )

    def generate_typed_actions(self, *args, **kwargs):
        task = kwargs.get("task")
        node = kwargs.get("node")
        k = kwargs.get("k")
        model = kwargs.get("model") or self.model
        temperature = float(kwargs.get("temperature", 0.0))
        top_p = float(kwargs.get("top_p", 1.0))
        prompt_template_version = kwargs.get("prompt_template_version", "v1")
        prompt = self._prompt_for_actions(task, node, k, prompt_template_version)
        started = time.perf_counter()
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "Return only JSON actions following the provided typed action schema."},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "top_p": top_p,
            "response_format": {"type": "json_object"},
        }
        req = request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload, sort_keys=True).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        body = self._post_json(req)
        content = body["choices"][0]["message"]["content"]
        usage = body.get("usage", {})
        return {
            "raw_output": content,
            "input_tokens": int(usage.get("prompt_tokens", 0)),
            "output_tokens": int(usage.get("completion_tokens", 0)),
            "latency": time.perf_counter() - started,
        }

    def generate_scheduling_solution(self, *args, **kwargs):
        task = kwargs.get("task")
        model = kwargs.get("model") or self.model
        temperature = float(kwargs.get("temperature", 0.0))
        top_p = float(kwargs.get("top_p", 1.0))
        prompt_template_version = kwargs.get("prompt_template_version", "scheduling-native-v1")
        method = kwargs.get("method", "direct")
        sample_index = int(kwargs.get("sample_index", 0))
        prompt = self._prompt_for_scheduling_solution(task, prompt_template_version, method, sample_index)
        started = time.perf_counter()
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "Return only JSON for a complete verified scheduling plan. Include concise reasoning inside the JSON object, not as prose outside JSON."},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "top_p": top_p,
            "response_format": {"type": "json_object"},
        }
        req = request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload, sort_keys=True).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        body = self._post_json(req)
        content = body["choices"][0]["message"]["content"]
        usage = body.get("usage", {})
        return {
            "raw_output": content,
            "input_tokens": int(usage.get("prompt_tokens", 0)),
            "output_tokens": int(usage.get("completion_tokens", 0)),
            "latency": time.perf_counter() - started,
        }

    def score_state(self, *args, **kwargs):
        task = kwargs.get("task")
        node = kwargs.get("node")
        model = kwargs.get("model") or self.model
        started = time.perf_counter()
        prompt = self._prompt_for_state_score(task, node)
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "Score search frontier states for Tree-of-Thought style value guidance. Return only JSON."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "top_p": 1.0,
            "response_format": {"type": "json_object"},
        }
        req = request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload, sort_keys=True).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        body = self._post_json(req)
        content = body["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        score = max(0.0, min(1.0, float(parsed.get("score", 0.0))))
        usage = body.get("usage", {})
        return {
            "score": score,
            "input_tokens": int(usage.get("prompt_tokens", 0)),
            "output_tokens": int(usage.get("completion_tokens", 0)),
            "latency": time.perf_counter() - started,
            "raw_output": content,
        }

    def score_frontier_states(self, *args, **kwargs):
        task = kwargs.get("task")
        nodes = list(kwargs.get("nodes", []))
        model = kwargs.get("model") or self.model
        started = time.perf_counter()
        prompt = self._prompt_for_frontier_scores(task, nodes)
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "Rank search frontier states for Tree-of-Thought style value guidance. Return only JSON."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "top_p": 1.0,
            "response_format": {"type": "json_object"},
        }
        req = request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload, sort_keys=True).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        body = self._post_json(req)
        content = body["choices"][0]["message"]["content"]
        scores = _parse_frontier_scores(content, nodes)
        usage = body.get("usage", {})
        return {
            "scores": scores,
            "input_tokens": int(usage.get("prompt_tokens", 0)),
            "output_tokens": int(usage.get("completion_tokens", 0)),
            "latency": time.perf_counter() - started,
            "raw_output": content,
        }

    def rank_alfworld_actions(self, *args, **kwargs):
        objective = str(kwargs.get("objective", ""))
        observation = str(kwargs.get("observation", ""))
        history = list(kwargs.get("history", []))
        admissible = [str(item) for item in kwargs.get("admissible", [])]
        k = int(kwargs.get("k", 4))
        model = kwargs.get("model") or self.model
        temperature = float(kwargs.get("temperature", 0.0))
        top_p = float(kwargs.get("top_p", 1.0))
        started = time.perf_counter()
        prompt = json.dumps(
            {
                "task": "alfworld_admissible_action_shortlist",
                "instruction": (
                    "Choose the most promising next actions for completing the ALFWorld task. "
                    "Return JSON with ranked_indices, using zero-based indices from admissible_actions. "
                    "Return each selected index at most once. Do not invent actions."
                ),
                "response_schema": {"ranked_indices": ["integer index from admissible_actions, best first"]},
                "objective": objective,
                "observation": observation,
                "recent_history": history[-12:],
                "k": k,
                "admissible_actions": [{"index": idx, "command": command} for idx, command in enumerate(admissible)],
            },
            sort_keys=True,
        )
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "Rank ALFWorld admissible actions. Return only JSON."},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "top_p": top_p,
            "response_format": {"type": "json_object"},
        }
        req = request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload, sort_keys=True).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        body = self._post_json(req)
        content = body["choices"][0]["message"]["content"]
        usage = body.get("usage", {})
        ranked_indices = _parse_ranked_indices(content, len(admissible))
        return {
            "ranked_indices": ranked_indices[: max(1, k)],
            "input_tokens": int(usage.get("prompt_tokens", 0)),
            "output_tokens": int(usage.get("completion_tokens", 0)),
            "latency": time.perf_counter() - started,
            "raw_output": content,
        }

    def _prompt_for_state_score(self, task, node) -> str:
        if isinstance(task, SchedulingTask) and hasattr(node, "structured_state"):
            scheduled = tuple(node.structured_state.get("scheduled", []))
            return json.dumps(
                {
                    "task": "precedence_constrained_scheduling_state_value",
                    "task_id": task.task_id,
                    "node_id": getattr(node, "node_id", None),
                    "instruction": (
                        "Return a JSON object with score in [0, 1]. Higher means this partial schedule is more promising "
                        "for finding an optimal or near-optimal complete schedule within the remaining search budget. "
                        "Use only the visible scheduling data and current partial schedule; do not invent hidden oracle labels."
                    ),
                    "response_schema": {"score": "number between 0 and 1"},
                    "jobs": {
                        job: {
                            "duration": task.durations[job],
                            "release_time": task.release_times[job],
                            "due_date": task.due_dates[job],
                            "allowed_machines": list(task.allowed[job]),
                            "predecessors": list(task.predecessors.get(job, ())),
                        }
                        for job in task.durations
                    },
                    "machines": list(task.machines),
                    "partial_schedule": list(scheduled),
                    "machine_loads": node.structured_state.get("machine_loads", {}),
                    "eligible_jobs": task.ready_jobs(scheduled),
                    "depth": getattr(node, "depth", 0),
                    "lower_bound": getattr(node, "lower_bound", None),
                    "incumbent_at_creation": getattr(node, "incumbent_at_creation", None),
                },
                sort_keys=True,
            )
        return json.dumps(
            {
                "instruction": "Return a JSON object with score in [0, 1]. Higher means this state is more promising for reaching the task goal.",
                "response_schema": {"score": "number between 0 and 1"},
                "node": _safe_jsonable(node),
            },
            sort_keys=True,
        )

    def _prompt_for_frontier_scores(self, task, nodes: Sequence[object]) -> str:
        if isinstance(task, SchedulingTask):
            candidates = []
            for idx, node in enumerate(nodes):
                scheduled = tuple(getattr(node, "structured_state", {}).get("scheduled", []))
                candidates.append(
                    {
                        "candidate_id": str(getattr(node, "node_id", f"candidate-{idx}")),
                        "partial_schedule": list(scheduled),
                        "machine_loads": getattr(node, "structured_state", {}).get("machine_loads", {}),
                        "eligible_jobs": task.ready_jobs(scheduled),
                        "depth": getattr(node, "depth", 0),
                        "lower_bound": getattr(node, "lower_bound", None),
                        "incumbent_at_creation": getattr(node, "incumbent_at_creation", None),
                    }
                )
            return json.dumps(
                {
                    "task": "precedence_constrained_scheduling_frontier_ranking",
                    "task_id": task.task_id,
                    "instruction": (
                        "Rank all candidate frontier states from most to least promising for finding an optimal or near-optimal "
                        "complete schedule within the remaining search budget. Return each candidate_id exactly once. "
                        "Use only the visible scheduling data and candidate states; do not invent hidden oracle labels."
                    ),
                    "response_schema": {"ranking": ["candidate_id from most promising to least promising"]},
                    "jobs": {
                        job: {
                            "duration": task.durations[job],
                            "release_time": task.release_times[job],
                            "due_date": task.due_dates[job],
                            "allowed_machines": list(task.allowed[job]),
                            "predecessors": list(task.predecessors.get(job, ())),
                        }
                        for job in task.durations
                    },
                    "machines": list(task.machines),
                    "candidates": candidates,
                },
                sort_keys=True,
            )
        return json.dumps(
            {
                "instruction": "Rank all candidate frontier states from most to least promising. Return each candidate_id exactly once.",
                "response_schema": {"ranking": ["candidate_id from most promising to least promising"]},
                "candidates": [
                    {"candidate_id": str(getattr(node, "node_id", f"candidate-{idx}")), "node": _safe_jsonable(node)}
                    for idx, node in enumerate(nodes)
                ],
            },
            sort_keys=True,
        )

    def _post_json(self, req):
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                with self.urlopen(req, timeout=self.timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except (TimeoutError, error.URLError, http.client.RemoteDisconnected, http.client.IncompleteRead) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(self.retry_sleep_seconds * (attempt + 1))
        raise RuntimeError(f"LLM request failed after {self.max_retries + 1} attempts: {last_error}") from last_error

    def _prompt_for_scheduling_solution(self, task, prompt_template_version: str, method: str, sample_index: int) -> str:
        if isinstance(task, SchedulingTask):
            return json.dumps(
                {
                    "prompt_template_version": prompt_template_version,
                    "task": "precedence_constrained_scheduling_complete_solution",
                    "task_id": task.task_id,
                    "method": method,
                    "sample_index": sample_index,
                    "instruction": (
                        "Think step by step about precedence, release times, machine eligibility, and tardiness, then produce one complete feasible schedule. "
                        "Return only a JSON object with a concise reasoning array and an actions array. "
                        "Each action must schedule one currently eligible unscheduled job on an allowed machine. "
                        "The action order must respect all predecessor constraints. Try to minimize makespan plus total tardiness."
                    ),
                    "objective": "minimize makespan + total tardiness",
                    "jobs": {
                        job: {
                            "duration": task.durations[job],
                            "release_time": task.release_times[job],
                            "due_date": task.due_dates[job],
                            "allowed_machines": list(task.allowed[job]),
                            "predecessors": list(task.predecessors.get(job, ())),
                        }
                        for job in task.durations
                    },
                    "machines": list(task.machines),
                    "response_schema": {
                        "reasoning": ["brief step-by-step scheduling rationale"],
                        "actions": [{"action_type": "schedule", "job": "job id", "machine": "allowed machine id"}],
                    },
                    "constraints": [
                        "schedule every job exactly once",
                        "choose only allowed machines",
                        "respect predecessor order in the returned action sequence",
                        "do not include starts or finishes; the verifier computes them deterministically",
                    ],
                },
                sort_keys=True,
            )
        return json.dumps(
            {
                "prompt_template_version": prompt_template_version,
                "task": _safe_jsonable(task),
                "instruction": "Think step by step, then return only JSON with reasoning and actions for one complete solution.",
                "response_schema": {
                    "reasoning": ["brief step-by-step rationale"],
                    "actions": [{"action_type": "schedule", "job": "job id", "machine": "machine id"}],
                },
            },
            sort_keys=True,
        )

    def _prompt_for_actions(self, task, node, k, prompt_template_version: str) -> str:
        if getattr(task, "task_id", "").startswith("game24-") and hasattr(node, "structured_state"):
            state = node.structured_state
            return json.dumps(
                {
                    "prompt_template_version": prompt_template_version,
                    "task": "Game24",
                    "target": 24,
                    "task_id": getattr(task, "task_id", None),
                    "node_id": getattr(node, "node_id", None),
                    "current_values": state.get("values", []),
                    "current_expressions": state.get("expressions", []),
                    "instruction": (
                        "Propose up to k legal next arithmetic operations. Each action must combine exactly two numbers "
                        "that currently appear in current_values, consuming one occurrence of each. Do not use numbers "
                        "from prior states or intermediate results unless they appear in current_values. Return only JSON."
                    ),
                    "k": k,
                    "operators": ["+", "-", "*", "/"],
                    "response_schema": {"actions": [{"action_type": "game24_op", "left": "number from current_values", "right": "number from current_values", "operator": "+|-|*|/"}]},
                },
                sort_keys=True,
            )
        if isinstance(task, SchedulingTask) and hasattr(node, "structured_state"):
            scheduled = tuple(node.structured_state.get("scheduled", []))
            ready_jobs = task.ready_jobs(scheduled)
            return json.dumps(
                {
                    "prompt_template_version": prompt_template_version,
                    "task": "precedence_constrained_scheduling",
                    "task_id": task.task_id,
                    "node_id": getattr(node, "node_id", None),
                    "instruction": (
                        "Propose up to k valid next scheduling actions for the current partial schedule. "
                        "Return only JSON with an actions array. Each action must choose one currently eligible job "
                        "and one allowed machine for that job. Do not include explanations."
                    ),
                    "k": k,
                    "jobs": {
                        job: {
                            "duration": task.durations[job],
                            "release_time": task.release_times[job],
                            "due_date": task.due_dates[job],
                            "allowed_machines": list(task.allowed[job]),
                            "predecessors": list(task.predecessors.get(job, ())),
                        }
                        for job in task.durations
                    },
                    "machines": list(task.machines),
                    "partial_schedule": list(scheduled),
                    "machine_loads": node.structured_state.get("machine_loads", {}),
                    "eligible_jobs": ready_jobs,
                    "response_schema": {"actions": [{"action_type": "schedule", "job": "eligible job id", "machine": "allowed machine id"}]},
                },
                sort_keys=True,
            )
        return json.dumps(
            {
                "prompt_template_version": prompt_template_version,
                "instruction": "Propose up to k typed actions. Return a JSON object with an actions array.",
                "k": k,
                "allowed_schema_examples": [
                    {"action_type": "assign", "task": "T0", "resource": "R0"},
                    {"action_type": "game24_op", "left": 10, "right": 4, "operator": "-"},
                ],
                "response_schema": {"actions": [{"action_type": "game24_op", "left": "number", "right": "number", "operator": "+|-|*|/"}]},
                "task": _safe_jsonable(task),
                "node": _safe_jsonable(node),
            },
            sort_keys=True,
        )


def _safe_jsonable(value):
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _safe_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_jsonable(v) for v in value]
    return repr(value)


def _parse_frontier_scores(raw: str, nodes: Sequence[object]) -> list[float]:
    node_ids = [str(getattr(node, "node_id", f"candidate-{idx}")) for idx, node in enumerate(nodes)]
    fallback = [float(len(nodes) - idx) for idx in range(len(nodes))]
    try:
        parsed = json.loads(raw)
    except Exception:
        return fallback
    if isinstance(parsed, dict) and isinstance(parsed.get("scores"), dict):
        score_map = parsed["scores"]
        scores = []
        for node_id in node_ids:
            try:
                scores.append(float(score_map.get(node_id, 0.0)))
            except (TypeError, ValueError):
                scores.append(0.0)
        return scores
    ranking = parsed.get("ranking") if isinstance(parsed, dict) else parsed
    if not isinstance(ranking, list):
        return fallback
    rank_map = {str(node_id): float(len(ranking) - idx) for idx, node_id in enumerate(ranking)}
    return [rank_map.get(node_id, 0.0) for node_id in node_ids]


def _parse_ranked_indices(raw: str, candidate_count: int) -> list[int]:
    fallback = list(range(candidate_count))
    try:
        parsed = json.loads(raw)
    except Exception:
        return fallback
    ranking = parsed.get("ranked_indices") if isinstance(parsed, dict) else parsed
    if ranking is None and isinstance(parsed, dict):
        ranking = parsed.get("ranking")
    if not isinstance(ranking, list):
        return fallback
    raw_indices: list[int] = []
    for item in ranking:
        value = item
        if isinstance(item, dict):
            value = item.get("index", item.get("action_index", item.get("candidate_index")))
        try:
            raw_indices.append(int(value))
        except (TypeError, ValueError):
            continue
    if raw_indices and min(raw_indices) >= 1 and max(raw_indices) == candidate_count:
        raw_indices = [idx - 1 for idx in raw_indices]
    clean: list[int] = []
    seen: set[int] = set()
    for idx in raw_indices:
        if 0 <= idx < candidate_count and idx not in seen:
            clean.append(idx)
            seen.add(idx)
    clean.extend(idx for idx in fallback if idx not in seen)
    return clean


class RealLLMProposerAdapter:
    def __init__(self, client: LLMClient, model: str, prompt_template_version: str, temperature: float = 0.0, top_p: float = 1.0):
        self.client = client
        self.model = model
        self.prompt_template_version = prompt_template_version
        self.temperature = temperature
        self.top_p = top_p
        self.call_records = []
        self._lock = threading.Lock()

    def propose(self, task, node, k, budget_state=None):
        actions, _ = self.propose_with_record(task, node, k, budget_state=budget_state)
        return actions

    def propose_with_record(self, task, node, k, budget_state=None):
        response = self.client.generate_typed_actions(
            task=task,
            node=node,
            k=k,
            model=self.model,
            prompt_template_version=self.prompt_template_version,
            temperature=self.temperature,
            top_p=self.top_p,
            budget_state=budget_state,
        )
        if budget_state is not None:
            budget_state.charge_proposal(
                input_tokens=int(response.get("input_tokens", 0)),
                output_tokens=int(response.get("output_tokens", 0)),
            )
        raw = response.get("raw_output", "")
        actions, parse_failures = self._parse_actions(raw, k)
        record = {
            "model": self.model,
            "prompt_template_version": self.prompt_template_version,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "k": k,
            "input_tokens": response.get("input_tokens", 0),
            "output_tokens": response.get("output_tokens", 0),
            "latency": response.get("latency", 0.0),
            "raw_output": raw,
            "parsed_action_count": len(actions),
            "parse_failures": parse_failures,
        }
        with self._lock:
            self.call_records.append(record)
        return actions, record

    def _parse_actions(self, raw, k):
        actions = []
        parse_failures = 0
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict) and "actions" in parsed:
                parsed = parsed["actions"]
            if not isinstance(parsed, list):
                raise ValueError("expected list or object with actions")
            for item in parsed[:k]:
                if not isinstance(item, dict):
                    parse_failures += 1
                    continue
                action = action_from_dict(item)
                if action is not None and action.validate_schema().ok and action.action_type != "unknown":
                    actions.append(action)
                else:
                    parse_failures += 1
        except Exception:
            parse_failures += 1
            actions = []
        return actions, parse_failures
