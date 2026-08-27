from __future__ import annotations

from dataclasses import dataclass
import http.client
import json
import os
import threading
import time
from typing import Protocol
from urllib import error
from urllib import request

from cgr_exp.datasets.level1_scheduling import SchedulingTask
from cgr_exp.schemas import action_from_dict


class LLMClient(Protocol):
    def generate_typed_actions(self, *args, **kwargs):
        ...

    def score_state(self, *args, **kwargs):
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

    def score_state(self, *args, **kwargs):
        del args, kwargs
        return {"score": 0.0, "input_tokens": 0, "output_tokens": 0, "latency": 0.0}


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

    def score_state(self, *args, **kwargs):
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

    def score_state(self, *args, **kwargs):
        node = kwargs.get("node")
        model = kwargs.get("model") or self.model
        started = time.perf_counter()
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "Score Game24 search states for Tree-of-Thought style value guidance. Return only JSON."},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "instruction": "Return a JSON object with score in [0, 1]. Higher means this state is more promising for reaching 24.",
                            "response_schema": {"score": "number between 0 and 1"},
                            "node": _safe_jsonable(node),
                        },
                        sort_keys=True,
                    ),
                },
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

    def _post_json(self, req):
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                with self.urlopen(req, timeout=self.timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except (TimeoutError, error.URLError, http.client.RemoteDisconnected) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(self.retry_sleep_seconds * (attempt + 1))
        raise RuntimeError(f"LLM request failed after {self.max_retries + 1} attempts: {last_error}") from last_error

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
