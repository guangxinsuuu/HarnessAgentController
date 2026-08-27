from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
from pathlib import Path
import threading
from typing import Dict, Iterable, List, Optional, Sequence

from cgr_exp.budgets import BudgetState
from cgr_exp.datasets.game24 import Game24Action, Game24Task, game24_feature_vector
from cgr_exp.game24_experiments import Game24VectorPolicy, generate_game24_ranking_pairs, generate_game24_tasks
from cgr_exp.proposers.llm_adapter import OpenAICompatibleHTTPClient, RealLLMProposerAdapter
from cgr_exp.ranker import LinearRanker
from cgr_exp.schemas import CompletionStatus, CostLedger, SearchNode, action_from_dict
from cgr_exp.search.dfs import DFSPolicy
from cgr_exp.search.oracle_policy import OraclePolicy
from cgr_exp.verification.game24_verifier import Game24Verifier


class ScriptedGame24Client:
    def __init__(self):
        self.proposal_calls = 0
        self.value_calls = 0

    def generate_typed_actions(self, *args, **kwargs):
        node = kwargs["node"]
        values = [float(v) for v in node.structured_state["values"]]
        actions = _scripted_actions(values, int(kwargs.get("k", 4)))
        self.proposal_calls += 1
        return {
            "raw_output": json.dumps({"actions": [action.to_dict() for action in actions]}),
            "input_tokens": 20,
            "output_tokens": 8 * len(actions),
            "latency": 0.0,
        }

    def score_state(self, *args, **kwargs):
        node = kwargs["node"]
        values = [float(v) for v in node.structured_state["values"]]
        closest = min(abs(v - 24.0) for v in values)
        self.value_calls += 1
        return {"score": 1.0 / (1.0 + closest), "input_tokens": 12, "output_tokens": 2, "latency": 0.0}


class SharedProposalCacheProposer:
    def __init__(
        self,
        adapter: RealLLMProposerAdapter,
        proposal_k: int,
        cache_path: Optional[Path] = None,
        progress_path: Optional[Path] = None,
        resume: bool = False,
    ):
        self.adapter = adapter
        self.proposal_k = proposal_k
        self.cache_path = cache_path
        self.progress_path = progress_path
        self.cache: Dict[str, List[Game24Action]] = {}
        self.records: List[Dict[str, object]] = []
        self.cost = CostLedger()
        self._lock = threading.Lock()
        if self.cache_path is not None:
            if resume and self.cache_path.exists():
                self._load_existing_cache()
            else:
                self.cache_path.write_text("", encoding="utf-8")
        self._write_progress()

    def propose(self, task: Game24Task, node: SearchNode, k: int, budget_state=None):
        del k, budget_state
        with self._lock:
            if node.node_id in self.cache:
                return list(self.cache[node.node_id])
        budget = BudgetState()
        proposed, call_record = self.adapter.propose_with_record(task, node, self.proposal_k, budget_state=budget)
        actions = [action for action in proposed if isinstance(action, Game24Action)]
        valid_actions = [action for action in actions if task.is_valid_action(node, action)]
        record = dict(call_record)
        record["task_id"] = task.task_id
        record["node_id"] = node.node_id
        record["actions"] = [action.to_dict() for action in valid_actions]
        record["proposal_recall_at_k"] = _proposal_recall(task, node, valid_actions)
        with self._lock:
            if node.node_id in self.cache:
                return list(self.cache[node.node_id])
            self.cache[node.node_id] = valid_actions
            self.cost = self.cost + budget.cost
            self.records.append(record)
            if self.cache_path is not None:
                with self.cache_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
            self._write_progress_locked()
        return list(valid_actions)

    def _load_existing_cache(self) -> None:
        assert self.cache_path is not None
        for line in self.cache_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            actions = []
            for item in record.get("actions", []):
                action = action_from_dict(item)
                if isinstance(action, Game24Action):
                    actions.append(action)
            node_id = str(record["node_id"])
            self.cache[node_id] = actions
            self.records.append(record)
            self.cost = self.cost + CostLedger(
                proposal_calls=1,
                input_tokens=int(record.get("input_tokens", 0)),
                output_tokens=int(record.get("output_tokens", 0)),
            )

    def _write_progress(self) -> None:
        if self.progress_path is None:
            return
        self.progress_path.write_text(
            json.dumps(
                {
                    "stage": "building_or_using_shared_proposal_cache",
                    "proposal_cache_states": len(self.cache),
                    "proposal_calls": self.cost.proposal_calls,
                    "input_tokens": self.cost.input_tokens,
                    "output_tokens": self.cost.output_tokens,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    def _write_progress_locked(self) -> None:
        self._write_progress()


class CachedProposalReplay:
    def __init__(self, cache: SharedProposalCacheProposer):
        self.cache = cache

    def propose(self, task, node, k, budget_state=None):
        del k, budget_state
        return list(self.cache.cache.get(node.node_id, []))


class ToTValuePolicy:
    def __init__(self, client, model: str, batch_size: int = 1):
        self.client = client
        self.model = model
        self.batch_size = max(1, int(batch_size))
        self.cost = CostLedger()
        self.cache: Dict[str, float] = {}

    def score(self, node: SearchNode) -> float:
        if node.node_id in self.cache:
            return self.cache[node.node_id]
        response = self.client.score_state(node=node, model=self.model)
        self.cost = self.cost + CostLedger(
            evaluation_calls=1,
            input_tokens=int(response.get("input_tokens", 0)),
            output_tokens=int(response.get("output_tokens", 0)),
        )
        score = float(response.get("score", 0.0))
        self.cache[node.node_id] = score
        return score

    def score_many(self, nodes: Sequence[SearchNode]) -> List[float]:
        missing = [node for node in nodes if node.node_id not in self.cache]
        if self.batch_size <= 1 or len(missing) <= 1:
            for node in missing:
                self.score(node)
            return [self.cache[node.node_id] for node in nodes]
        with ThreadPoolExecutor(max_workers=min(self.batch_size, len(missing))) as executor:
            future_to_node = {executor.submit(self.client.score_state, node=node, model=self.model): node for node in missing}
            for future in as_completed(future_to_node):
                node = future_to_node[future]
                response = future.result()
                self.cost = self.cost + CostLedger(
                    evaluation_calls=1,
                    input_tokens=int(response.get("input_tokens", 0)),
                    output_tokens=int(response.get("output_tokens", 0)),
                )
                self.cache[node.node_id] = float(response.get("score", 0.0))
        return [self.cache[node.node_id] for node in nodes]


def run_game24_llm_pilot(
    out_dir: Path,
    client,
    model: str,
    seed: int = 0,
    start: int = 900,
    count: int = 20,
    proposal_k: int = 4,
    node_budget: int = 80,
    batch_size: int = 1,
    resume: bool = True,
) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    all_tasks = generate_game24_tasks(limit=max(1000, start + count), seed=seed)
    train_tasks = all_tasks[:700]
    pilot_tasks = all_tasks[start : start + count]
    pairs = generate_game24_ranking_pairs(train_tasks[:80], node_budget=80)
    vector_pairs = [(pair.positive_features, pair.negative_features) for pair in pairs]
    ranker = LinearRanker.fresh(len(vector_pairs[0][0]) if vector_pairs else 17)
    ranker.train(vector_pairs, epochs=8, lr=0.04, seed=seed)

    adapter = RealLLMProposerAdapter(client=client, model=model, prompt_template_version="game24-step7-v1", temperature=0.0, top_p=1.0)
    shared_cache = SharedProposalCacheProposer(
        adapter,
        proposal_k=proposal_k,
        cache_path=out_dir / "game24_proposal_cache.jsonl",
        progress_path=out_dir / "progress.json",
        resume=resume,
    )
    _populate_shared_cache(pilot_tasks, shared_cache, node_budget=node_budget, proposal_k=proposal_k, batch_size=batch_size)
    replay = CachedProposalReplay(shared_cache)

    methods = {
        "CoT": DFSPolicy(),
        "ToT-style": ToTValuePolicy(client, model, batch_size=batch_size),
        "MLP priority": Game24VectorPolicy(ranker),
        "Oracle priority": None,
    }
    method_rows = []
    method_metrics = {}
    for method, policy in methods.items():
        rows = []
        policy_obj = policy
        for task in pilot_tasks:
            if policy_obj is None:
                policy_obj = OraclePolicy(task)
            result = _run_cached_game24_search(task, policy_obj, replay, node_budget=node_budget, proposal_k=proposal_k)
            rows.append(result)
        value_cost = policy.cost if isinstance(policy, ToTValuePolicy) else CostLedger()
        summary = _summarize_llm_rows(rows, proposal_cost=shared_cache.cost, value_cost=value_cost)
        method_metrics[method] = summary
        method_rows.append({"Method": method, **summary})

    _write_csv(out_dir / "game24_llm_pilot_table.csv", method_rows)
    _write_jsonl(out_dir / "game24_proposal_cache.jsonl", shared_cache.records)
    metrics = {
        "model": model,
        "seed": seed,
        "start": start,
        "count": count,
        "proposal_k": proposal_k,
        "node_budget": node_budget,
        "batch_size": batch_size,
        "live_network_required": not isinstance(client, ScriptedGame24Client),
        "proposal_cache": {
            "shared": True,
            "state_count": len(shared_cache.cache),
            "proposal_calls": shared_cache.cost.proposal_calls,
            "input_tokens": shared_cache.cost.input_tokens,
            "output_tokens": shared_cache.cost.output_tokens,
            "proposal_recall_at_k": _mean(float(record["proposal_recall_at_k"]) for record in shared_cache.records),
        },
        "methods": method_metrics,
    }
    (out_dir / "game24_llm_pilot_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return metrics


def run_game24_llm_pilot_from_env(
    out_dir: Path,
    model: Optional[str],
    seed: int,
    start: int,
    count: int,
    proposal_k: int,
    node_budget: int,
    timeout: float = 120.0,
    batch_size: int = 1,
) -> Dict[str, object]:
    client = OpenAICompatibleHTTPClient.from_env()
    client.timeout = timeout
    return run_game24_llm_pilot(out_dir, client=client, model=model or client.model, seed=seed, start=start, count=count, proposal_k=proposal_k, node_budget=node_budget, batch_size=batch_size)


def _populate_shared_cache(tasks: Sequence[Game24Task], cache: SharedProposalCacheProposer, node_budget: int, proposal_k: int, batch_size: int = 1) -> None:
    if batch_size <= 1 or len(tasks) <= 1:
        for task in tasks:
            _populate_one_task_cache(task, cache, node_budget=node_budget, proposal_k=proposal_k)
        return
    with ThreadPoolExecutor(max_workers=min(batch_size, len(tasks))) as executor:
        futures = [executor.submit(_populate_one_task_cache, task, cache, node_budget, proposal_k) for task in tasks]
        for future in as_completed(futures):
            future.result()


def _populate_one_task_cache(task: Game24Task, cache: SharedProposalCacheProposer, node_budget: int, proposal_k: int) -> None:
    frontier = [task.root_node()]
    expanded = 0
    seen = set()
    while frontier and expanded < node_budget:
        node = frontier.pop()
        if node.node_id in seen or node.completion_status == CompletionStatus.COMPLETE:
            continue
        seen.add(node.node_id)
        expanded += 1
        actions = cache.propose(task, node, proposal_k)
        for idx, action in enumerate(actions):
            child = task.apply(node, action, idx)
            if child.completion_status != CompletionStatus.COMPLETE:
                frontier.append(child)


def _run_cached_game24_search(task: Game24Task, policy, proposer: CachedProposalReplay, node_budget: int, proposal_k: int) -> Dict[str, object]:
    verifier = Game24Verifier()
    frontier = [task.root_node()]
    expanded = 0
    solved = False
    proposal_calls = 0
    while frontier and expanded < node_budget and not solved:
        if hasattr(policy, "score_many"):
            scores = policy.score_many(frontier)
            idx = max(range(len(frontier)), key=lambda i: (scores[i], frontier[i].node_id))
        else:
            idx = max(range(len(frontier)), key=lambda i: (policy.score(frontier[i]), frontier[i].node_id))
        node = frontier.pop(idx)
        expanded += 1
        if node.completion_status == CompletionStatus.COMPLETE:
            solved = node.objective_if_complete == 0.0
            continue
        proposal_calls += 1
        for child_index, action in enumerate(proposer.propose(task, node, proposal_k)):
            result = verifier.verify(task, node, action)
            if not result.feasible:
                continue
            child = verifier.make_child(task, node, action, result, child_index)
            if child.completion_status == CompletionStatus.COMPLETE and child.objective_if_complete == 0.0:
                solved = True
                break
            if child.completion_status != CompletionStatus.COMPLETE:
                frontier.append(child)
    return {"task_id": task.task_id, "success": solved, "expanded_nodes": expanded, "proposal_calls": proposal_calls}


def _proposal_recall(task: Game24Task, node: SearchNode, actions: Sequence[Game24Action]) -> float:
    current = task.min_future_expansions(node)
    if current is None:
        return 1.0
    for idx, action in enumerate(actions):
        child = task.apply(node, action, idx)
        child_steps = task.min_future_expansions(child)
        if child_steps is not None and child_steps < current:
            return 1.0
    return 0.0


def _scripted_actions(values: Sequence[float], k: int) -> List[Game24Action]:
    task = Game24Task("scripted", tuple(int(v) for v in values))
    return task.candidate_actions(task.root_node(), k)


def _summarize_llm_rows(rows: Sequence[Dict[str, object]], proposal_cost: CostLedger, value_cost: CostLedger) -> Dict[str, object]:
    total_tokens = proposal_cost.input_tokens + proposal_cost.output_tokens + value_cost.input_tokens + value_cost.output_tokens
    return {
        "success": _mean(1.0 if row["success"] else 0.0 for row in rows),
        "expanded_nodes": _mean(float(row["expanded_nodes"]) for row in rows),
        "proposal_calls": _mean(float(row["proposal_calls"]) for row in rows),
        "value_calls": value_cost.evaluation_calls,
        "proposal_tokens": proposal_cost.input_tokens + proposal_cost.output_tokens,
        "value_tokens": value_cost.input_tokens + value_cost.output_tokens,
        "total_tokens": total_tokens,
    }


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _mean(values: Iterable[float]) -> float:
    data = list(values)
    return 0.0 if not data else sum(data) / len(data)
