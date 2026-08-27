from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from typing import Dict, List, Optional, Tuple

from cgr_exp.schemas import CompletionStatus, CostLedger, FeasibleStatus, SearchNode, ValidationResult


TARGET = 24.0
EPS = 1e-9


@dataclass(frozen=True)
class Game24Action:
    left: float
    right: float
    operator: str
    action_type: str = "game24_op"

    def validate_schema(self) -> ValidationResult:
        errors: List[str] = []
        if self.operator not in {"+", "-", "*", "/"}:
            errors.append("operator must be one of + - * /")
        if not isinstance(self.left, (int, float)):
            errors.append("left must be numeric")
        if not isinstance(self.right, (int, float)):
            errors.append("right must be numeric")
        if self.operator == "/" and abs(float(self.right)) <= EPS:
            errors.append("divide-by-zero")
        return ValidationResult(not errors, errors)

    def to_dict(self) -> Dict[str, object]:
        return {"action_type": self.action_type, "left": self.left, "right": self.right, "operator": self.operator}


@dataclass(frozen=True)
class Game24Task:
    task_id: str
    numbers: Tuple[int, ...]

    def root_node(self) -> SearchNode:
        expressions = tuple(str(n) for n in self.numbers)
        return self._make_node(_canonical_values(tuple(float(n) for n in self.numbers)), expressions, None, None, f"{self.task_id}:root")

    def candidate_actions(self, node: SearchNode, k: int) -> List[Game24Action]:
        values = tuple(float(v) for v in node.structured_state["values"])
        return _all_actions(values)[:k]

    def apply(self, node: SearchNode, action: Game24Action, child_index: int = 0) -> SearchNode:
        if not self.is_valid_action(node, action):
            return node.with_updates(feasible_status=FeasibleStatus.INFEASIBLE)
        values = list(float(v) for v in node.structured_state["values"])
        expressions = list(str(v) for v in node.structured_state["expressions"])
        li = _find_value_index(values, action.left)
        left_expr = expressions.pop(li)
        left_value = values.pop(li)
        ri = _find_value_index(values, action.right)
        right_expr = expressions.pop(ri)
        right_value = values.pop(ri)
        result = _apply_operator(left_value, right_value, action.operator)
        values.append(result)
        expressions.append(f"({left_expr}{action.operator}{right_expr})")
        paired = sorted(zip(values, expressions), key=lambda item: (round(item[0], 8), item[1]))
        next_values = tuple(value for value, _ in paired)
        next_expressions = tuple(expr for _, expr in paired)
        return self._make_node(
            next_values,
            next_expressions,
            parent_id=node.node_id,
            action=action,
            node_id=f"{node.node_id}/{child_index}:{_format_number(action.left)}{action.operator}{_format_number(action.right)}",
        )

    def is_valid_action(self, node: SearchNode, action: Game24Action) -> bool:
        if not action.validate_schema().ok:
            return False
        values = list(float(v) for v in node.structured_state["values"])
        li = _find_value_index(values, action.left)
        if li < 0:
            return False
        values.pop(li)
        return _find_value_index(values, action.right) >= 0

    def can_reach_24(self, node: SearchNode) -> bool:
        return _min_future_expansions(tuple(float(v) for v in node.structured_state["values"])) is not None

    def min_future_expansions(self, node: SearchNode) -> Optional[int]:
        return _min_future_expansions(tuple(float(v) for v in node.structured_state["values"]))

    def best_descendant_objective(self, node: SearchNode) -> float:
        steps = self.min_future_expansions(node)
        return float(steps) if steps is not None else float("inf")

    def exhaustive_oracle(self) -> float:
        steps = _min_future_expansions(_canonical_values(tuple(float(n) for n in self.numbers)))
        return float(steps) if steps is not None else float("inf")

    def lower_bound(self, node: SearchNode) -> float:
        del node
        return 0.0

    def _make_node(
        self,
        values: Tuple[float, ...],
        expressions: Tuple[str, ...],
        parent_id: Optional[str],
        action: Optional[Game24Action],
        node_id: str,
    ) -> SearchNode:
        complete = len(values) == 1
        solved = complete and abs(values[0] - TARGET) <= EPS
        feature_names, feature_values = _game24_features(values, action)
        return SearchNode(
            node_id=node_id,
            parent_id=parent_id,
            depth=len(self.numbers) - len(values),
            task_id=self.task_id,
            semantic_state={"target": TARGET, "family": "game24"},
            structured_state={"values": list(values), "expressions": list(expressions), "expression_count": len(expressions)},
            verified_constraints=[] if action is None else [f"game24:{_format_number(action.left)}{action.operator}{_format_number(action.right)}"],
            uncertain_constraints=[],
            lower_bound=0.0,
            incumbent_at_creation=None,
            objective_if_complete=0.0 if solved else (1.0 if complete else None),
            feasible_status=FeasibleStatus.FEASIBLE,
            completion_status=CompletionStatus.COMPLETE if complete else CompletionStatus.PARTIAL,
            proposal_action=action,
            observation=None,
            universal_features={"depth": float(len(self.numbers) - len(values)), "branch_factor": float(len(_all_actions(values)))},
            verifier_features={"lower_bound": 0.0, "verified_constraint_count": float(len(self.numbers) - len(values)), "uncertain_constraint_count": 0.0},
            semantic_features=feature_values,
            missingness_mask={"lower_bound": False, "semantic_features": False},
            accumulated_expensive_cost=CostLedger(),
            metadata={"family": "game24", "feature_names": feature_names},
        )


def game24_feature_vector(task: Game24Task, node: SearchNode) -> Tuple[List[str], List[float]]:
    del task
    values = tuple(float(v) for v in node.structured_state["values"])
    action = node.proposal_action if isinstance(node.proposal_action, Game24Action) else None
    return _game24_features(values, action)


def _game24_features(values: Tuple[float, ...], action: Optional[Game24Action]) -> Tuple[List[str], List[float]]:
    names = [
        "bias",
        "depth",
        "remaining_value_count",
        "value_mean",
        "value_min",
        "value_max",
        "value_std",
        "distance_closest_to_24",
        "branching_factor",
        "last_operator_add",
        "last_operator_sub",
        "last_operator_mul",
        "last_operator_div",
        "has_value_24",
        "one_step_can_make_24",
        "integer_value_fraction",
        "pair_result_min_distance_to_24",
    ]
    mean = sum(values) / max(1, len(values))
    variance = sum((v - mean) ** 2 for v in values) / max(1, len(values))
    op = action.operator if action is not None else ""
    pair_distances = []
    one_step = 0.0
    for candidate in _all_actions(values):
        result = _apply_operator(candidate.left, candidate.right, candidate.operator)
        distance = abs(result - TARGET)
        pair_distances.append(distance)
        if distance <= EPS:
            one_step = 1.0
    features = [
        1.0,
        float(4 - len(values)),
        float(len(values)),
        mean / 24.0,
        min(values or (0.0,)) / 24.0,
        max(values or (0.0,)) / 24.0,
        variance ** 0.5 / 24.0,
        min(abs(v - TARGET) for v in values) / 24.0,
        float(len(_all_actions(values))) / 36.0,
        1.0 if op == "+" else 0.0,
        1.0 if op == "-" else 0.0,
        1.0 if op == "*" else 0.0,
        1.0 if op == "/" else 0.0,
        1.0 if any(abs(v - TARGET) <= EPS for v in values) else 0.0,
        one_step,
        sum(1 for v in values if abs(v - round(v)) <= EPS) / max(1, len(values)),
        min(pair_distances or [abs(values[0] - TARGET) if values else TARGET]) / TARGET,
    ]
    return names, features


def _all_actions(values: Tuple[float, ...]) -> List[Game24Action]:
    actions: List[Game24Action] = []
    seen = set()
    for i, j in combinations(range(len(values)), 2):
        a, b = values[i], values[j]
        raw_actions = [
            Game24Action(a, b, "+"),
            Game24Action(a, b, "*"),
            Game24Action(a, b, "-"),
            Game24Action(b, a, "-"),
        ]
        if abs(b) > EPS:
            raw_actions.append(Game24Action(a, b, "/"))
        if abs(a) > EPS:
            raw_actions.append(Game24Action(b, a, "/"))
        for action in raw_actions:
            key = (round(action.left, 8), round(action.right, 8), action.operator)
            if key not in seen:
                seen.add(key)
                actions.append(action)
    actions.sort(key=lambda action: (abs(_apply_operator(action.left, action.right, action.operator) - TARGET), action.operator, action.left, action.right))
    return actions


def _canonical_values(values: Tuple[float, ...]) -> Tuple[float, ...]:
    return tuple(sorted(round(float(v), 8) for v in values))


def _find_value_index(values: List[float], target: float) -> int:
    for idx, value in enumerate(values):
        if abs(value - float(target)) <= EPS:
            return idx
    return -1


def _apply_operator(left: float, right: float, operator: str) -> float:
    if operator == "+":
        return left + right
    if operator == "-":
        return left - right
    if operator == "*":
        return left * right
    if operator == "/":
        if abs(right) <= EPS:
            raise ZeroDivisionError("divide by zero")
        return left / right
    raise ValueError(f"unknown operator: {operator}")


def _format_number(value: float) -> str:
    return str(int(value)) if abs(float(value) - int(value)) <= EPS else f"{value:.4f}"


def _next_values(values: Tuple[float, ...]) -> List[Tuple[float, ...]]:
    out = []
    for i, j in combinations(range(len(values)), 2):
        remaining = [values[idx] for idx in range(len(values)) if idx not in (i, j)]
        a, b = values[i], values[j]
        results = [a + b, a * b, a - b, b - a]
        if abs(b) > EPS:
            results.append(a / b)
        if abs(a) > EPS:
            results.append(b / a)
        for result in results:
            out.append(_canonical_values(tuple(remaining + [result])))
    return out


@lru_cache(maxsize=None)
def _min_future_expansions(values: Tuple[float, ...]) -> Optional[int]:
    values = _canonical_values(values)
    if len(values) == 1:
        return 0 if abs(values[0] - TARGET) <= EPS else None
    best: Optional[int] = None
    for nxt in _next_values(values):
        child = _min_future_expansions(nxt)
        if child is None:
            continue
        candidate = 1 + child
        if best is None or candidate < best:
            best = candidate
    return best
