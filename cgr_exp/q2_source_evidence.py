from __future__ import annotations

from typing import Dict, List


Q2_SOURCE_EVIDENCE: Dict[str, Dict[str, object]] = {
    "CP-SAT": {
        "source_url": "https://github.com/google/or-tools",
        "source_files": [
            "ortools/sat/docs/scheduling.md",
            "ortools/sat/python/cp_model.py",
        ],
        "official_primitives": [
            "integer variables for scheduling decisions",
            "interval variables",
            "NoOverlap machine-capacity constraints",
            "precedence constraints",
            "objective minimization",
        ],
        "local_promotion_gates": [
            "structured_input true",
            "llm_calls equals 0",
            "tokens equals 0",
        ],
    },
    "Direct / CoT": {
        "source_url": "https://arxiv.org/abs/2201.11903",
        "source_files": ["paper"],
        "official_primitives": [
            "single chain-of-thought reasoning path",
            "intermediate natural-language rationale",
            "final answer extraction",
        ],
        "local_promotion_gates": [
            "selection_rule first_valid",
            "exactly one sample per held-out task",
            "oracle future labels excluded",
        ],
    },
    "Self-Consistency": {
        "source_url": "https://arxiv.org/abs/2203.11171",
        "source_files": ["paper"],
        "official_primitives": [
            "multiple independent reasoning paths",
            "final-answer aggregation",
            "consistency-based selection",
        ],
        "local_promotion_gates": [
            "selection_rule majority_job_order_then_objective",
            "multiple samples per held-out task",
            "not Best-of-N objective selection",
        ],
    },
    "Best-of-N": {
        "source_url": "https://arxiv.org/abs/2408.03314",
        "source_files": ["paper"],
        "official_primitives": [
            "multiple complete answer samples",
            "verifier or reward scoring",
            "best sample selection",
        ],
        "local_promotion_gates": [
            "selection_rule best_verified_objective",
            "verifier objective selection disclosed",
            "declared sample_count coverage per held-out task",
        ],
    },
    "Tree of Thoughts": {
        "source_url": "https://github.com/princeton-nlp/tree-of-thought-llm",
        "source_files": ["src/tot/methods/bfs.py"],
        "official_primitives": [
            "method_generate",
            "method_evaluate",
            "method_select",
            "breadth-first search",
            "greedy top-b selection",
        ],
        "local_promotion_gates": [
            "native_tot_protocol declares official BFS entrypoints",
            "trace contains generate events",
            "trace contains evaluation and selection events",
            "trace contains value sample records",
            "one finish event per held-out rendering",
        ],
    },
    "Target-PRM priority": {
        "source_url": "https://github.com/openai/prm800k",
        "source_files": ["README.md", "data/*.jsonl"],
        "official_primitives": [
            "intermediate-step labels",
            "process supervision",
            "reward/value model over partial solutions",
        ],
        "local_promotion_gates": [
            "target LLM-induced training trajectories",
            "target PRM ranker artifact",
            "target PRM trace and pair artifacts",
            "no training/evaluation task overlap",
            "not zero-shot",
        ],
    },
    "PRM-guided MCTS (ReST-MCTS-style)": {
        "source_url": "https://github.com/THUDM/ReST-MCTS",
        "source_files": ["MCTS/mcts.py", "MCTS/base.py", "README.md"],
        "official_primitives": [
            "MCTS selection",
            "MCTS expansion",
            "MCTS simulation/evaluation",
            "MCTS backpropagation",
            "process reward guidance",
        ],
        "local_promotion_gates": [
            "target PRM training distribution",
            "mcts_components include selection",
            "mcts_components include expansion",
            "mcts_components include simulation/evaluation",
            "mcts_components include backpropagation",
            "full_iterative_self_training false unless fully implemented",
        ],
    },
}


def q2_source_evidence(method: str) -> Dict[str, object]:
    return dict(Q2_SOURCE_EVIDENCE[method])


def q2_source_evidence_methods() -> List[str]:
    return sorted(Q2_SOURCE_EVIDENCE)
