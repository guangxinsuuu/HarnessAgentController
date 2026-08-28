from __future__ import annotations

from typing import Dict, List


Q2_METHOD_CONTRACTS: Dict[str, Dict[str, object]] = {
    "CP-SAT": {
        "reference": "google/or-tools CP-SAT scheduling model family",
        "source_url": "https://github.com/google/or-tools",
        "claim_scope": "structured_reference",
        "official_reproduction": False,
        "required_table_wording": "CP-SAT structured-input calibration reference",
        "implemented_scope": [
            "structured SchedulingTask input",
            "integer start/end variables",
            "optional machine intervals with NoOverlap",
            "release time and precedence constraints",
            "local makespan plus total tardiness objective",
        ],
        "non_claims": ["not an LLM-agent baseline", "not exposed to rendered natural language"],
    },
    "EST / list scheduling heuristic": {
        "reference": "standard earliest-start list scheduling heuristic",
        "source_url": "",
        "claim_scope": "structured_reference",
        "official_reproduction": False,
        "required_table_wording": "EST / list scheduling structured-input calibration reference",
        "implemented_scope": [
            "structured SchedulingTask input",
            "ready-job dispatch by earliest feasible start/finish",
            "tardiness and due-date tie breaks",
        ],
        "non_claims": ["not a learned method", "not a direct competitor to semantic LLM agents"],
    },
    "Direct / CoT": {
        "reference": "Chain-of-Thought prompting",
        "source_url": "https://arxiv.org/abs/2201.11903",
        "claim_scope": "native_adaptation",
        "official_reproduction": False,
        "required_table_wording": "Direct / CoT frozen native Scheduling adaptation",
        "implemented_scope": [
            "one native complete-schedule LLM sample",
            "step-by-step reasoning requested inside JSON",
            "typed action parsing and verifier validation",
        ],
        "non_claims": ["no frontier search", "no target trajectory training"],
    },
    "Self-Consistency": {
        "reference": "self-consistency decoding for chain-of-thought reasoning",
        "source_url": "https://arxiv.org/abs/2203.11171",
        "claim_scope": "native_adaptation",
        "official_reproduction": False,
        "required_table_wording": "Self-Consistency frozen native Scheduling adaptation",
        "implemented_scope": [
            "multiple complete-schedule samples",
            "aggregation by canonical final job order",
            "verified objective used only as a tie-break within selected answers",
        ],
        "non_claims": ["not frontier search", "not Best-of-N objective selection"],
    },
    "Best-of-N": {
        "reference": "test-time sampling with verifier/reward selection",
        "source_url": "",
        "claim_scope": "native_adaptation",
        "official_reproduction": False,
        "required_table_wording": "Best-of-N + verifier",
        "implemented_scope": [
            "multiple complete-schedule samples",
            "verifier filters invalid schedules",
            "lowest verified objective selects the final sample",
        ],
        "non_claims": ["strong verifier-selected variant; table notes must disclose this"],
    },
    "Tree of Thoughts": {
        "reference": "princeton-nlp/tree-of-thought-llm BFS",
        "source_url": "https://github.com/princeton-nlp/tree-of-thought-llm",
        "claim_scope": "native_adaptation",
        "official_reproduction": False,
        "required_table_wording": "Tree of Thoughts native BFS Scheduling adaptation",
        "implemented_scope": [
            "level/frontier BFS",
            "LLM proposal of candidate states",
            "LLM value/ranking of generated frontier states",
            "averaged value samples",
            "greedy top-b selection",
        ],
        "non_claims": ["Scheduling task adaptation; not the original Game24 prompt"],
    },
    "Target-PRM priority": {
        "reference": "process reward modeling / PRM800K process-supervision setup",
        "source_url": "https://github.com/openai/prm800k",
        "claim_scope": "domain_adapted_baseline",
        "official_reproduction": False,
        "required_table_wording": "Target-PRM priority, in-domain target LLM states",
        "implemented_scope": [
            "target LLM-induced training trajectories on disjoint instances",
            "process/frontier ranking supervision",
            "canonical state features matched to solver-trained controller",
        ],
        "non_claims": ["not zero-shot", "not outcome-only PRM"],
    },
    "Outcome-PRM": {
        "reference": "outcome-supervised reward model diagnostic",
        "source_url": "",
        "claim_scope": "diagnostic_ablation",
        "official_reproduction": False,
        "required_table_wording": "Outcome-PRM diagnostic only",
        "implemented_scope": [
            "target LLM-induced states",
            "terminal outcome supervision propagated to visited states",
        ],
        "non_claims": ["not a PRM800K-style process reward model"],
    },
    "PRM-guided MCTS (ReST-MCTS-style)": {
        "reference": "THUDM/ReST-MCTS MCTS search component",
        "source_url": "https://github.com/THUDM/ReST-MCTS",
        "claim_scope": "style_component_adaptation",
        "official_reproduction": False,
        "required_table_wording": "PRM-guided MCTS (ReST-MCTS-style search component)",
        "implemented_scope": [
            "target PRM priors",
            "persistent visit and value statistics",
            "PUCT/UCB-style exploration",
            "leaf value estimates",
            "selected-node expansion observation",
            "backup along the selected path",
        ],
        "non_claims": ["not full ReST-MCTS iterative self-training"],
    },
    "Solver-trained priority": {
        "reference": "CGR solver-to-LLM zero-shot controller",
        "source_url": "",
        "claim_scope": "ours",
        "official_reproduction": True,
        "required_table_wording": "Solver prior, solver-to-LLM zero-shot transfer",
        "implemented_scope": [
            "source solver-induced training trajectories only",
            "frozen controller during real LLM-induced Scheduling search",
            "canonical state features without oracle label leakage",
        ],
        "non_claims": ["no target LLM trajectory supervision"],
    },
    "Solver-trained priority + certified pruning": {
        "reference": "CGR solver-to-LLM controller plus verified pruning",
        "source_url": "",
        "claim_scope": "ours",
        "official_reproduction": True,
        "required_table_wording": "Solver prior + certification, solver-to-LLM zero-shot transfer",
        "implemented_scope": [
            "same solver-trained priority controller",
            "verified pruning authority separated from learned ranking",
            "on-demand proposal accounting for avoided calls",
        ],
        "non_claims": ["learned scores do not authorize irreversible pruning"],
    },
}


def q2_method_contract(name: str) -> Dict[str, object]:
    return dict(Q2_METHOD_CONTRACTS[name])


def q2_method_contracts(names: List[str]) -> Dict[str, Dict[str, object]]:
    return {name: q2_method_contract(name) for name in names}
