# Q2 Table 4 Official-Source Implementation Plan

This file is the human-facing checklist for the Q2 end-to-end Scheduling table.
It must agree with the machine-readable contracts in
`cgr_exp/q2_method_contracts.py`, `cgr_exp/q2_source_evidence.py`, and
`cgr_exp/q2_implementation_matrix.py`.

The external-source references are statically checked with:

```bash
python3 -m cgr_exp.cli verify-q2-source-refs \
  --output-dir runs/q2_source_ref_verification \
  --overwrite
```

Current static verifier result: pinned GitHub SHAs, pinned raw GitHub refs, and
versioned arXiv references are present and internally consistent. Add
`--online --timeout 20` on a machine with network access to verify remote
reachability and raw-anchor line locations.

## Required Rows

| Table row | Original source | Required original-method primitive | Local code that implements the Scheduling adaptation | Promotion boundary |
| --- | --- | --- | --- | --- |
| CP-SAT | `https://github.com/google/or-tools` at `98c165af62df62b3056c2ee0fca66b24e79097cb` | Structured CP-SAT scheduling model with integer variables, optional intervals, `NoOverlap`, and objective minimization. | `cgr_exp/scheduling_structured_q2.py` | Calibration reference only; receives exact structured RCPSP input; `LLM calls = 0`, `tokens = 0`. |
| EST / list scheduling heuristic | Standard structured scheduling heuristic | Deterministic ready-list dispatch by earliest feasible placement and fixed tie breaks. | `cgr_exp/scheduling_structured_q2.py` | Calibration reference only; not an LLM-agent baseline. |
| Direct / CoT | Chain-of-Thought paper `arXiv:2201.11903v6` | One sampled reasoning path before extracting the final answer. | `cgr_exp/scheduling_native_q2.py` | Exactly one complete-schedule sample per held-out task; no frontier search. |
| Self-Consistency | Self-Consistency paper `arXiv:2203.11171v4` | Multiple independent reasoning paths followed by final-answer aggregation. | `cgr_exp/scheduling_native_q2.py` | Aggregate by canonical final job order; do not select by verified objective. |
| Best-of-N | Test-time compute scaling / Best-of-N paper `arXiv:2408.03314v1` | Multiple complete candidate outputs selected by verifier or reward score. | `cgr_exp/scheduling_native_q2.py` | Local row is `Best-of-N + verifier` because the verifier selects the lowest verified objective. |
| Tree of Thoughts / GPT-frontier-rank | `https://github.com/princeton-nlp/tree-of-thought-llm` at `8050e67d0e3a0fddc424d7fa5801538722a4c4cc` | Official BFS loop over `method_generate`, `method_evaluate`, and `method_select`, keeping top breadth after value/vote scoring. | `cgr_exp/scheduling_tot_q2.py` | Must come from `run-scheduling-tot-q2`; controller-isolation `GPT-frontier-rank` artifacts are not enough. |
| Target-PRM | `https://github.com/openai/prm800k` at `7ecc794703b2877f63226f2477a49b34f9b25163` | Process supervision over intermediate steps, not terminal-only outcome labels. | `cgr_exp/scheduling_transfer.py` | In-domain target LLM-induced training states; not zero-shot; Outcome-PRM is diagnostic only. |
| PRM-guided MCTS (ReST-MCTS-style) | `https://github.com/THUDM/ReST-MCTS` at `2d5f488c3d6e24f99d50a9860e818383b1bb5883` | MCTS selection, expansion, simulation/evaluation, and backpropagation guided by process reward/value. | `cgr_exp/scheduling_transfer.py` | Search-component adaptation only. Keep `style` unless the full iterative self-training pipeline is implemented. |
| Solver prior | CGR method | Solver-induced source trajectories train a frozen frontier-priority controller. | `cgr_exp/scheduling_transfer.py` | Zero-shot transfer to real LLM-induced target search; no target LLM trajectory supervision. |
| Solver prior + certification | CGR method | Solver prior plus verified pruning authority. | `cgr_exp/scheduling_transfer.py`, `cgr_exp/certification/prune.py` | Learned score ranks frontier states only; irreversible pruning must be authorized by verified logic. |

## Implementation Work Still Needed Before Table Promotion

The code paths above exist, but a row is not promotion-ready until the current
readiness audit says it is ready. Run:

```bash
PYTHON_BIN=.venv/bin/python scripts/run_paper_audits.sh all
```

Then inspect:

- `runs/q2_table_current_seed0_protocol_v2_audit/table5_q2_readiness.csv`
- `runs/q2_remaining_runs/q2_remaining_runs.md`
- `runs/q2_run_preflight/q2_run_preflight.md`

The current intended Q2 completion order is:

1. Structured references: CP-SAT and EST/list scheduling.
2. Learned-search artifact containing Target-PRM, ReST-MCTS-style, Solver
   prior, and Solver prior + certification under the same held-out split.
3. Native frozen baselines: Direct / CoT, Self-Consistency, and Best-of-N.
4. Native ToT BFS with `run-scheduling-tot-q2`.

Do not fill Table 4 from older diagnostic artifacts if their manifest lacks
the current `q2_method_alignment`, `source_evidence`, split coverage, token
accounting, or required protocol fields.
