# Current Experiment Promotion Status

This file is generated from machine-readable readiness artifacts. Do not
promote a paper-table row unless the relevant readiness row is ready.

Secret safety audit: `runs/secret_safety_audit/secret_safety_audit.csv`.

Audit command:

```bash
PYTHON_BIN=.venv/bin/python scripts/run_paper_audits.sh all
```

## Q2 Real-LLM Scheduling

Machine audit: `runs/q2_table_current_seed0_protocol_v2_audit/table5_q2_readiness.csv`.

| Row | Source artifact | Implementation status | Current promotion status | Compute status |
| --- | --- | --- | --- | --- |
| CP-SAT | runs/q2_structured_100_seed0_runtime_v2/scheduling_structured_q2_table.csv (structured) | Implemented. structured SchedulingTask input; integer start/end variables; optional machine intervals with NoOverlap; release time and precedence constraints; local makespan plus total tardiness objective | Ready in current audit. | Ready. |
| EST / list scheduling heuristic | runs/q2_structured_100_seed0_runtime_v2/scheduling_structured_q2_table.csv (structured) | Implemented. structured SchedulingTask input; ready-job dispatch by earliest feasible start/finish; tardiness and due-date tie breaks | Ready in current audit. | Ready. |
| Direct / CoT | runs/q2_native_llm_baselines_100_seed0 | Implemented. one native complete-schedule LLM sample; step-by-step reasoning requested inside JSON; typed action parsing and verifier validation | Blocked: pending; artifact missing | Blocked: missing token accounting; result not complete |
| Self-Consistency | runs/q2_native_llm_baselines_100_seed0 | Implemented. multiple complete-schedule samples; aggregation by canonical final job order; tie-breaking by earliest valid sample rather than verified objective | Blocked: pending; artifact missing | Blocked: missing token accounting; result not complete |
| Best-of-N | runs/q2_native_llm_baselines_100_seed0 | Implemented. multiple complete-schedule samples; verifier filters invalid schedules; lowest verified objective selects the final sample | Blocked: pending; artifact missing | Blocked: missing token accounting; result not complete |
| Tree of Thoughts | runs/scheduling_missing_gpt_frontier_rank_100_seed0 | Implemented. level/frontier BFS; LLM proposal of candidate states; LLM value/ranking of generated frontier states; averaged value samples; greedy top-b selection | Blocked: pending; artifact missing | Blocked: missing token accounting; result not complete |
| Target-PRM priority | runs/scheduling_table5_prm_rest_cert_100_seed0/scheduling_llm_pilot_table.csv (Target-PRM) | Implemented. target LLM-induced training trajectories on disjoint instances; process/frontier ranking supervision; canonical state features matched to solver-trained controller | Ready in current audit. | Blocked: shared-cache fallback accounting |
| PRM-guided MCTS (ReST-MCTS-style) | runs/scheduling_table5_prm_rest_cert_100_seed0/scheduling_llm_pilot_table.csv (PRM-guided MCTS (ReST-MCTS-style)) | Implemented. target PRM priors; persistent visit and value statistics; PUCT/UCB-style exploration; leaf value estimates; selected-node expansion observation; backup along the selected path | Blocked: invalid artifacts: ReST-MCTS-style table row must record positive MCTS selection calls, ReST-MCTS-style table row must record positive MCTS expansion observations, ReST-MCTS-style table row must record positive PRM prior evaluations, ReST-MCTS-style table row must record positive MCTS simulation/evaluation calls, ReST-MCTS-style table row must record positive MCTS backpropagation updates, ReST-MCTS-style diagnostics must record the PRM-guided PUCT/UCB selection score | Blocked: shared-cache fallback accounting |
| Solver-trained priority | runs/scheduling_table5_prm_rest_cert_100_seed0/scheduling_llm_pilot_table.csv (Solver-trained MLP) | Implemented. source solver-induced training trajectories only; frozen controller during real LLM-induced Scheduling search; canonical state features without oracle label leakage | Ready in current audit. | Blocked: shared-cache fallback accounting |
| Solver-trained priority + certified pruning | runs/scheduling_table5_prm_rest_cert_100_seed0/scheduling_llm_pilot_table.csv (Solver-trained MLP + certified pruning) | Implemented. same solver-trained priority controller; verified pruning authority separated from learned ranking; on-demand proposal accounting for avoided calls | Ready in current audit. | Blocked: shared-cache fallback accounting |

Implementation/source matrix: `runs/q2_implementation_matrix/q2_implementation_matrix.csv`.
Official implementation plan: `docs/q2_table4_official_implementation_plan.md`.
Local primitive audit: `runs/q2_local_primitive_audit/q2_local_primitive_audit.csv`.
Remaining Q2 run queue: `runs/q2_remaining_runs/q2_remaining_runs.md`.
Q2 run preflight: `runs/q2_run_preflight/q2_run_preflight.md`.

## Q4 tau/tau2 Tool-Agent Evaluation

Machine audit: `runs/q4_tau_table_current_audit/q4_tau_readiness.csv`.

Native adapter contract: `runs/q4_tau_adapter_contract/q4_tau_adapter_contract.md`.

All Q4 tau/tau2 rows remain pending; the repository still has no native tau/tau2 runner artifact in the audited directories.

| Row | Source artifact | Current promotion status | Notes |
| --- | --- | --- | --- |
| Native agent rerun | runs/q4_tau_native | Blocked: pending; q4_tau_manifest.json missing; q4_tau_per_instance.csv partial task coverage: 0/100; q4_tau_row.json or q4_tau_table.csv missing aggregate row | native tau/tau2 artifact missing; external adapter is currently an explicit stub |
| Best-of-N | runs/q4_tau_best_of_n | Blocked: pending; q4_tau_manifest.json missing; q4_tau_per_instance.csv partial task coverage: 0/100; q4_tau_row.json or q4_tau_table.csv missing aggregate row | native tau/tau2 artifact missing; external adapter is currently an explicit stub |
| Value-guided search | runs/q4_tau_value_guided | Blocked: pending; q4_tau_manifest.json missing; q4_tau_per_instance.csv partial task coverage: 0/100; q4_tau_row.json or q4_tau_table.csv missing aggregate row | native tau/tau2 artifact missing; external adapter is currently an explicit stub |
| Target-PRM | runs/q4_tau_target_prm | Blocked: pending; q4_tau_manifest.json missing; q4_tau_per_instance.csv partial task coverage: 0/100; q4_tau_row.json or q4_tau_table.csv missing aggregate row | native tau/tau2 artifact missing; external adapter is currently an explicit stub |
| Solver-derived controller | runs/q4_tau_solver_controller | Blocked: pending; q4_tau_manifest.json missing; q4_tau_per_instance.csv partial task coverage: 0/100; q4_tau_row.json or q4_tau_table.csv missing aggregate row | native tau/tau2 artifact missing; external adapter is currently an explicit stub |

Pinned Q4 source evidence:

- tau-bench: `59a200c6d575d595120f1cb70fea53cef0632f6b`
- tau2-bench: `a2c024725189473d2d7cea3a5cfdbcc67478e41f`

## Certification Tables

Machine audit: `runs/certification_tables_readiness/certification_tables_readiness.csv`.

| Table | Source artifact | Current promotion status |
| --- | --- | --- |
| Table 6 clean certification | runs/q2_structured_cert_reference_100_seed0_budget200_protocol_v2 | Ready in current audit. |
| Table 7 grounding robustness | runs/table7_grounding_robustness_100_seed0_budget200_protocol_v2 | Ready in current audit. |

Required certification authority contract:

Authority source audit: `runs/certification_authority_source_audit/certification_authority_source_audit.csv` (3/3 authority checks ready).

- Verified lower-bound or feasibility logic is the only pruning authority.
- Learned scores may rank frontier states only.
- Uncertain constraints may not authorize irreversible pruning.
- Oracle labels are not inference features.
- Pruned/rejected states are audited against the uncorrupted descendant subproblem.

## Immediate Rerun Order

1. Rerun structured Q2 if CP-SAT is blocked by stale source evidence.
2. Rerun learned-search Q2 with Target-PRM, ReST-MCTS-style, Solver prior, and Solver prior + certification in the same current artifact.
3. Rerun native Q2 Direct / CoT, Self-Consistency, and Best-of-N.
4. Rerun strict native ToT BFS with `run-scheduling-tot-q2`.
5. Rerun clean certification and grounding robustness with current code if their readiness rows are blocked.
6. Keep Q4 tau/tau2 rows pending until a native tau/tau2 adapter exists.

Only rows marked ready by the relevant readiness CSV should be promoted into paper tables.
