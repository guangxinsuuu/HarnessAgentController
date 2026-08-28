# Current Experiment Promotion Status

This file is generated from machine-readable readiness artifacts. Do not
promote a paper-table row unless the relevant readiness row is ready.

Audit command:

```bash
PYTHON_BIN=.venv/bin/python scripts/run_paper_audits.sh all
```

## Q2 Real-LLM Scheduling

Machine audit: `runs/q2_table_current_seed0_protocol_v2_audit/table5_q2_readiness.csv`.

| Row | Implementation status | Current promotion status |
| --- | --- | --- |
| CP-SAT | Implemented. structured SchedulingTask input; integer start/end variables; optional machine intervals with NoOverlap; release time and precedence constraints; local makespan plus total tardiness objective | Ready in current audit. |
| EST / list scheduling heuristic | Implemented. structured SchedulingTask input; ready-job dispatch by earliest feasible start/finish; tardiness and due-date tie breaks | Ready in current audit. |
| Direct / CoT | Implemented. one native complete-schedule LLM sample; step-by-step reasoning requested inside JSON; typed action parsing and verifier validation | Blocked: pending; artifact missing |
| Self-Consistency | Implemented. multiple complete-schedule samples; aggregation by canonical final job order; tie-breaking by earliest valid sample rather than verified objective | Blocked: pending; artifact missing |
| Best-of-N | Implemented. multiple complete-schedule samples; verifier filters invalid schedules; lowest verified objective selects the final sample | Blocked: pending; artifact missing |
| Tree of Thoughts | Implemented. level/frontier BFS; LLM proposal of candidate states; LLM value/ranking of generated frontier states; averaged value samples; greedy top-b selection | Blocked: pending; artifact missing |
| Target-PRM priority | Implemented. target LLM-induced training trajectories on disjoint instances; process/frontier ranking supervision; canonical state features matched to solver-trained controller | Blocked: pending; artifact missing |
| PRM-guided MCTS (ReST-MCTS-style) | Implemented. target PRM priors; persistent visit and value statistics; PUCT/UCB-style exploration; leaf value estimates; selected-node expansion observation; backup along the selected path | Blocked: pending; artifact missing |
| Solver-trained priority | Implemented. source solver-induced training trajectories only; frozen controller during real LLM-induced Scheduling search; canonical state features without oracle label leakage | Blocked: pending; artifact missing |
| Solver-trained priority + certified pruning | Implemented. same solver-trained priority controller; verified pruning authority separated from learned ranking; on-demand proposal accounting for avoided calls | Blocked: pending; artifact missing |

Implementation/source matrix: `runs/q2_implementation_matrix/q2_implementation_matrix.csv`.

## Q4 tau/tau2 Tool-Agent Evaluation

Machine audit: `runs/q4_tau_table_current_audit/q4_tau_readiness.csv`.

All Q4 tau/tau2 rows remain pending; the repository still has no native
tau/tau2 runner artifact in the audited directories.

Pinned Q4 source evidence:

- tau-bench: `59a200c6d575d595120f1cb70fea53cef0632f6b`
- tau2-bench: `a2c024725189473d2d7cea3a5cfdbcc67478e41f`

## Certification Tables

Machine audit: `runs/certification_tables_readiness/certification_tables_readiness.csv`.

| Table | Current promotion status |
| --- | --- |
| Table 6 clean certification | Blocked: table6_clean_certification.csv missing; missing certification_protocol |
| Table 7 grounding robustness | Blocked: table6_authority_metrics.json missing; table7_grounding_robustness.csv missing; table6_authority_per_instance.csv missing; table6_authority_per_instance.csv partial task coverage: 0/100 |

Required certification authority contract:

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

Only rows marked ready by the relevant readiness CSV should be promoted into
paper tables.
