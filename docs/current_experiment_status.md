# Current Experiment Promotion Status

This document records the current promotion state after running the local
paper-level audit. It is intentionally conservative: a method can be
method-ready while its current run artifact is still blocked from being used in
the paper table.

Audit command:

```bash
TEST_COUNT=100 zsh scripts/run_paper_audits.sh all
```

The audit does not call the LLM API. It checks source evidence, pinned source
references, table readiness, coverage, and certification authority metadata.

## Q2 Real-LLM Scheduling

Machine audit:
`runs/q2_table_current_seed0_protocol_v2_audit/table5_q2_readiness.csv`.

Current status:

| Row | Implementation status | Current promotion status |
| --- | --- | --- |
| CP-SAT | Implemented structured-input reference. | Blocked: current artifact lacks embedded `source_evidence` for CP-SAT; rerun structured Q2 with current code. |
| EST / list scheduling heuristic | Implemented structured-input reference. | Ready in current audit. |
| Direct / CoT | Implemented native complete-schedule sampler. | Pending: no complete current native artifact in default directory. |
| Self-Consistency | Implemented native multi-sample aggregation. | Pending: no complete current native artifact in default directory. |
| Best-of-N | Implemented verifier-selected Best-of-N. | Pending: no complete current native artifact in default directory. |
| Tree of Thoughts | Implemented strict ToT BFS adaptation. | Pending: no complete current `run-scheduling-tot-q2` artifact in default directory. |
| Target-PRM priority | Implemented in-domain process/frontier controller. | Pending: no complete current learned-search artifact in default directory. |
| PRM-guided MCTS (ReST-MCTS-style) | Implemented search-component adaptation with MCTS primitives. | Pending: no complete current learned-search artifact in default directory. |
| Solver-trained priority | Implemented zero-shot solver-to-LLM controller. | Pending: no complete current learned-search artifact in default directory. |
| Solver-trained priority + certified pruning | Implemented solver prior plus verified pruning. | Pending: no complete current learned-search artifact in default directory. |

The Q2 implementation and source-alignment matrix is current at
`runs/q2_implementation_matrix/q2_implementation_matrix.csv`. It records the
official-source primitive for each external baseline and the local code path
that implements the Scheduling adaptation.

## Q4 tau/tau2 Tool-Agent Evaluation

Machine audit:
`runs/q4_tau_table_current_audit/q4_tau_readiness.csv`.

Current status:

All Q4 tau/tau2 rows remain pending because the repository still has no native
tau/tau2 runner artifact in the default directories. This is expected and
correct. The local synthetic Level-III tool environment must not be reported as
tau-bench.

Rows still requiring native artifacts:

- Native agent rerun
- Best-of-N
- Value-guided search
- Target-PRM
- Solver-derived controller

The source evidence is now pinned:

- tau-bench: `59a200c6d575d595120f1cb70fea53cef0632f6b`
- tau2-bench / current tau successor repo:
  `a2c024725189473d2d7cea3a5cfdbcc67478e41f`

The original tau-bench repo currently warns that its airline/retail tasks are
outdated and points users to the tau2/tau3 successor repository. Any future Q4
result must therefore explicitly declare whether it targets legacy tau-bench or
the current tau2/tau3 protocol.

## Certification Tables

Machine audit:
`runs/certification_tables_readiness/certification_tables_readiness.csv`.

Current status:

| Table | Current promotion status |
| --- | --- |
| Table 6 clean certification | Blocked for current default directory: `table6_clean_certification.csv` and `certification_protocol` are missing. Rerun `run-scheduling-certification` with current code. |
| Table 7 grounding robustness | Blocked for current default directory: `table6_authority_metrics.json`, `table7_grounding_robustness.csv`, and per-instance coverage are missing. Rerun `run-table6-authority` with current code. |

The required certification authority contract is now machine-audited:

- Verified lower-bound or feasibility logic is the only pruning authority.
- Learned scores may rank frontier states only.
- Uncertain constraints may not authorize irreversible pruning.
- Oracle labels are not inference features.
- Pruned/rejected states are audited against the uncorrupted descendant
  subproblem.

## Immediate Rerun Order

Use this order to replace stale or missing artifacts:

1. Rerun structured Q2 so CP-SAT carries current `source_evidence`.
2. Rerun learned-search Q2 with Target-PRM, ReST-MCTS-style, Solver prior, and
   Solver prior + certification in the same current artifact.
3. Rerun native Q2 Direct / CoT, Self-Consistency, and Best-of-N.
4. Rerun strict native ToT BFS with `run-scheduling-tot-q2`.
5. Rerun clean certification and grounding robustness with current code.
6. Keep Q4 tau/tau2 rows pending until a native tau/tau2 adapter exists.

After any rerun, run:

```bash
PYTHON_BIN=.venv/bin/python scripts/run_paper_audits.sh all
```

Only rows marked ready by the relevant readiness CSV should be promoted into
paper tables.
