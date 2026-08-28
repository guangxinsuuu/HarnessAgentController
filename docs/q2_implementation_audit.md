# Q2 Implementation Audit

This audit maps each row in the Scheduling Q2 quality-compute table to the
current implementation evidence. A row is table-ready only when both the method
logic and the completed run artifact match the claimed scope.

The normative row checklist is `docs/q2_table4_required_algorithms.md`; this
audit records the current implementation and artifact evidence against that
checklist.

## Current Evidence Summary

| Row | Implementation evidence | Test evidence | Run evidence | Claim status |
| --- | --- | --- | --- | --- |
| CP-SAT | `cgr_exp/scheduling_structured_q2.py::_solve_with_cpsat` uses OR-Tools optional intervals, `NoOverlap`, release times, precedences, and `makespan + tardiness` objective. | `tests/unit/test_scheduling_structured_q2.py` covers row generation. | `.venv` run with OR-Tools 9.15.6755 wrote `runs/q2_structured_100_seed0/scheduling_structured_q2_table.csv`: success 1.0000, gap 0.00000, tokens 0. | Result-ready as a structured-input calibration reference. |
| EST / list scheduling heuristic | `cgr_exp/scheduling_structured_q2.py::_earliest_start_list_schedule` schedules ready jobs by earliest start/finish/tardiness/due-date tie breaks. | `tests/unit/test_scheduling_structured_q2.py`. | `runs/q2_structured_100_seed0/scheduling_structured_q2_table.csv`: success 0.4200, gap 0.16829, tokens 0. | Result-ready as a structured-input heuristic reference. |
| Direct / CoT | `cgr_exp/scheduling_native_q2.py` uses one complete-schedule LLM sample and verifier validation; raw outputs, parse failures, valid sample rate, and token usage are audited. | `tests/unit/test_scheduling_native_q2.py`. | No complete 100-instance native run found locally. | Method-ready; result pending. |
| Self-Consistency | `cgr_exp/scheduling_native_q2.py` samples N complete schedules and selects the majority canonical job order with objective tie-break; raw outputs, parse failures, valid sample rate, and token usage are audited. | `tests/unit/test_scheduling_native_q2.py`. | No complete 100-instance native run found locally. | Method-ready as an optimization-compatible SC adaptation; result pending. |
| Best-of-N | `cgr_exp/scheduling_native_q2.py` samples N complete schedules, filters with the verifier, and selects minimum objective; raw outputs, parse failures, valid sample rate, and token usage are audited. | `tests/unit/test_scheduling_native_q2.py`. | No complete 100-instance native run found locally. | Method-ready as verifier-selected Best-of-N; result pending. |
| Tree of Thoughts | `cgr_exp/scheduling_tot_q2.py` implements BFS propose/evaluate/select with configurable breadth, averaged value samples, and a JSONL trace of generate/evaluate/select events. | `tests/unit/test_scheduling_tot_q2.py`; CLI mock smoke. | No complete real-LLM 100-instance ToT BFS run found locally. Older `GPT-frontier-rank / ToT-style` artifacts are diagnostics only. | Method-ready as native ToT BFS adaptation; result pending. |
| Target-PRM | `cgr_exp/scheduling_transfer.py` trains on target LLM-induced trajectories from disjoint target training instances and evaluates on held-out target instances. | `tests/unit/test_scheduling_transfer.py` checks target PRM artifacts. | `runs/scheduling_table5_prm_rest_cert_100_seed0` contains a complete 100-instance shared-replay row. | Controller-isolation result-ready; native Q2 wording must state frozen proposer/cache if reused. |
| PRM-guided MCTS (ReST-MCTS-style) | `cgr_exp/scheduling_transfer.py::PRMGuidedMCTSStylePolicy` has visit counts, value sums, PUCT-style exploration, PRM priors, expansion observation, and backup along the selected path. | `tests/unit/test_scheduling_transfer.py::test_prm_guided_mcts_policy_tracks_visits_and_backups` and `test_prm_guided_mcts_policy_backs_up_observed_expansion`. | `runs/scheduling_table5_prm_rest_cert_100_seed0` predates the MCTS upgrade; rerun needed for updated result. | Method-ready as ReST-MCTS-style search component, not full ReST-MCTS* self-training; result must be rerun. |
| Solver prior | `cgr_exp/scheduling_transfer.py` trains only on solver-induced source traces and freezes before target evaluation. | `tests/unit/test_scheduling_transfer.py`; leakage manifests. | `runs/scheduling_table5_prm_rest_cert_100_seed0` complete 100-instance row exists. | Result-ready for shared-replay zero-shot transfer. |
| Solver prior + certification | `cgr_exp/scheduling_transfer.py` adds certified pruning only for this policy; `table6_authority.py` audits authority boundaries; per-policy on-demand proposal token accounting is now emitted alongside shared-cache tokens. | `tests/unit/test_scheduling_transfer.py`, `tests/unit/test_table6_authority.py`. | `runs/scheduling_table5_prm_rest_cert_100_seed0` complete 100-instance row exists, but it predates the on-demand accounting field. | Shared-replay result-ready; rerun needed for Q2 on-demand token numbers. |

## Token Accounting Rule

`build-table4-q2` prefers `on_demand_total_tokens` when a Scheduling LLM pilot
artifact contains per-policy proposal accounting. If the artifact predates that
field, the builder falls back to shared-cache `total_tokens` and writes
`tokens use shared-cache fallback accounting` in the row notes. Rows with this
fallback are acceptable for controller-isolation comparisons but should not be
used as final Q2 end-to-end compute claims.

Use `audit-table4-q2` with the same input directories as `build-table4-q2` to
write `table5_q2_readiness.csv` and `table5_q2_readiness.json`. The readiness
audit marks each row's `result_ready` and `compute_ready` status and lists
blockers such as missing artifacts, partial checkpoints, missing token
accounting, or shared-cache fallback accounting.
For native LLM rows, the audit additionally checks that the sample or ToT trace
artifact covers the expected number of distinct held-out test cases. This
prevents a manually written complete summary row from promoting a partial run.

The Q2 table builder treats `Tree of Thoughts` as a native protocol row and
only promotes artifacts produced by `run-scheduling-tot-q2`. Older
`GPT-frontier-rank / ToT-style` rows from `run-scheduling-llm-pilot` are
controller-isolation diagnostics and remain pending for the strict ToT table
row.

## Immediate Run Requirements

To make the Q2 table result-complete without ToT:

1. Run `run-scheduling-native-q2` for Direct / Self-Consistency / Best-of-N.
2. Rerun `run-scheduling-llm-pilot` with `--include-target-prm --include-rest-mcts --include-solver-certification` so the upgraded MCTS row and on-demand token accounting replace the older result.
3. Build the table with `build-table4-q2 --structured-dir runs/q2_structured_100_seed0 ...`.

To add the strict ToT row, run `run-scheduling-tot-q2`; older
`GPT-frontier-rank / ToT-style` runs should remain diagnostic.

## Non-Claims

- CP-SAT must not be reported without OR-Tools or an equivalent CP-SAT backend.
- The local ReST-MCTS row is not full ReST-MCTS* iterative self-training.
- Outcome-PRM is not Target-PRM.
- Shared-replay learned-search rows are not identical to native independent LLM
  sampling rows; captions must state the frozen proposer/cache protocol when
  reused in Q2.
