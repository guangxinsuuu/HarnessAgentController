# Q4 tau-bench Source Alignment

This document records the current status of the proposed stateful tool-agent
evaluation table. It is intentionally conservative: the repository currently
has a local synthetic Level-III tool environment and an explicit external
benchmark adapter stub, but it does not yet implement a native tau-bench runner.

## Source Evidence

- Original tau-bench: `https://github.com/sierra-research/tau-bench`
  - Source revision pinned for Q4 evidence:
    `59a200c6d575d595120f1cb70fea53cef0632f6b`.
- Current recommended successor: `https://github.com/sierra-research/tau2-bench`
  - Source revision pinned for Q4 evidence:
    `a2c024725189473d2d7cea3a5cfdbcc67478e41f`.

The original tau-bench repository currently points users to newer τ-bench task
versions. A future Q4 implementation should therefore state whether it targets
the original tau-bench protocol or the current tau2/tau3 successor protocol.
The machine-readable checklist lives in
`cgr_exp/q4_tau_source_evidence.py`, and the source-ref verifier checks full
GitHub SHAs, pinned raw-source URLs, and symbol/anchor references. Its default
mode is offline and deterministic; `--online` additionally verifies that the
pinned revision and raw-source URLs are reachable, and records
`raw_anchor_locations` for matched raw-source anchors so the official benchmark
line evidence can be inspected directly.
The native-adapter acceptance contract can be generated with
`build-q4-tau-adapter-contract`; it records the required official protocol
components, row-specific method primitives, and artifact schema that any future
tau/tau2 runner must satisfy.

## Required Rows

| Row | Required native primitive | Current status |
| --- | --- | --- |
| Native agent rerun | Run the benchmark's native agent protocol on held-out tasks, preserving tool state, user simulator, task verifier, and official metrics. | Pending; no native runner implemented. |
| Best-of-N | Reuse the native protocol with multiple independent trajectories and select using the benchmark-compatible evaluator/verifier. | Pending; cannot be inferred from Q2 Scheduling Best-of-N. |
| Value-guided search | Online search over tool-agent states using a value/evaluator signal while preserving the native tool environment. | Pending; compatibility must be checked against the official environment. |
| Target-PRM | Train on target tau-bench/tau2 tool-agent trajectories from disjoint training tasks and evaluate on held-out tasks. | Pending. |
| Solver-derived controller | Transfer a source-trained controller to native tool-agent trajectories without target training. | Pending. |

## Non-Claims

- The local synthetic `level3_tool_env` is useful for CI and development, but it
  is not tau-bench.
- `ExternalToolBenchmarkAdapter` is an explicit stub. Its manifest sets
  `result_ready = false`.
- No tau-bench row should be reported until a native adapter can load tasks,
  execute tool calls, preserve tool state, and emit official pass/tool-call/token
  metrics.

## Table Assembly

The repository can now build and audit a conservative Q4 table shell, but it
does not generate native tau-bench trajectories by itself.

```bash
python3 -m cgr_exp.cli audit-q4-tau-source-evidence \
  --output-dir runs/q4_tau_source_evidence_audit \
  --overwrite

python3 -m cgr_exp.cli verify-q4-tau-source-refs \
  --output-dir runs/q4_tau_source_ref_verification \
  --overwrite

python3 -m cgr_exp.cli build-q4-tau-adapter-contract \
  --output-dir runs/q4_tau_adapter_contract \
  --expected-test-count 100 \
  --overwrite

python3 -m cgr_exp.cli build-table-q4-tau \
  --output-dir runs/q4_tau_table_current \
  --expected-test-count 100 \
  --native-dir runs/q4_tau_native \
  --best-of-n-dir runs/q4_tau_best_of_n \
  --value-guided-dir runs/q4_tau_value_guided \
  --target-prm-dir runs/q4_tau_target_prm \
  --solver-controller-dir runs/q4_tau_solver_controller \
  --overwrite

python3 -m cgr_exp.cli audit-table-q4-tau \
  --output-dir runs/q4_tau_table_current_audit \
  --expected-test-count 100 \
  --native-dir runs/q4_tau_native \
  --best-of-n-dir runs/q4_tau_best_of_n \
  --value-guided-dir runs/q4_tau_value_guided \
  --target-prm-dir runs/q4_tau_target_prm \
  --solver-controller-dir runs/q4_tau_solver_controller \
  --overwrite
```

Each row directory must contain native tau/tau2 evidence before the row is
marked ready:

- `q4_tau_manifest.json`: declares `benchmark` as `tau-bench`, `tau2-bench`, or
  `tau3-bench`; sets `native_protocol_preserved = true`; records the official
  source URL and pinned `source_revision`; embeds the matching `source_evidence`
  block from `cgr_exp/q4_tau_source_evidence.py`; and lists the covered methods.
- `q4_tau_manifest.json` must also include a `native_protocol` object proving
  that the run preserved the official stateful tool environment, user
  simulator, task verifier, pass-k aggregation, tool-call accounting, and token
  accounting. The audit requires `uses_official_user_simulator = true`,
  `uses_official_task_verifier = true`, and `preserves_tool_state = true`.
- `q4_tau_manifest.json` must include `method_protocols[method]`:
  - `Native agent rerun`: `single_native_agent_rollout`.
  - `Best-of-N`: `multiple_independent_native_rollouts` and
    `benchmark_compatible_selection`.
  - `Value-guided search`: `online_value_guidance` and
    `native_tool_state_preserved`.
  - `Target-PRM`: target tool-agent trajectory training, process/step
    supervision, and disjoint train/test tasks; it must not claim zero-shot
    transfer.
  - `Solver-derived controller`: source-solver training, zero-shot target-tool
    transfer, and native tool-state preservation.
- `q4_tau_per_instance.csv`: one held-out task row per evaluated instance, with
  `task_id` or `instance_id` coverage matching `--expected-test-count`.
  If the aggregate row reports `pass^4`, each task must also provide pass-k
  evidence either through at least four rollout/sample rows or through an
  explicit per-task `rollout_count`, `sample_count`, `trajectory_count`, or
  `attempt_count` field of at least 4.
- `q4_tau_row.json` or `q4_tau_table.csv`: aggregate `pass^1`, `pass^4`,
  `violations`, `tool_calls`, `tokens`, `wall_clock_seconds`, and `status`.

Rows without this evidence remain pending by construction.
