# Q4 tau-bench Source Alignment

This document records the current status of the proposed stateful tool-agent
evaluation table. It is intentionally conservative: the repository currently
has a local synthetic Level-III tool environment and an explicit external
benchmark adapter stub, but it does not yet implement a native tau-bench runner.

## Source Evidence

- Original tau-bench: `https://github.com/sierra-research/tau-bench`
- Current recommended successor: `https://github.com/sierra-research/tau2-bench`

The original tau-bench repository currently points users to newer τ-bench task
versions. A future Q4 implementation should therefore state whether it targets
the original tau-bench protocol or the current tau2/tau3 successor protocol.

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
