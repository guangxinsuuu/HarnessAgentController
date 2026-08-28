# Q2 Method Alignment Notes

This note records the implementation contract for the Scheduling Q2
quality-compute table. The table mixes structured optimization references,
native LLM inference, target-domain learned search, and solver-to-LLM
zero-shot transfer. Rows must not be promoted unless their code path matches
the stated method scope.

The machine-readable version of these contracts lives in
`cgr_exp/q2_method_contracts.py`. Q2 runners and the table builder include the
relevant contract entries in their JSON manifests so that each reported row
can be audited against its claimed method scope.

## Structured Optimization References

### CP-SAT

Reference implementation target: Google OR-Tools CP-SAT scheduling models
(`google/or-tools`, CP-SAT scheduling examples).

Required local semantics:

- Use exact structured `SchedulingTask` inputs, not rendered natural language.
- Create integer start/end variables for every job.
- Encode release times and precedence constraints directly.
- Encode machine capacity with optional intervals and `NoOverlap`.
- Minimize `makespan + sum(tardiness_j)`, matching the local Scheduling
  objective.
- Report `llm_calls = 0` and `tokens = 0`.
- If OR-Tools is unavailable, mark the row `dependency_missing`; do not replace
  it with the exhaustive oracle while keeping the CP-SAT label.

Local command:

```bash
python3 -m cgr_exp.cli run-scheduling-structured-q2 \
  --output-dir runs/q2_structured_100_seed0 \
  --seed 0 \
  --train-count 200 \
  --validation-count 50 \
  --test-count 100 \
  --job-count 6 \
  --machine-count 3 \
  --cpsat-time-limit-seconds 5 \
  --overwrite
```

### EST / List Scheduling Heuristic

Required local semantics:

- Use exact structured `SchedulingTask` inputs.
- Repeatedly schedule one ready job on one allowed machine.
- Rank candidates by earliest feasible start, earliest finish, tardiness,
  due date, job id, and machine id.
- Report `llm_calls = 0` and `tokens = 0`.

This row is a calibration reference, not a direct competitor to LLM agents.

## Native LLM Inference

### Direct / CoT

Reference method scope: one native chain-of-thought-style sample that directly
produces a final answer.

Required local semantics:

- One LLM call per test instance.
- The model returns a complete schedule as typed JSON actions.
- The verifier validates the returned schedule.
- No oracle future labels or target trajectory training are used.

### Self-Consistency

Reference method scope: sample multiple reasoning paths and aggregate the final
answer distribution.

Scheduling adaptation:

- Generate `N` complete schedule samples with the same prompt family as Direct.
- Group valid outputs by canonical final job order.
- Choose the majority group; break ties by the earliest valid sample rather
  than by verified objective.

This is distinct from Best-of-N because objective minimization is not the
selection rule, including in all-distinct final-answer cases.

### Best-of-N

Reference method scope: generate multiple final answers and select the best one
with an external verifier or reward signal.

Scheduling adaptation:

- Generate `N` complete schedule samples with the same prompt family as
  Self-Consistency.
- Filter with the verifier.
- Select the valid schedule with the lowest objective.

Because this uses the structured verifier/objective for selection, captions or
notes should identify it as a verifier-selected Best-of-N variant. This is a
stronger and cleaner optimization baseline than unverified majority voting, but
it should not be confused with a purely frozen LLM answer-selection protocol.

## Tree of Thoughts / GPT-Frontier-Rank

Reference implementation target: Princeton NLP Tree-of-Thoughts
(`princeton-nlp/tree-of-thought-llm`).

Required local semantics before promoting to a strict ToT row:

- Level-wise or frontier BFS search.
- Generate/propose multiple candidate thoughts/states.
- Evaluate states with LLM value or vote prompts.
- Aggregate repeated value/vote samples when configured.
- Greedily select the top frontier states for the next step.

Current local status:

- `run-scheduling-tot-q2` implements a native ToT BFS row with
  propose/evaluate/select, configurable breadth, and averaged LLM value samples.
- It writes `scheduling_tot_q2_trace.jsonl` with generation, value-sampling,
  averaged score, and top-b selection events for protocol audit.
- Older `GPT-frontier-rank / ToT-style` runs are still online frontier-ranking
  diagnostics and should not be mixed with this stricter native ToT row.

## Target-Domain Learned Search

### Target-PRM

Reference method scope: process reward modeling, where intermediate states or
steps receive process-level supervision (`openai/prm800k` data release and
process-supervision setup).

Required local semantics:

- Training states are induced by target-domain LLM search on disjoint training
  instances.
- Test instances are held out.
- The model uses the same canonical state representation and comparable
  capacity as the solver-trained controller.
- Labels are process/frontier ranking labels, not outcome-only labels.

The `Outcome-PRM` diagnostic is not a replacement for this row.

### PRM-Guided MCTS (ReST-MCTS-Style)

Reference method scope: ReST-MCTS* combines policy/reward models and MCTS-style
tree search (`THUDM/ReST-MCTS`). The local row is explicitly a style baseline
rather than the full iterative self-training pipeline.

Required local semantics before using the MCTS name:

- Maintain per-search-node visit counts and value estimates.
- Select actions/nodes with UCB or PUCT-style exploration.
- Expand with the shared or native proposer.
- Evaluate leaves with the target PRM.
- Back up values along the selected path.

Current local status:

- The row uses a frontier-compatible MCTS-style policy with persistent visit
  counts, value sums, PUCT-style exploration, PRM leaf priors, expansion
  observation, and backup along the selected path.
- It still does not implement the full ReST-MCTS* iterative self-training loop,
  so the row must remain `ReST-MCTS-style`.
- A complete Q2 result should be rerun after MCTS implementation changes; older
  artifacts are valid historical diagnostics but should not be promoted as the
  final MCTS row.

## Ours

### Solver Prior

Required semantics:

- Train only on solver-induced source trajectories.
- Use no target LLM trajectory supervision before evaluation.
- Freeze the controller before target evaluation.
- Evaluate on LLM-induced target search.

### Solver Prior + Certification

Required semantics:

- Use the same solver-trained controller.
- Only verified lower bounds or verifier-proven infeasibility can authorize
  irreversible pruning.
- Learned scores and uncertain constraints may rank states but must not hard
  prune states.
- For Q2 compute accounting, the Scheduling LLM pilot emits per-policy
  on-demand proposal tokens in addition to shared-cache tokens, so certified
  pruning can receive credit only when it actually avoids proposal calls.
