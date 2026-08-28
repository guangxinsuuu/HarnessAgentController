# Q2 Method Alignment Notes

This note records the implementation contract for the Scheduling Q2
quality-compute table. The table mixes structured optimization references,
native LLM inference, target-domain learned search, and solver-to-LLM
zero-shot transfer. Rows must not be promoted unless their code path matches
the stated method scope.

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
- Choose the majority group; use verified objective only as a tie-break inside
  majority-equivalent groups.

This is distinct from Best-of-N because objective minimization is not the
primary selection rule.

### Best-of-N

Reference method scope: generate multiple final answers and select the best one
with an external verifier or reward signal.

Scheduling adaptation:

- Generate `N` complete schedule samples with the same prompt family as
  Self-Consistency.
- Filter with the verifier.
- Select the valid schedule with the lowest objective.

## Tree of Thoughts / GPT-Frontier-Rank

Reference implementation target: Princeton NLP Tree-of-Thoughts
(`princeton-nlp/tree-of-thought-llm`).

Required local semantics before promoting to a strict ToT row:

- Level-wise or frontier BFS search.
- Generate/propose multiple candidate thoughts/states.
- Evaluate states with LLM value or vote prompts.
- Aggregate repeated value/vote samples when configured.
- Greedily select the top frontier states for the next step.

The current `GPT-frontier-rank / ToT-style` row is a useful online LLM
frontier-ranking baseline, but it should remain `ToT-style` unless the BFS
generate/evaluate/select protocol above is enabled and recorded in the
manifest.

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

If the implementation only computes `PRM score + exploration bonus` for a
priority queue, the row must be called `Target-PRM priority with exploration`
instead of `PRM-guided MCTS`.

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
- For Q2 compute accounting, a native/on-demand mode should count only proposal
  calls actually made after certification pruning.
