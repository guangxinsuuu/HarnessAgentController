# Q2 Table 4 Required Algorithms

This checklist defines what each Q2 Scheduling table row must implement before
it can be promoted as a result. It is source-aligned by method family: when a
baseline has an official codebase, the local row should mirror the official
algorithmic primitive rather than a convenient substitute.

## Official Sources

- OR-Tools CP-SAT: https://github.com/google/or-tools
- Tree of Thoughts: https://github.com/princeton-nlp/tree-of-thought-llm
- PRM800K / process supervision: https://github.com/openai/prm800k
- ReST-MCTS*: https://github.com/THUDM/ReST-MCTS

## Rows

| Row | Required algorithm | Local code path | Promotion status |
| --- | --- | --- | --- |
| CP-SAT | Structured CP-SAT model with interval variables, machine `NoOverlap`, release/precedence constraints, and exact objective minimization. | `cgr_exp/scheduling_structured_q2.py` | Implemented as a structured-input calibration reference. |
| EST / list scheduling heuristic | Deterministic dispatch heuristic over ready jobs using earliest feasible placement and fixed tie-breaks. | `cgr_exp/scheduling_structured_q2.py` | Implemented as a structured-input calibration reference. |
| Direct / CoT | One native LLM reasoning sample that emits a complete schedule. This is the direct/CoT sampling primitive, not tree search. | `cgr_exp/scheduling_native_q2.py` | Implemented; requires complete 100-instance artifact. |
| Self-Consistency | Multiple independent CoT samples followed by answer aggregation. Aggregation must operate on final schedules, not intermediate oracle labels. | `cgr_exp/scheduling_native_q2.py` | Implemented; requires complete 100-instance artifact. |
| Best-of-N | Multiple independent final-answer samples followed by verifier/reward selection. If the verifier selects the minimum verified objective, the row must be named or footnoted as verifier-selected. | `cgr_exp/scheduling_native_q2.py` | Implemented as `Best-of-N + verifier`; requires complete 100-instance artifact. |
| Tree of Thoughts | Native ToT BFS adaptation: generate candidate thoughts/actions, evaluate frontier states with LLM value/vote style scoring, and select the top breadth for the next level. | `cgr_exp/scheduling_tot_q2.py` | Implemented; only artifacts from `run-scheduling-tot-q2` may populate this row. Older GPT-frontier-rank diagnostics must not populate this row. |
| Target-PRM | Process-reward-style controller trained on target LLM-induced intermediate Scheduling states from disjoint target training instances. | `cgr_exp/scheduling_transfer.py` | Implemented as in-domain PRM-style priority; not zero-shot. |
| PRM-guided MCTS (ReST-MCTS-style) | MCTS search component using a process reward/value model for selection, expansion evaluation, and backup. | `cgr_exp/scheduling_transfer.py` | Implemented only as the ReST-MCTS-style search component. Do not claim full ReST-MCTS* unless iterative policy/value self-training is added. |
| Solver prior | Solver-induced source trajectories train a frozen controller, then evaluate zero-shot on real LLM-induced target search. | `cgr_exp/scheduling_transfer.py` | Implemented as our zero-shot transfer row. |
| Solver prior + certification | Same frozen solver-trained controller plus pruning authorized only by verified feasibility/lower-bound logic. Learned scores may rank but must not authorize pruning. | `cgr_exp/scheduling_transfer.py`, `cgr_exp/certification/prune.py` | Implemented; Q2 promotion should use on-demand LLM proposal accounting. |

## Non-Negotiable Fairness Rules

1. The held-out 100-instance Scheduling test split must be identical across
   rows.
2. Target-PRM and Solver prior must use matched canonical state features and
   comparable controller capacity; the intended difference is the training
   state distribution.
3. Oracle future labels, exact future costs, and verifier-only lower-bound
   authority must not appear in learned inference features.
4. Controller-isolation artifacts are not interchangeable with native
   end-to-end artifacts. In particular, the ToT row must come from
   `run-scheduling-tot-q2`, not from the older GPT-frontier-rank diagnostic.
5. Shared-cache results can support controller-isolation claims. End-to-end
   quality-compute claims must use row-appropriate LLM call and token
   accounting.
6. The ReST-MCTS row must retain the word `style` unless the full official
   iterative self-training loop is implemented and evaluated.

## Artifact Promotion Gates

- Direct / CoT must declare `selection_rule = first_valid`, exclude oracle
  future labels, and provide exactly one sample per held-out task.
- Self-Consistency must declare
  `selection_rule = majority_job_order_then_objective`, exclude oracle future
  labels, and provide multiple independent samples per held-out task.
- Best-of-N must declare `selection_rule = best_verified_objective`, disclose
  verifier objective selection, exclude oracle future labels, and provide the
  declared number of samples per held-out task.
- Tree of Thoughts must provide the strict ToT BFS manifest with
  `method_generate`, `method_evaluate`, and `method_select`, plus a trace with
  generation, evaluation, selection, value-sample records, and one finish event
  per held-out rendering.
- Target-PRM and ReST-MCTS-style rows must include target PRM trace records,
  training pairs, a PRM ranker artifact, Q2 method alignment, and no overlap
  between target PRM training task IDs and held-out evaluation task IDs.
- ReST-MCTS-style rows must additionally declare the MCTS search primitives:
  selection, expansion, simulation/evaluation, and backpropagation, plus PRM
  process-reward guidance. A plain Target-PRM priority artifact is not enough.
- Solver prior rows must include split manifests, checkpoint metadata,
  feature schema, and a manifest proving oracle labels are absent from
  inference features.
