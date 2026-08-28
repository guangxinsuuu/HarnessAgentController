# Q2 Table 4 Required Algorithms

This checklist defines what each Q2 Scheduling table row must implement before
it can be promoted as a result. It is source-aligned by method family: when a
baseline has an official codebase, the local row should mirror the official
algorithmic primitive rather than a convenient substitute.

The machine-readable version can be regenerated with:

```bash
python3 -m cgr_exp.cli build-q2-implementation-matrix \
  --output-dir runs/q2_implementation_matrix \
  --overwrite
```

`scripts/run_q2_table4.sh preflight` and `scripts/run_q2_table4.sh all` run this
matrix generation together with the source-evidence and table-readiness audits.

## Official Sources

- OR-Tools CP-SAT: https://github.com/google/or-tools,
  pinned at `98c165af62df62b3056c2ee0fca66b24e79097cb`.
- Tree of Thoughts: https://github.com/princeton-nlp/tree-of-thought-llm,
  pinned at `8050e67d0e3a0fddc424d7fa5801538722a4c4cc`.
- PRM800K / process supervision: https://github.com/openai/prm800k,
  pinned at `7ecc794703b2877f63226f2477a49b34f9b25163`.
- ReST-MCTS*: https://github.com/THUDM/ReST-MCTS,
  pinned at `2d5f488c3d6e24f99d50a9860e818383b1bb5883`.

## Source-to-Local Alignment

The table is an adaptation to Scheduling, not a claim that every external
baseline is reproduced on its original benchmark. Each baseline row must still
preserve the official algorithmic primitive that makes the method what it is.

| Row | Official primitive that must be preserved | Local Scheduling adaptation boundary |
| --- | --- | --- |
| CP-SAT | OR-Tools scheduling models use integer decision variables, optional interval variables, `NoOverlap` resource constraints, and `Minimize` objectives. | The local row receives exact structured `SchedulingTask` input and is a calibration reference with `llm_calls = 0`. It is not an LLM-agent baseline. |
| Direct / CoT | Chain-of-thought samples one reasoning trajectory before extracting the final answer. | The local row asks for one complete schedule with reasoning and validates the final typed schedule. It must not do frontier search. |
| Self-Consistency | Self-consistency samples multiple reasoning trajectories and marginalizes over final answers. | The local row aggregates complete schedules by canonical final job order. It must not select by verified objective, because that would become Best-of-N. |
| Best-of-N | Best-of-N draws multiple final candidates and selects one using a verifier/reward. | The local row is the stronger verifier-selected variant: invalid schedules are filtered and the lowest verified objective wins. The table footnote must disclose this. |
| Tree of Thoughts | Official BFS loops over `method_generate`, `method_evaluate`, and `method_select`; candidate thoughts are valued or voted on, then the top breadth is kept. | The local row must come from `run-scheduling-tot-q2`, which logs generate/evaluate/select events and value samples. Controller-isolation `GPT-frontier-rank` artifacts cannot fill this row. |
| Target-PRM | PRM800K is process supervision: intermediate solution steps receive labels/ratings, rather than only terminal outcome labels. | The local row trains a process/frontier ranker on target LLM-induced Scheduling states from disjoint target training instances. Outcome-PRM is only a diagnostic and cannot replace this row. |
| PRM-guided MCTS (ReST-MCTS-style) | ReST-MCTS uses MCTS rounds with selection, expansion, simulation/evaluation, and backpropagation, guided by step values/process rewards. | The local row must record positive MCTS primitive counts and PRM prior/value guidance. It remains `ReST-MCTS-style` unless the full iterative self-training pipeline is implemented. |
| Solver prior | This is the paper's method, not an external baseline. | It must train only on solver-induced source trajectories and freeze before real LLM-induced target evaluation. |
| Solver prior + certification | This is the paper's method plus verified pruning authority. | Learned scores may rank nodes only; irreversible pruning must be authorized by verified lower-bound or feasibility logic and audited separately. |

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
  `selection_rule = majority_job_order_then_first_valid`, exclude oracle
  future labels, provide multiple independent samples per held-out task, and
  avoid verified-objective tie-breaking that would collapse ties into
  Best-of-N.
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
- ReST-MCTS-style rows must not claim the complete ReST-MCTS* algorithm unless
  iterative target self-training and the official policy/reward update loop are
  also implemented and evaluated.
- Solver prior rows must include split manifests, checkpoint metadata,
  feature schema, and a manifest proving oracle labels are absent from
  inference features.
