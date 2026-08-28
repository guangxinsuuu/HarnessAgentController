# Q2 Source Alignment Matrix

This matrix compares each Scheduling Q2 row against the original method source
or paper-level implementation contract. It is stricter than the result table:
it separates exact structured references, native protocol adaptations, and
style baselines that must not be described as full official replications.

## Source Evidence

- OR-Tools CP-SAT: `https://github.com/google/or-tools`
- Tree of Thoughts: `https://github.com/princeton-nlp/tree-of-thought-llm`
  - Official README describes `run.py` as implementing ToT+BFS and exposes
    `method_generate`, `method_evaluate`, and `method_select`.
  - Official `src/tot/methods/bfs.py` performs generation, evaluation, and
    selection in sequence, with greedy selection implemented by sorting values
    and keeping `n_select_sample`.
- PRM800K / Let's Verify Step by Step:
  `https://github.com/openai/prm800k`
  - README describes PRM800K as a process supervision dataset with 800,000
    step-level correctness labels.
  - Released examples place labels under `label.steps[].completions[].rating`.
- Best-of-N / test-time compute scaling: `https://arxiv.org/abs/2408.03314`
  - The Q2 row uses the generic Best-of-N primitive from test-time scaling:
    independently sample multiple complete answers and select the best one
    under a verifier/reward. Because the local verifier sees the structured
    Scheduling objective, the row must be disclosed as `Best-of-N + verifier`.
- ReST-MCTS*: `https://github.com/THUDM/ReST-MCTS`
  - README describes process-reward-guided tree search plus self-training.
  - `MCTS/mcts.py` implements MCTS rounds with selection, expansion, simulation, and backpropagation.

## Machine-Checked Source Primitive Contract

Each Q2 artifact must now carry the `source_primitives` field from
`cgr_exp/q2_method_contracts.py` inside its method-alignment manifest. The
machine-readable official-source checklist lives in
`cgr_exp/q2_source_evidence.py`, which records the source URL, source files,
source anchors, official primitives, and local promotion gates for each
external baseline. The Table 4 audit rejects completed rows whose artifact does
not declare the expected primitives for that method. This is intentionally
stricter than a row name match: old or ad hoc artifacts cannot populate a
baseline row unless they also declare the official-method primitive that the
local Scheduling adaptation implements.

Examples:

- Tree of Thoughts requires `method_generate`, `method_evaluate`,
  `method_select`, breadth-first search, and greedy top-b selection.
- PRM-style rows require intermediate-step/process-supervision primitives, not
  terminal outcome labels.
- ReST-MCTS-style requires MCTS selection, expansion, simulation/evaluation,
  backpropagation, and process reward guidance; full iterative self-training is
  explicitly outside the current local row.
- Solver-prior rows require solver-induced source trajectories, frozen
  zero-shot transfer, and feature-schema evidence excluding oracle labels.

## Row-by-Row Alignment

| Q2 row | Original-method primitive | Local implementation | Alignment status | Required wording |
| --- | --- | --- | --- | --- |
| CP-SAT | Structured CP-SAT scheduling optimization. | OR-Tools optional intervals, `NoOverlap`, release/precedence constraints, and local objective. | Direct structured reference. | Calibration reference, not LLM-agent competitor. |
| EST / list scheduling heuristic | Dispatch ready jobs by a handcrafted priority rule. | Earliest feasible start/finish with tardiness, due-date, job, and machine tie-breaks. | Direct heuristic reference. | Calibration reference. |
| Direct / CoT | One native reasoning sample to a final answer. | One complete-schedule JSON sample with reasoning field and verifier validation. | Native Scheduling adaptation. | Frozen LLM baseline. |
| Self-Consistency | Sample multiple reasoning paths and aggregate final answers. | Multiple complete schedules, grouped by canonical final job order, with earliest valid sample as the tie-break. | Native Scheduling adaptation. | Frozen LLM baseline; distinct from Best-of-N. |
| Best-of-N | Sample multiple final answers and select with verifier or reward. | Multiple complete schedules, verifier filters invalid outputs, minimum verified objective selected. | Strong verifier-selected adaptation. | Write `Best-of-N + verifier` or disclose verifier selection. |
| Tree of Thoughts | BFS with generate/propose, value/vote evaluation, greedy/sample selection. | `run-scheduling-tot-q2` proposes scheduling actions, LLM-ranks frontier states, averages value samples, keeps top breadth. | Native ToT BFS adaptation. | Do not use older controller-isolation GPT-frontier-rank artifacts as this row. |
| Target-PRM priority | Process reward model trained on intermediate-step labels. | Target LLM-induced Scheduling states, process/frontier ranking pairs, matched canonical features. | Domain-adapted PRM-style controller. | In-domain trained; not zero-shot; not Outcome-PRM. |
| Outcome-PRM | Outcome reward model diagnostic. | Terminal outcome supervision propagated to target LLM-induced states. | Diagnostic only. | Do not place in Target-PRM row. |
| PRM-guided MCTS (ReST-MCTS-style) | MCTS selection, expansion, simulation/value, backprop with process reward guidance; full method also self-trains. | PRM priors, PUCT/UCB-style selection, expansion observation, leaf value, backup along selected path. | Search-component adaptation only. | Must include `style`; do not claim full ReST-MCTS* self-training. |
| Solver prior | CGR source-trained external controller. | Solver-induced source traces only; frozen MLP; real LLM-induced target search. | Ours. | Zero-shot transfer. |
| Solver prior + certification | CGR controller plus verified pruning authority. | Same controller plus verifier/lower-bound-authorized pruning and on-demand proposal accounting. | Ours. | Learned score ranks only; verified logic prunes. |

## Promotion Rules

- Run `python3 -m cgr_exp.cli audit-q2-source-evidence --output-dir
  runs/q2_source_evidence_audit --overwrite` before promoting a Q2 table. This
  checks that external baseline source URLs, source files, official primitives,
  and local promotion gates remain consistent with the method contracts.
- A row is result-ready only if `audit-table4-q2` marks `result_ready=True`.
- A row is compute-ready for Q2 only if `audit-table4-q2` marks
  `compute_ready=True`.
- `audit-table5-q2` remains a backwards-compatible alias, but new Table 4/Q2
  commands should use `audit-table4-q2`.
- `shared-cache fallback accounting` rows can support controller isolation but
  need a qualification before use in an end-to-end quality-compute table.
- ReST-MCTS-style rows must be rerun after changes to
  `PRMGuidedMCTSStylePolicy`.
- Native ToT rows must come from `run-scheduling-tot-q2`, not from older
  `--include-gpt-frontier-ranking` diagnostic artifacts.
