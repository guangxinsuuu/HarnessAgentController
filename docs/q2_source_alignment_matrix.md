# Q2 Source Alignment Matrix

This matrix compares each Scheduling Q2 row against the original method source
or paper-level implementation contract. It is stricter than the result table:
it separates exact structured references, native protocol adaptations, and
style baselines that must not be described as full official replications.

## Source Evidence

- OR-Tools CP-SAT: `https://github.com/google/or-tools`
  - Source revision pinned for Q2 evidence:
    `98c165af62df62b3056c2ee0fca66b24e79097cb`.
- Tree of Thoughts: `https://github.com/princeton-nlp/tree-of-thought-llm`
  - Source revision pinned for Q2 evidence:
    `8050e67d0e3a0fddc424d7fa5801538722a4c4cc`.
  - Official README describes `run.py` as implementing ToT+BFS and exposes
    `method_generate`, `method_evaluate`, and `method_select`.
  - Official `src/tot/methods/bfs.py` performs generation, evaluation, and
    selection in sequence, with greedy selection implemented by sorting values
    and keeping `n_select_sample`.
  - Machine-readable evidence now records raw-source refs to
    `src/tot/methods/bfs.py::def solve`, the `args.method_generate`
    sample/propose branches, the `args.method_evaluate` vote/value branches,
    the `args.method_select` sample/greedy branches, and
    `args.n_select_sample`.
- PRM800K / Let's Verify Step by Step:
  `https://github.com/openai/prm800k`
  - Source revision pinned for Q2 evidence:
    `7ecc794703b2877f63226f2477a49b34f9b25163`.
  - README describes PRM800K as a process supervision dataset with 800,000
    step-level correctness labels.
  - Released examples place labels under `label.steps[].completions[].rating`.
  - Machine-readable evidence records the released schema anchor
    `data/*.jsonl::label.steps[].completions[].rating`.
- Best-of-N / test-time compute scaling: `https://arxiv.org/abs/2408.03314`
  - Source revision pinned for Q2 evidence: `arXiv:2408.03314v1`.
  - The Q2 row uses the generic Best-of-N primitive from test-time scaling:
    independently sample multiple complete answers and select the best one
    under a verifier/reward. Because the local verifier sees the structured
    Scheduling objective, the row must be disclosed as `Best-of-N + verifier`.
- Chain-of-Thought: `https://arxiv.org/abs/2201.11903`
  - Source revision pinned for Q2 evidence: `arXiv:2201.11903v6`.
  - The Q2 Direct / CoT row uses the single sampled reasoning path primitive
    adapted to a complete Scheduling JSON answer.
- Self-Consistency: `https://arxiv.org/abs/2203.11171`
  - Source revision pinned for Q2 evidence: `arXiv:2203.11171v4`.
  - The Q2 row samples multiple independent reasoning paths and aggregates
    final schedules by answer consistency rather than verified objective.
- ReST-MCTS*: `https://github.com/THUDM/ReST-MCTS`
  - Source revision pinned for Q2 evidence:
    `2d5f488c3d6e24f99d50a9860e818383b1bb5883`.
  - README describes process-reward-guided tree search plus self-training.
  - At the algorithmic level, the required retained primitive is MCTS
    selection, expansion, simulation, and backpropagation.
  - `MCTS/mcts.py` implements MCTS rounds through `MCTS_search`,
    `executeRound`, `selectNode`, `expand`, `greedyPolicy`/`randomPolicy`
    rollout, `back_propagate`, and `getBestChild`.
  - Machine-readable evidence records raw-source function refs for
    `MCTS/mcts.py::def MCTS_search`, `::def executeRound`,
    `::def selectNode`, `::def expand`, `::def greedyPolicy`,
    `::def randomPolicy`, `::def back_propagate`, `::def getBestChild`,
    and `MCTS/base.py::class treeNode`.

## Machine-Checked Source Primitive Contract

Each Q2 artifact must now carry the `source_primitives` field from
`cgr_exp/q2_method_contracts.py` inside its method-alignment manifest. The
machine-readable official-source checklist lives in
`cgr_exp/q2_source_evidence.py`, which records the source URL, pinned source
revision, source files, function/schema-level code refs, source anchors,
official primitives, and local promotion gates for each
external baseline. The Table 4 audit rejects completed rows whose artifact does
not declare the expected primitives for that method. This is intentionally
stricter than a row name match: old or ad hoc artifacts cannot populate a
baseline row unless they also declare the official-method primitive that the
local Scheduling adaptation implements.

The same readiness gate also checks `claim_scope`,
`official_reproduction`, `required_table_wording`, `source_primitives`, and
`non_claims` against the method contract. Source evidence must cover every
contract primitive, not merely overlap with one related primitive. This matters
for adapted baselines: for example, a
ReST-MCTS-style artifact may not promote itself as a full official ReST-MCTS*
reproduction unless the full iterative self-training protocol has actually
been implemented and evaluated.

`audit-q2-local-primitives` adds the local side of the same check. For every
Q2 row, it records `primitive_alignment_json`, mapping each required
official-method primitive to concrete source anchors and file-line locations in
the local Scheduling adaptation. A completed row is not promotion-ready if any
contract primitive has no local mapping, or if a mapped anchor cannot be found
in the current source tree. This avoids a weak row-level check where a file
exists but one required algorithmic primitive is absent.

`verify-q2-source-refs` is the stricter source-ref verifier used by preflight.
It writes `q2_source_ref_verification.csv` and checks that GitHub-backed rows
use full pinned SHAs, raw GitHub URLs are pinned to the same SHA, symbol/schema
anchors are present, and paper-backed rows cite explicit arXiv versions. Its
default mode is offline and deterministic; `--online` additionally verifies
that the pinned revision and raw-source URLs are reachable, and records
`raw_anchor_locations` for matched raw-source anchors so the official-source
line evidence can be inspected without manually reopening each GitHub file.

Each current Q2 runner embeds the relevant `source_evidence` block in its own
manifest (`scheduling_structured_q2_metrics.json`,
`scheduling_native_q2_manifest.json`, `scheduling_tot_q2_manifest.json`, or
`scheduling_llm_pilot_metrics.json`). The readiness audit rejects external
baseline artifacts whose embedded evidence is missing or no longer matches the
pinned source evidence in `cgr_exp/q2_source_evidence.py`.

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
  checks that external baseline source URLs, pinned source revisions, source
  files, symbol/schema-level code refs, official primitives, and local
  promotion gates remain consistent with the method contracts. For GitHub
  sources, code refs must include the pinned commit rather than a floating
  branch name. For arXiv paper-only baselines, source revisions must include an
  explicit arXiv version such as `arXiv:2201.11903v6`.
- Run `python3 -m cgr_exp.cli verify-q2-source-refs --output-dir
  runs/q2_source_ref_verification --overwrite` as part of the same promotion
  gate. Add `--online` only when the machine has network access and you want to
  verify remote reachability of pinned revision/raw-source URLs.
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
- Method-alignment manifests must match the row contract's claim scope,
  official-reproduction flag, required wording, source primitives, and
  non-claims; a row name plus partial primitive list is not enough.
