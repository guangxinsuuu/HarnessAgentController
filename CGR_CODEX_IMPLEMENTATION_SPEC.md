# CGR Full Implementation Specification

## Purpose

This document is the implementation contract for **Certified Graph-Guided Reasoning (CGR)**.

The goal is to build a reproducible research codebase that can test two coupled scientific hypotheses:

1. **Transferable search-prior hypothesis**

   Solver-induced and LLM-induced trajectories need not follow the same distribution, but they may share a partially common structural search-state representation. We test whether search-ranking knowledge learned from high-quality solver traces transfers to LLM-induced reasoning trajectories.

2. **Certified partial-structure hypothesis**

   When only part of a reasoning state is machine verifiable, conservative verification can still produce sound lower bounds and safe pruning. Missing verified structure should primarily reduce efficiency; false verified constraints are the correctness-critical failure mode.

The implementation MUST preserve the following separation of authority:

```text
LLM proposal policy
    decides which semantic branches exist

learned search policy
    decides which surviving branch receives the next unit of expensive compute

certificate rule
    decides which branch may be irreversibly removed
```

Formally:

```math
a \sim \pi_\theta(a \mid v, q)
```

```math
\pi_\phi(H_t, \mathcal F_t)
=
\arg\max_{v\in\mathcal F_t}
S_\phi^+(v \mid H_t)
```

and certified pruning is allowed only if a sound verifier/solver condition holds, for example

```math
L_V(v) \ge U_t
```

or verified infeasibility is proven.

A learned score, LLM self-evaluation, graph edge, episodic-memory score, or heuristic MUST NEVER directly authorize certified pruning.

---

# 1. Non-negotiable implementation principles

## 1.1 Tests before implementation

Before implementing any production module, Codex MUST first add or update tests under `tests/`.

Required workflow for every feature:

```text
1. add failing test
2. run test and confirm failure
3. implement minimal code
4. run targeted tests
5. run full suite
6. run smoke end-to-end experiment
7. verify generated artifacts
```

Do not merge or consider a feature complete if the full test suite does not pass.

## 1.2 No fabricated experimental results

The repository may contain expected qualitative sanity checks, but MUST NOT hardcode or invent CGR performance values.

All reported results must be generated from executable runs.

Tables must remain empty or contain `--` until data are produced.

## 1.3 Reproducibility

Every experiment MUST save:

```text
seed
git commit if available
Python version
dependency versions
full config
dataset split identifiers
model hyperparameters
training regime
search budget
proposal budget
verification corruption settings
feature schema version
graph edge configuration
memory configuration
wall-clock timestamps
```

Every source of randomness MUST use explicitly seeded RNG objects.

Do not use module-global randomness implicitly.

## 1.4 Separation of correctness and efficiency

Correctness-critical code and learned-efficiency code must be separated in the module structure.

Recommended split:

```text
certification/
    correctness-critical

search/
models/
memory/
    efficiency-oriented
```

No function in `models/` or `memory/` may call `prune_permanently()` directly.

Only the certification/controller layer may permanently remove a state.

## 1.5 Shared schema does not mean identical observed features

Represent each search node as

```math
x_v =
[
x_v^{univ},
x_v^{ver},
x_v^{sem},
m_v
]
```

where

```text
x_univ
    universal search features

x_ver
    verifier-dependent features

x_sem
    semantic features

m_v
    missingness mask
```

Level I, II, and III share the schema, but not every field must be observed.

Example:

```text
tool-use Level III state
    may have no numerical MILP lower bound

result
    bound fields are masked
    they are NOT replaced with fake zeros that mean a real observed zero
```

---

# 2. Research questions

The codebase must support experiments for the following questions.

## RQ1 Search quality

Under matched expensive-compute budgets, does CGR improve:

```text
success
objective quality
optimality gap
constraint satisfaction
```

## RQ2 Search efficiency

Does learned priority reduce:

```text
expanded nodes
LLM proposal calls
LLM evaluation calls
generated tokens
wall-clock latency
time-to-target-quality
```

## RQ3 Solver-to-LLM transfer

Can a search prior trained only on Level-I solver traces transfer zero-shot to Level-II/III LLM-induced trajectories?

## RQ4 Few-shot adaptation

How many target-domain LLM trajectories are required to adapt a solver-pretrained ranking model?

Required adaptation sizes where feasible:

```text
0
10
50
100
500
1000
```

## RQ5 Solver pretraining contribution

Compare:

```text
solver-only zero-shot
solver pretraining + target adaptation
LLM-only training
```

to determine whether solver supervision adds value beyond target-domain trajectories.

## RQ6 Proposal bottleneck

How strongly does proposal quality limit final performance?

Measure at least:

```text
ProposalRecall@K
schema validity
duplicate rate
invalid-action rate
semantic perturbation stability
```

## RQ7 Certification robustness

How do false negatives and false positives in verified constraints affect:

```text
bound tightness
false-prune rate
success
search cost
```

## RQ8 Graph necessity

Does graph augmentation improve ranking beyond:

```text
node-local MLP
path/history Transformer
Tree-GNN
```

If Graph-GNN does not outperform Tree-GNN, the code and paper must permit the scientific conclusion that graph augmentation is unnecessary.

## RQ9 Episodic-memory utility

Does optional episodic search memory improve structural transfer under distribution shift?

It must be tested as an extension, not assumed to be core.

## RQ10 Distribution shift

How does source-target discrepancy relate to:

```text
zero-shot ranking error
expanded nodes
LLM-call reduction
```

---

# 3. Required repository layout

Implement at minimum:

```text
.
├── README.md
├── pyproject.toml
├── requirements.txt
├── configs/
│   ├── smoke.json
│   ├── pilot.json
│   ├── research_level1.json
│   ├── research_level2.json
│   ├── research_level3.json
│   └── transfer_matrix.json
├── cgr_exp/
│   ├── __init__.py
│   ├── cli.py
│   ├── config.py
│   ├── seeds.py
│   ├── schemas.py
│   ├── nodes.py
│   ├── trajectories.py
│   ├── budgets.py
│   ├── metrics.py
│   ├── serialization.py
│   ├── datasets/
│   │   ├── __init__.py
│   │   ├── splits.py
│   │   ├── level1_assignment.py
│   │   ├── level1_scheduling.py
│   │   ├── level1_setcover.py
│   │   ├── level2_renderer.py
│   │   ├── level2_proposer_data.py
│   │   ├── level3_tool_env.py
│   │   └── perturbations.py
│   ├── proposers/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── simulator.py
│   │   ├── replay.py
│   │   └── llm_adapter.py
│   ├── verification/
│   │   ├── __init__.py
│   │   ├── contract.py
│   │   ├── conservative.py
│   │   ├── corruption.py
│   │   ├── level1_verifier.py
│   │   ├── level2_verifier.py
│   │   └── level3_verifier.py
│   ├── certification/
│   │   ├── __init__.py
│   │   ├── bounds.py
│   │   ├── incumbent.py
│   │   └── prune.py
│   ├── search/
│   │   ├── __init__.py
│   │   ├── controller.py
│   │   ├── frontier.py
│   │   ├── dfs.py
│   │   ├── random_policy.py
│   │   ├── best_bound.py
│   │   ├── oracle_policy.py
│   │   └── learned_policy.py
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── tree_builder.py
│   │   ├── graph_builder.py
│   │   ├── edge_types.py
│   │   └── features.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base_ranker.py
│   │   ├── linear_ranker.py
│   │   ├── mlp_ranker.py
│   │   ├── history_transformer.py
│   │   ├── tree_gnn.py
│   │   ├── graph_gnn.py
│   │   ├── llm_self_value.py
│   │   └── losses.py
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── episodes.py
│   │   ├── bank.py
│   │   ├── retrieval.py
│   │   └── gating.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── pairs.py
│   │   ├── solver_pretrain.py
│   │   ├── target_adapt.py
│   │   ├── llm_only.py
│   │   └── evaluate_ranker.py
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── domain_discrepancy.py
│   │   ├── ranking_analysis.py
│   │   ├── proposal_analysis.py
│   │   ├── certification_analysis.py
│   │   └── pareto.py
│   └── run_experiment.py
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── regression/
│   ├── property/
│   └── fixtures/
└── runs/
```

Exact names may differ, but all responsibilities above must exist.

---

# 4. Core data model

## 4.1 SearchNode

Create a serializable immutable or append-only search node.

Required fields:

```python
SearchNode:
    node_id: str
    parent_id: str | None
    depth: int
    task_id: str

    semantic_state: dict
    structured_state: dict

    verified_constraints: list
    uncertain_constraints: list

    lower_bound: float | None
    incumbent_at_creation: float | None
    objective_if_complete: float | None

    feasible_status: enum
    completion_status: enum

    proposal_action: TypedAction | None
    observation: dict | None

    universal_features: dict[str, float]
    verifier_features: dict[str, float | None]
    semantic_features: list[float] | None
    missingness_mask: dict[str, bool]

    accumulated_expensive_cost: CostLedger
    metadata: dict
```

## 4.2 TypedAction

Actions MUST be typed and schema validated.

Example:

```python
AssignAction(
    task="T7",
    resource="R2",
)
```

Do not allow free-form strings to enter the verifier directly.

Each action class must define:

```text
action_type
required arguments
argument types
serialization schema
validation method
```

## 4.3 VerificationResult

The verifier contract is:

```python
verify(state, action, observation) -> VerificationResult
```

Required fields:

```python
VerificationResult:
    verified_constraints
    uncertain_constraints
    feasible: bool | None
    lower_bound: float | None
    violations
    verification_evidence
    compiler_status
    verifier_backend
```

## 4.4 CostLedger

Every expensive operation must be counted explicitly.

```python
CostLedger:
    proposal_calls
    evaluation_calls
    reflection_calls
    input_tokens
    output_tokens
    solver_calls
    solver_seconds
    verifier_calls
    verifier_seconds
    ranker_calls
    ranker_seconds
    wall_clock_seconds
```

Never infer cost from node count alone.

---

# 5. Three policy layers

## 5.1 Proposal policy

Formal interface:

```python
TypedProposer.propose(
    task,
    node,
    k,
    budget_state,
) -> list[TypedAction]
```

The proposal policy corresponds to

```math
a \sim \pi_\theta(a \mid v,q)
```

Required implementations:

```text
DeterministicSimulatorProposer
ReplayProposer
RealLLMProposerAdapter
```

The simulator must support controlled:

```text
proposal recall
invalid-action probability
duplicate probability
semantic-noise probability
paraphrase sensitivity
branching factor K
```

## 5.2 Learned search policy

The learned policy chooses which frontier node gets the next expansion.

```python
select(frontier, search_history, memory=None) -> node_id
```

Formal rule:

```math
\pi_\phi(H_t,\mathcal F_t)
=
\arg\max_{v \in \mathcal F_t}
S_\phi^+(v \mid H_t)
```

The score may use:

```text
node-local features
path history
tree context
graph relations
optional episodic prior
```

It may defer states.

It may NOT irreversibly prune states.

## 5.3 Certificate rule

Only certification code can permanently prune.

For minimization:

```python
should_prune(node, incumbent) -> bool
```

may return true only if:

```text
verified infeasibility is proven
OR
sound lower_bound >= incumbent
```

The implementation must record the exact pruning reason.

---

# 6. Shared search-state feature schema

## 6.1 Universal features

Required where meaningful:

```text
depth
branch_factor
frontier_size_at_creation
num_siblings
num_failed_siblings
num_successful_siblings
ancestor_length
ancestor_failure_count
ancestor_bound_improvement_count
accumulated_search_cost
remaining_budget_fraction
duplicate_state_count
```

## 6.2 Verifier-dependent features

Required when available:

```text
lower_bound
normalized_bound_gap
verified_constraint_count
uncertain_constraint_count
verified_coverage
constraint_violation_count
feasibility_signal
bound_improvement_from_parent
incumbent_gap
```

Every unavailable value must be masked.

## 6.3 Semantic features

Pilot mode may use deterministic hand-built embeddings.

Research mode should support:

```text
frozen sentence embedding
LLM hidden-state embedding
task/action structured embedding
```

Semantic features must be optional so Level-I transfer can be tested without semantic leakage.

## 6.4 Missingness mask

Every feature group with unavailable fields must include a mask.

Test explicitly that:

```text
missing bound != observed numerical bound 0
```

---

# 7. Level-I datasets and solver-induced traces

Level I is a controlled source of high-quality search supervision.

It is NOT evidence by itself that CGR improves LLM reasoning.

## 7.1 Required Level-I problem families

Implement at least two for the first complete pilot:

```text
assignment
scheduling
```

Then add at least two of:

```text
set covering
capacitated facility location
independent set
combinatorial auction
```

## 7.2 Instance sizes

Smoke:

```text
very small
exact exhaustive or branch-and-bound search
< 5 seconds per instance
```

Pilot:

```text
small
medium
larger OOD size
```

Research:

```text
train sizes
IID test sizes
size-OOD test sizes
structural-OOD configurations
```

## 7.3 Trace content

For every expanded solver node store:

```text
state before expansion
branch action
children
feasibility
lower bound
incumbent
objective if complete
pruning reason
whether descendants contain optimal solution
best descendant objective
remaining exact expansions to best solution if computable
search depth
search cost
```

## 7.4 Ranking labels

Generate pairwise labels from frontier states.

Primary utility target:

```text
prefer state that reaches a better valid solution with lower future search cost
```

Use deterministic tie-breaking.

At minimum support:

```text
oracle-path membership
best descendant objective
remaining expansions to target quality
time-to-incumbent-improvement
```

---

# 8. Level-II natural-language constrained reasoning

Level II converts structured instances into semantic problems while keeping a hidden oracle structured formulation.

## 8.1 Data generation

For each hidden structured instance create multiple textual renderings:

```text
canonical wording
paraphrase A
paraphrase B
entity-renamed version
irrelevant-context version
mild ambiguity version
```

All renderings MUST map to the same hidden structured task.

## 8.2 Typed proposer simulator

Default offline proposer must support controlled errors:

```text
recall of useful action
invalid schema action
wrong entity
duplicate action
semantically plausible but suboptimal action
irrelevant action
```

It must generate typed actions, not free-form chain-of-thought text.

## 8.3 Real LLM adapter

Implement an interface but do not require network access for tests.

The adapter must record:

```text
model identifier
prompt template version
temperature
top_p
K
input tokens
output tokens
latency
raw output
parsed typed actions
parse failures
```

## 8.4 Level-II verifier

The verifier must:

```text
validate schema
resolve entities
validate types
compile only trusted typed predicates
separate verified vs uncertain information
compute sound bound when possible
```

Example:

```text
Natural language
    "Put T7 on R2 after T3"

LLM typed proposal
    Assign(T7, R2)
    After(T7, T3)

Verifier
    entity existence
    argument type
    allowed action
    deterministic state rule

Verified constraints
    assignment predicate
    precedence predicate
```

A phrase such as

```text
"prefer R2 if convenient"
```

must not become a certified hard constraint unless separately verified.

---

# 9. Level-III structured tool-use environment

Provide a local synthetic tool-use benchmark so tests do not depend on external services.

## 9.1 Required environment

Implement at least one stateful policy environment, for example:

```text
customer support / order operations
```

with:

```text
users
orders
payments
refunds
account ownership
policy thresholds
approval requirements
```

## 9.2 Typed tools

Example tools:

```text
lookup_order
lookup_customer
issue_refund
cancel_order
update_address
request_approval
```

## 9.3 Verifier backend

Machine-checkable predicates can include:

```text
database state
ownership
authorization
policy threshold
required approval
postcondition
```

Level III may have no numerical lower bound.

In that case:

```text
lower_bound = None
missingness mask = true
```

Certified pruning can still occur through verified infeasibility/policy impossibility if sound.

## 9.4 Optional external benchmark adapter

Provide adapter stubs for external structured tool benchmarks.

Do not make external data or network access a requirement for smoke or CI tests.

---

# 10. Distribution-shift design

The implementation must make source-target shift measurable.

Define:

```text
P_S
    solver-induced ranking-pair distribution

P_L
    LLM-induced ranking-pair distribution
```

Sources of shift include:

| Source | Level I | Level II/III |
|---|---|---|
| Node generation | structured branching | stochastic semantic proposal |
| Proposal support | controlled/exact search | may omit useful actions |
| State distribution | solver dynamics | additionally shaped by LLM behavior |
| Verified structure | usually high coverage | can be partial |
| Bound quality | typically strong | weaker or unavailable |
| Failure modes | infeasibility, poor objective | semantic error + infeasibility + poor search |
| Dominant cost | solver processing | LLM generation/evaluation + verification |

Required OOD conditions:

```text
IID
size OOD
structural OOD
semantic paraphrase OOD
entity-renaming OOD
proposal-quality shift
cross-family shift
Level-I -> Level-II
Level-I -> Level-III
```

---

# 11. Training regimes

## 11.1 Regime A Solver-only zero-shot

Train only on Level-I solver traces.

No Level-II/III training examples may be used.

Evaluate directly on target LLM-induced trajectories.

This is the strongest transfer test.

## 11.2 Regime B Solver pretraining plus few-shot adaptation

Pretrain on solver traces.

Then adapt with:

```text
10
50
100
500
1000
```

target trajectories where feasible.

Target data must be sampled only from target-train split.

## 11.3 Regime C LLM-only

Train only on Level-II/III target-domain trajectories.

No solver trace may enter training.

This isolates the contribution of solver pretraining.

## 11.4 Adaptation signal ablation

For Regime B compare:

```text
successful trajectories only
failed trajectories only
successful + failed
```

Do not assume failed trajectories are required.

---

# 12. Ranking losses

## 12.1 Primary pairwise ranking loss

Use pairwise preference:

```math
\mathcal L_{rank}
=
\sum_{(v^+,v^-)}
\log
\left(
1+\exp[-(S_\phi(v^+)-S_\phi(v^-))]
\right)
```

## 12.2 Optional relative cost-ranking loss

Do not regress absolute remaining token count by default.

If:

```math
C^*(v_i) < C^*(v_j)
```

prefer:

```math
v_i \succ v_j
```

with:

```math
\mathcal L_{cost-rank}
=
-\log \sigma(S_\phi(v_i)-S_\phi(v_j))
```

This is an optional target-adaptation signal.

It must be separately ablated.

---

# 13. Ranker baselines

All rankers must implement a common interface:

```python
score_nodes(nodes, search_context) -> dict[node_id, float]
```

Required models:

```text
DFS ordering
random ordering
certified best-bound
oracle ordering for diagnostics only
node-local linear/MLP
history/path model
Tree-GNN
Graph-GNN
LLM self-value adapter
Graph-GNN + optional episodic prior
```

## 13.1 Oracle policy

Oracle ordering is diagnostic only.

It must never be included as a deployable fair baseline.

It may use hidden ground truth to estimate the maximum possible value of good prioritization.

---

# 14. Tree and graph construction

## 14.1 Tree-GNN

Tree-GNN uses only:

```text
parent-child edges
```

## 14.2 Graph-GNN

Graph-GNN may additionally use:

```text
sibling/failure relation
shared verified constraint
equivalent structured state
shared entity
shared tool outcome
```

Every edge must have:

```text
edge_type
source
target
construction reason
```

## 14.3 Edge ablation

Required experiments:

```text
parent-child only
+ sibling/failure
+ shared constraints
+ equivalent states
+ tool-observation relations
all edges
```

If graph augmentation gives little or no benefit, report that result.

---

# 15. Episodic search memory

Episodic memory is OPTIONAL.

Core CGR must run without it.

## 15.1 Episode content

Store normalized search patterns rather than raw task-specific IDs.

Example episode:

```python
SearchEpisode:
    episode_id
    task_family
    task_embedding
    state_embedding_mean
    state_embedding_covariance_or_diag_variance
    utility
    normalized_depth_profile
    normalized_bound_gap_profile
    verified_coverage_profile
    branching_profile
    failure_neighborhood_profile
    metadata
```

## 15.2 Retrieval

Support similarity based on:

```text
task/query similarity
structural pattern similarity
episode utility
```

Weights must be configurable and logged.

## 15.3 Gated fusion

Use a clear fallback:

```math
S_\phi^+(v)
=
S_\phi(v)
+
g(v)p_{epi}(v)
```

where:

```math
g(v)\in[0,1]
```

If memory is irrelevant:

```math
g(v)\rightarrow0
```

and the model recovers the base GNN score.

## 15.4 Memory tests

Compare:

```text
no memory
random memory
IID relevant memory
OOD irrelevant memory
gated memory
```

Standard test evaluation must freeze the memory bank.

Do not add test instances to memory during standard evaluation.

---

# 16. Conservative verification experiments

## 16.1 False-negative corruption

Remove true constraints from `C^V`.

Required rates:

```text
0%
5%
10%
20%
30%
```

Maintain verified precision at 100%.

Expected sanity behavior, not a hard performance guarantee:

```text
coverage down
bound weaker
pruning opportunities down
nodes tend to increase
false-prune should remain zero under correct implementation
```

## 16.2 False-positive corruption

Inject incorrect constraints into `C^V`.

Required rates:

```text
0%
1%
2%
5%
10%
```

This intentionally violates the conservative assumption.

Measure:

```text
certificate validity
false-prune rate
final success
```

## 16.3 Uncertain-only corruption

Insert noisy candidate constraints into `C^U`.

They may influence learned features but MUST NEVER alter certified pruning.

A dedicated regression test must verify this.

---

# 17. Proposal-quality experiments

For each node define useful action set:

```math
\mathcal A^*(v)
```

and report:

```math
ProposalRecall@K
=
Pr[
\mathcal A_\theta(v)\cap\mathcal A^*(v)\neq\varnothing
]
```

Required semantic perturbations:

```text
canonical
paraphrase
entity rename
irrelevant context
mild ambiguity
```

Required proposer strengths:

```text
weak
medium
strong
```

For simulator mode these correspond to explicit controlled probabilities.

For real LLM mode they correspond to different model backbones or decoding settings.

Report relationship between:

```text
ProposalRecall@K
final success
ranking accuracy conditional on useful action support
```

---

# 18. Matched-budget experimental protocol

The primary fairness rule is:

```text
same task
same base proposer
same proposal branching factor K
same expensive-compute budget
same information access
```

Do NOT claim fair comparison by equalizing only node count.

## 18.1 Primary budgets

Support:

```text
LLM proposal-call budget
LLM total-call budget
generated-token budget
wall-clock budget
```

Recommended token budget sweep:

```text
2k
4k
8k
16k
32k
```

for real LLM experiments, adjusted to model/task scale.

For simulator mode use normalized expensive-call units.

## 18.2 Node expansion contract

For a fair expansion:

```text
select one frontier state
generate up to K typed candidates
verify each candidate
create child states
update budget ledger
```

Methods that use additional LLM value/evaluation calls must pay for those calls.

## 18.3 No hidden free LLM evaluation

If ToT, RAP, LATS, or LLM self-value uses an LLM evaluator:

```text
count the call
count input tokens
count output tokens
count latency
```

GNN scoring and solver/verifier time are separately recorded.

---

# 19. Primary metrics

Every run must report at least:

## Quality

```text
success
objective
normalized optimality gap
constraint violations
policy violations
```

## Certification

```text
false-prune count
false-prune rate
certificate validity
verified precision
verified recall
bound slack
```

## Proposal

```text
ProposalRecall@K
schema validity
invalid-action rate
duplicate-action rate
```

## Ranking

```text
pairwise ranking accuracy
pairwise ranking error
NDCG or MRR over frontier where meaningful
oracle-path top-1 rate
```

## Search

```text
expanded nodes
generated children
frontier peak size
time to first feasible
time to target quality
```

## Expensive compute

```text
proposal calls
evaluation calls
total LLM calls
input tokens
output tokens
solver seconds
verifier seconds
ranker seconds
wall-clock
```

## Transfer

```text
source ranking risk
target ranking risk
domain discrepancy proxy
adaptation sample count
```

---

# 20. Domain discrepancy proxy

Do not pretend to know exact high-dimensional TV distance.

Implement a domain-classifier proxy.

Input:

```text
shared state features or pair features
```

Label:

```text
solver-induced
LLM-induced
```

Train/evaluate with held-out examples.

Report:

```text
domain classifier accuracy
AUC
proxy discrepancy value
```

Then relate discrepancy to:

```text
zero-shot ranking error
expanded nodes
LLM calls
```

The exact theoretical proxy formula must be documented in code comments and result metadata.

---

# 21. Required experiment phases

## Phase 0 Correctness infrastructure

Goal:

```text
prove the code obeys CGR invariants
```

Tasks:

```text
SearchNode schema
TypedAction schema
VerificationResult
CostLedger
exact small solver
frontier/controller
conservative prune rule
serialization
```

Exit criteria:

```text
all unit tests pass
all property tests pass
false-prune = 0 under conservative verifier on exhaustive tiny problems
same seed reproduces byte-equivalent core trace fields
```

## Phase 1 Level-I search-prior pilot

Datasets:

```text
assignment
scheduling
```

Models:

```text
DFS
random
best-bound
oracle
MLP/linear
Tree-GNN or graph-compatible research backend
```

Results:

```text
ranking accuracy
expanded nodes
time to optimal
bound statistics
```

Exit criterion:

```text
learned model trains end to end
generated traces can be replayed
oracle ordering is best or near-best as a sanity ceiling
```

Do NOT require learned policy to beat every heuristic as a unit-test condition.

## Phase 2 Level-II simulator transfer

Generate natural-language renderings and typed proposer errors.

Run:

```text
solver-only zero-shot
solver + few-shot
LLM-only
```

Results:

```text
ProposalRecall@K
ranking error
success
expanded nodes
proposer calls
domain discrepancy
```

This is the first real solver-to-LLM transfer experiment.

## Phase 3 Certification robustness

Run:

```text
false-negative corruption
false-positive corruption
uncertain-only corruption
```

Produce:

```text
coverage -> bound slack
coverage -> pruning
false positive -> false prune
```

## Phase 4 Representation ablation

Compare:

```text
MLP
history/path model
Tree-GNN
Graph-GNN
```

Then edge ablations.

## Phase 5 Episodic-memory extension

Compare:

```text
GNN only
GNN + relevant memory
GNN + irrelevant memory
GNN + gated memory
```

## Phase 6 Level-III tool-use

Run on local synthetic tool environment first.

Then optionally connect to an external structured tool benchmark.

## Phase 7 Real-LLM matched-budget evaluation

Replace simulator with real LLM proposer.

Run primary baselines under equal expensive-compute budgets.

Produce final paper tables and Pareto plots.

---

# 22. Dataset split requirements

Every dataset family must have explicit:

```text
train
validation
test IID
test size OOD
test structural OOD
```

Level-II additionally:

```text
semantic paraphrase OOD
entity-renaming OOD
proposal-quality OOD
```

Rules:

```text
no test instance in training traces
no test trajectory in episodic memory
no target-test trajectory in few-shot adaptation
no validation task in final test aggregate
```

For generated datasets save the generator parameters and instance IDs.

---

# 23. Required result artifacts

Every run directory:

```text
runs/<run_id>/
├── config.json
├── environment.json
├── metrics.json
├── metrics_by_instance.jsonl
├── primary_table.csv
├── transfer_table.csv
├── ranking_table.csv
├── certification_table.csv
├── proposal_table.csv
├── budget_table.csv
├── domain_shift_table.csv
├── trace_sample.jsonl
├── train_history.jsonl
├── stdout.log
└── manifest.json
```

Optional research plots:

```text
quality_compute_pareto.png
transfer_sample_efficiency.png
distribution_shift_vs_ranking.png
distribution_shift_vs_llm_calls.png
coverage_vs_bound.png
coverage_vs_nodes.png
false_positive_vs_false_prune.png
representation_ablation.png
edge_ablation.png
proposal_recall_vs_success.png
```

---

# 24. Paper-facing tables

## 24.1 Primary matched-budget table

Columns:

```text
Method
Success
Gap
Violations
False prune
LLM calls
Tokens
Nodes
Latency
```

Rows:

```text
CoT
Self-consistency
Best-of-N
ToT
RAP/MCTS
LATS-style
LLM self-value
Certified best-bound
Tree-GNN
Graph-GNN
CGR
CGR + optional episodic prior
```

Only include a baseline if a faithful compatible implementation exists.

Do not include a fake RAP/LATS row with a materially different protocol.

## 24.2 Transfer table

Rows:

```text
Solver zero-shot
Solver + 10
Solver + 50
Solver + 100
Solver + 500
Solver + 1000
LLM-only
```

Columns:

```text
Target trajectories used
Ranking accuracy
Success
Expanded nodes
LLM calls
```

## 24.3 Certification robustness table

Rows:

```text
no corruption
FN 5
FN 10
FN 20
FN 30
FP 1
FP 2
FP 5
uncertain-only 10
```

Columns:

```text
verified precision
verified recall
bound slack
false prune
nodes
success
```

---

# 25. Statistical reporting

For research presets:

```text
minimum 3 seeds
prefer 5 seeds for main results
paired instance seeds across methods
```

Report:

```text
mean
median where heavy-tailed
standard deviation
95% bootstrap confidence interval
p95 for latency/search cost
```

Do not select only the best seed.

For matched-budget comparisons, use the same instance ordering across methods.

---

# 26. Mandatory tests

This section is a hard implementation requirement.

## 26.1 Unit tests

### Search node schema

Test:

```text
serialization round trip
missingness mask
parent-child consistency
depth consistency
```

### Typed actions

Test:

```text
valid action accepted
wrong type rejected
missing argument rejected
unknown entity rejected by verifier
```

### Cost ledger

Test exact accounting for:

```text
proposal calls
evaluation calls
token counts
solver calls
wall time fields
```

### Verifier contract

Test:

```text
verified constraint
uncertain constraint
no-bound task
infeasible task
```

### Conservative pruning

Tiny exhaustive problems:

```text
if C_V subset of true constraints
false prune must equal zero
```

### Uncertain information isolation

Change only `C_U`.

Assert:

```text
certified pruning decision unchanged
```

### Learned-priority isolation

Change GNN score arbitrarily.

Assert:

```text
pruning decision unchanged
```

### Episodic-memory isolation

Change memory score arbitrarily.

Assert:

```text
pruning decision unchanged
```

### Missing bound

For Level III:

```text
lower_bound=None
```

must not become:

```text
lower_bound=0
```

and must not trigger bound pruning.

## 26.2 Property tests

Using many random tiny instances:

### Sound conservative bound property

For minimization:

```math
L_V(v) <= true_best_descendant(v)
```

under conservative verified subsets.

### Monotonic verified coverage

For nested verified sets:

```math
C_V^1 \subseteq C_V^2
```

assert:

```math
L_1 <= L_2
```

when using the exact relaxation implementation.

### Pruning safety

If:

```math
L_V(v) >= U
```

verify by exhaustive descendant enumeration that no descendant improves `U`.

### Ordering invariance

Run same finite exact search with:

```text
DFS
random
reverse
learned fake scores
```

with unlimited budget.

Assert all return the same best reachable objective under the same proposal support and sound pruning.

## 26.3 Transfer-regime leakage tests

### Zero-shot

Assert target trajectory IDs used for training:

```text
empty
```

### Few-shot

Assert exact target adaptation count.

### LLM-only

Assert solver trace IDs in training:

```text
empty
```

### Episodic memory

Assert target-test IDs in memory bank:

```text
empty
```

## 26.4 Dataset split tests

Assert no overlap among:

```text
train
validation
test IID
test OOD
```

Use stable task IDs.

## 26.5 Budget fairness tests

Given identical budget:

```text
no method may exceed token/call budget
```

If a method uses evaluator calls:

```text
they must be charged
```

If budget is exceeded mid-expansion, behavior must be deterministic and documented.

## 26.6 Graph construction tests

For every graph edge:

```text
edge source exists
edge target exists
edge type valid
edge construction condition true
```

Tree-GNN fixture must contain only parent-child edges.

Graph-GNN fixture must contain requested extra edge types.

## 26.7 Ranking-pair tests

Pair labels must be:

```text
deterministic
antisymmetric
free of train/test leakage
```

If two nodes tie exactly under utility:

```text
skip pair
OR
use documented deterministic tie policy
```

## 26.8 ProposalRecall@K tests

Construct fixtures where useful action is:

```text
present
absent
duplicated
invalidly typed
```

Only valid useful typed actions count toward recall.

## 26.9 Domain-discrepancy tests

Use synthetic distributions:

```text
identical source/target
clearly separable source/target
```

Assert classifier discrepancy is lower in the first case.

Do not require an exact numerical constant across model/library versions.

## 26.10 End-to-end smoke test

Command:

```bash
python -m cgr_exp.run_experiment --preset smoke
```

Must:

```text
finish quickly
create run directory
write all required core JSON/CSV artifacts
contain no NaN in mandatory metrics
produce at least one solver trace
produce at least one Level-II trace
run zero-shot transfer
run conservative pruning checks
```

CI smoke target:

```text
prefer < 60 seconds
hard target < 120 seconds
```

---

# 27. Regression tests for known dangerous bugs

Add explicit regression tests for:

## Bug A Learned score accidentally prunes

A node with terrible learned score but valid lower bound below incumbent must remain reachable.

## Bug B Uncertain constraint enters solver

A constraint marked uncertain must never change the certified solver model.

## Bug C Missing bound treated as zero

Must never happen.

## Bug D Test-memory leakage

Standard evaluation memory bank must remain frozen.

## Bug E Few-shot uses test labels

Adaptation may only use target-train trajectories.

## Bug F Node count treated as total cost

Primary budget accounting must include evaluator/proposal calls separately.

## Bug G Random baseline unfairly uses different proposals

All search policies in a matched comparison must use identical proposer behavior or replayed proposal outputs.

## Bug H Oracle information leaks into learned features

Fields such as:

```text
contains_optimal_descendant
future_best_objective
future_remaining_cost
```

may be labels only.

They must never appear in inference features.

## Bug I Semantic renderer changes hidden problem

Paraphrase/entity rename must preserve the exact hidden structured instance.

## Bug J Graph edge uses future information

Graph edges at time `t` may use only information available by time `t`.

Do not construct an inference graph using future trace outcomes.

---

# 28. Recommended replay protocol for fair search comparison

To isolate search policy from stochastic proposal noise, support a replay mode.

For each task and state:

```text
pre-generate or deterministically seed candidate proposals
```

Then compare:

```text
DFS
random
best-bound
MLP
Tree-GNN
Graph-GNN
CGR
```

using the same available candidate branches.

Run a second experiment with live stochastic proposal to measure full-system behavior.

This yields:

```text
Policy-only comparison
Full-system comparison
```

Both are required for a strong paper.

---

# 29. Real-LLM integration contract

The code must not depend on one provider.

Define:

```python
class LLMClient(Protocol):
    def generate_typed_actions(...): ...
    def score_state(...): ...
```

Adapters may include:

```text
OpenAI-compatible endpoint
local Transformers model
mock/replay client
```

Never commit secrets.

Read credentials only from environment variables.

Tests must use mocks.

---

# 30. CLI requirements

Required commands:

```bash
python -m cgr_exp.run_experiment --preset smoke
python -m cgr_exp.run_experiment --preset pilot --seed 7
python -m cgr_exp.run_experiment --config configs/research_level1.json
```

Recommended additional commands:

```bash
python -m cgr_exp.cli generate-level1
python -m cgr_exp.cli generate-level2
python -m cgr_exp.cli train-ranker
python -m cgr_exp.cli adapt-ranker
python -m cgr_exp.cli evaluate-search
python -m cgr_exp.cli run-certification-corruption
python -m cgr_exp.cli run-transfer-matrix
python -m cgr_exp.cli analyze-domain-shift
python -m cgr_exp.cli build-tables
```

All commands must support:

```text
--seed
--output-dir
--overwrite false by default
```

---

# 31. Configuration contract

Use one serializable configuration object.

Example:

```json
{
  "seed": 7,
  "dataset": {
    "level": 2,
    "family": "scheduling",
    "train_instances": 500,
    "val_instances": 100,
    "test_instances": 200,
    "ood_mode": "paraphrase"
  },
  "proposer": {
    "type": "simulator",
    "k": 4,
    "useful_action_recall": 0.8,
    "invalid_action_prob": 0.05,
    "duplicate_prob": 0.05
  },
  "verification": {
    "false_negative_rate": 0.0,
    "false_positive_rate": 0.0,
    "uncertain_noise_rate": 0.0
  },
  "search": {
    "policy": "graph_gnn",
    "node_budget": 200,
    "proposal_call_budget": 100
  },
  "training": {
    "regime": "solver_zero_shot",
    "target_adaptation_trajectories": 0
  },
  "graph": {
    "parent_child": true,
    "sibling_failure": true,
    "shared_constraint": true,
    "equivalent_state": true,
    "tool_observation": false
  },
  "memory": {
    "enabled": false,
    "top_r": 4,
    "frozen_at_test": true
  }
}
```

Validate config before any run starts.

---

# 32. Expected sanity checks

These are not paper claims and should not be hard-coded as success criteria, except where they are correctness invariants.

## Hard invariants

Must hold:

```text
conservative verified constraints + exact verifier => false prune = 0
uncertain-only changes cannot alter certified pruning
learned score cannot alter certified pruning
unlimited exact search returns same best reachable objective regardless of node ordering
zero-shot training contains zero target trajectories
test data never enters memory/training
```

## Soft sanity expectations

Investigate, but do not force:

```text
oracle ordering tends to expand fewer nodes than random
better ranking tends to reduce expansions
higher verified recall at perfect precision tends to tighten bounds
few-shot adaptation may improve target ranking under shift
graph augmentation may or may not beat tree representation
episodic memory may or may not help OOD transfer
```

A negative result is scientifically valid.

---

# 33. Final paper figures to generate

The implementation should eventually generate data for these figures.

## Figure A Quality-compute Pareto

```text
x = total LLM tokens or normalized expensive compute
y = success / task quality
series = baselines + CGR
```

## Figure B Solver-to-LLM transfer sample efficiency

```text
x = target adaptation trajectories
y1 = target ranking accuracy
y2 = LLM-call reduction
```

Include:

```text
zero-shot
10
50
100
500
1000
LLM-only reference
```

## Figure C Representation ablation

```text
MLP
history Transformer
Tree-GNN
Graph-GNN
Graph-GNN + episodic
```

Metrics:

```text
ranking error
expanded nodes
LLM calls
```

## Figure D Verification coverage chain

```text
verified recall
    ->
bound slack
    ->
certified pruning
    ->
nodes
```

with verified precision fixed at 100%.

## Figure E False-positive risk

```text
x = verified false-positive rate
y = false-prune rate / certificate invalidity
```

## Figure F Distribution shift

```text
x = domain discrepancy proxy
y1 = zero-shot ranking error
y2 = expanded nodes
y3 = LLM-call reduction
```

## Figure G Proposal bottleneck

```text
x = ProposalRecall@K
y = final success
```

---

# 34. Minimum viable publishable experiment set

Before calling the implementation paper-ready, complete at least:

```text
2 Level-I families
1 Level-II family with >= 4 semantic perturbations
1 local Level-III tool environment
3 transfer regimes
5 search/ranking baselines
Tree-GNN vs Graph-GNN
certification corruption experiment
ProposalRecall@K experiment
domain discrepancy experiment
quality-compute budget sweep
>= 3 seeds
```

Strong version:

```text
4-6 Level-I families
multiple Level-II families
one external structured tool benchmark
real LLM proposer with >= 2 model strengths
5 seeds for main results
full episodic-memory ablation
```

---

# 35. Implementation order Codex MUST follow

Do not start by implementing a GNN.

Implement in this order:

```text
Phase A
tests
schemas
exact tiny problems
verifier contract
certified pruning
budget ledger

Phase B
search controller
DFS/random/best-bound
trace generation
replay

Phase C
feature schema
ranking pairs
simple linear/MLP ranking baseline
zero-shot transfer pipeline

Phase D
Level-II renderer
typed proposer simulator
semantic perturbations
proposal metrics

Phase E
few-shot and LLM-only training
domain discrepancy analysis

Phase F
Tree-GNN
Graph-GNN
edge ablations

Phase G
optional episodic memory
gated fallback

Phase H
Level-III local tool environment

Phase I
real LLM adapter
external benchmark adapters
paper tables and plots
```

At the end of every phase:

```text
run full tests
run smoke preset
inspect generated metrics
inspect at least one serialized trace
```

---

# 36. Definition of done

The repository is considered a complete CGR research implementation only when all of the following hold:

```text
[ ] full test suite passes
[ ] smoke run passes offline
[ ] Level-I traces reproducible
[ ] Level-II proposer simulator reproducible
[ ] Level-III local tool environment runnable
[ ] conservative pruning invariant tested exhaustively on tiny tasks
[ ] zero-shot transfer executable
[ ] few-shot adaptation executable
[ ] LLM-only regime executable
[ ] proposal recall measured
[ ] domain discrepancy measured
[ ] Tree-GNN/Graph-GNN ablation executable
[ ] episodic memory can be disabled without changing core behavior
[ ] uncertain constraints cannot authorize pruning
[ ] matched-budget accounting includes every expensive LLM call
[ ] primary result tables generated from run artifacts
[ ] no test/train/memory leakage
[ ] all final results trace back to saved config + seed
```

---

# 37. One-sentence method summary for code comments and README

Use this exact conceptual summary:

> CGR treats LLM reasoning as frontier search: the LLM proposes semantic branches, a transferable learned ranking policy allocates expensive compute across surviving partial states, and only conservatively verified structure can authorize irreversible pruning.

---

# 38. Scientific interpretation rules

The codebase must make it possible to reach any of the following conclusions without changing the protocol.

## Outcome A Strongest result

```text
solver-only zero-shot already improves LLM-induced search
```

Interpretation:

```text
high-quality solver traces contain transferable structural search knowledge
```

## Outcome B Adaptation result

```text
zero-shot weak
few-shot solver-pretrained model improves rapidly
```

Interpretation:

```text
shared structure exists but source-target distribution shift requires modest target adaptation
```

## Outcome C Solver pretraining weak

```text
LLM-only matches solver-pretrained models
```

Interpretation:

```text
solver trace pretraining contributes limited additional value under this representation/task
```

## Outcome D Graph unnecessary

```text
Tree-GNN ~= Graph-GNN
```

Interpretation:

```text
graph augmentation is not necessary for the tested tasks
```

## Outcome E Episodic memory unnecessary

```text
GNN ~= GNN + episodic prior
```

Interpretation:

```text
episodic transfer does not add value beyond learned structural ranking
```

## Outcome F Certification tradeoff confirmed

```text
false negatives increase compute without false pruning
false positives increase false-prune risk
```

Interpretation:

```text
precision is correctness-critical; recall is primarily efficiency-critical
```

The implementation must never be written so that only the desired positive result can appear.

---

# 39. First Codex task

Codex should begin with ONLY the following:

```text
1. create repository skeleton
2. create tests/ structure
3. implement schemas for SearchNode, TypedAction, VerificationResult, CostLedger
4. implement tiny assignment problem fixture
5. implement exhaustive oracle
6. implement conservative verifier
7. write property tests proving zero false prune under conservative subsets
8. implement DFS/random/best-bound search
9. implement trace serialization
10. add smoke CLI
```

Do not implement the GNN, episodic memory, or real LLM adapter until the correctness foundation passes all tests.

After Phase A is complete, report:

```text
tests added
tests passing
smoke command
example run directory
known limitations
next phase
```

Only then proceed to the learned ranking implementation.
