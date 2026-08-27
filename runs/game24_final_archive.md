# Game24 Final Archive

Status: frozen after the real-LLM 100-case pilot. Do not tune Game24 further unless a reviewer specifically asks for an ablation.

Fixed configuration:

- Final split: `game24-0900` through `game24-0999`
- Real LLM: `gpt-4o-mini`
- Prompt: `prompt_v2` state-local Game24 action prompt
- Proposal replay: shared GPT proposal cache
- `proposal_k = 8`
- `node_budget = 80`
- Learned policy: MLP priority trained from exact offline Game24 ranking pairs

## Real-LLM Stability

| Count | Method | Success | Mean nodes | Total tokens |
|---:|---|---:|---:|---:|
| 20 | CoT | 17/20 | 18.35 | 586,533 |
| 20 | ToT-style | 17/20 | 22.85 | 1,054,799 |
| 20 | MLP priority | 17/20 | 14.35 | 586,533 |
| 20 | Oracle priority | 17/20 | 10.75 | 586,533 |
| 50 | CoT | 35/50 | 26.80 | 1,458,809 |
| 50 | ToT-style | 35/50 | 27.20 | 2,700,327 |
| 50 | MLP priority | 35/50 | 22.16 | 1,458,809 |
| 50 | Oracle priority | 35/50 | 17.00 | 1,458,809 |
| 100 | CoT | 62/100 | 29.17 | 2,857,668 |
| 100 | ToT-style | 62/100 | 29.20 | 5,401,695 |
| 100 | MLP priority | 62/100 | 23.93 | 2,857,668 |
| 100 | Oracle priority | 62/100 | 20.08 | 2,857,668 |

## Compute Claim

| Count | MLP / ToT tokens | Token saving | MLP node reduction vs ToT |
|---:|---:|---:|---:|
| 20 | 55.6% | 44.4% | 37.2% |
| 50 | 54.0% | 46.0% | 18.5% |
| 100 | 52.9% | 47.1% | 18.0% |

Main Game24 claim:

> On the 100-problem Game24 final split with shared GPT proposal replay, a learned MLP priority policy matches ToT-style success while reducing LLM token usage by 47.1% and expanded nodes by 18.0%.

This is a same-quality lower-compute result, not an accuracy win.

## Proposal Support

| Count | Proposal calls | Recall@K | Empty valid action rate |
|---:|---:|---:|---:|
| 20 | 1,082 | 0.989 | 0.92% |
| 50 | 2,692 | 0.992 | 1.04% |
| 100 | 5,275 | 0.990 | 0.74% |

100-case solvable-only accounting:

- Total instances: 100
- Theoretically solvable by exhaustive exact oracle: 81
- Theoretically unsolvable: 19
- Solved by CoT / ToT-style / MLP / Oracle under shared proposal replay: 62
- Solvable-only success: 62 / 81 = 76.5%
- Solvable but proposal-support failed: 19

The fact that Oracle priority fails on the same remaining solvable cases shows that these failures are proposal-support limited, not learned-ranker limited.

## Offline Pre-LLM Three-Seed Summary

The trusted offline baseline established that learned priority has search signal, while graph structure is not necessary on Game24.

Three-seed mean ordering:

```text
Oracle nodes = 40.24
MLP nodes = 52.16
Tree-GNN nodes = 54.46
Graph-GNN nodes = 54.46
DFS nodes = 88.94
Random nodes = 91.66
```

Offline conclusion:

```text
learned frontier priority works
graph structure is not yet necessary on Game24
```

## Paper Framing

Use Game24 to support:

- Learned frontier prioritization reduces search and LLM evaluation compute.
- Shared proposal replay separates proposal support from ranking policy quality.
- Oracle diagnostics identify proposal-support bottlenecks.

Do not use Game24 to claim:

- MLP improves final accuracy over ToT.
- Graph-GNN is necessary.
- Remaining failures are caused by ranking errors.

Next stage: freeze Game24 and move to Scheduling solver-to-LLM transfer.
