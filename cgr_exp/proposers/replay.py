from __future__ import annotations


class ReplayProposer:
    def __init__(self, replay):
        self.replay = replay

    def propose(self, task, node, k, budget_state=None):
        del task, budget_state
        return list(self.replay.get(node.node_id, []))[:k]


def build_replay_from_task(task, k: int):
    replay = {}
    frontier = [task.root_node()]
    seen = set()
    while frontier:
        node = frontier.pop()
        if node.node_id in seen:
            continue
        seen.add(node.node_id)
        actions = task.candidate_actions(node, k)
        replay[node.node_id] = actions
        for idx, action in enumerate(actions):
            child = task.apply(node, action, child_index=idx)
            if child.objective_if_complete is None:
                frontier.append(child)
    return replay
