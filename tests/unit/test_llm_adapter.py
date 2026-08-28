from __future__ import annotations

import http.client
import json
import random
import unittest

from cgr_exp.datasets.game24 import Game24Action, Game24Task
from cgr_exp.datasets.level1_scheduling import ScheduleAction, generate_scheduling_fixture
from cgr_exp.proposers.llm_adapter import EnvConfiguredLLMClient, MockLLMClient, OpenAICompatibleHTTPClient, RealLLMProposerAdapter
from cgr_exp.budgets import BudgetState


class LLMAdapterTest(unittest.TestCase):
    def test_adapter_records_metadata_for_successful_parse(self) -> None:
        client = MockLLMClient(raw_output='[{"action_type":"assign","task":"T1","resource":"R2"}]', input_tokens=12, output_tokens=9, latency=0.01)
        adapter = RealLLMProposerAdapter(client=client, model="mock-model", prompt_template_version="v1", temperature=0.2, top_p=0.9)
        actions = adapter.propose(task={"task_id": "t"}, node={"node_id": "n"}, k=3)
        self.assertEqual(len(actions), 1)
        record = adapter.call_records[-1]
        self.assertEqual(record["model"], "mock-model")
        self.assertEqual(record["prompt_template_version"], "v1")
        self.assertEqual(record["temperature"], 0.2)
        self.assertEqual(record["top_p"], 0.9)
        self.assertEqual(record["k"], 3)
        self.assertEqual(record["input_tokens"], 12)
        self.assertEqual(record["output_tokens"], 9)
        self.assertEqual(record["raw_output"], client.raw_output)
        self.assertEqual(record["parse_failures"], 0)
        self.assertEqual(record["parsed_action_count"], 1)

    def test_adapter_parses_game24_actions_from_actions_object(self) -> None:
        client = MockLLMClient(raw_output='{"actions":[{"action_type":"game24_op","left":10,"right":4,"operator":"-"}]}')
        adapter = RealLLMProposerAdapter(client=client, model="mock-model", prompt_template_version="game24-v1")
        actions = adapter.propose(task={}, node={}, k=2)
        self.assertEqual(actions, [Game24Action(left=10, right=4, operator="-")])
        self.assertEqual(adapter.call_records[-1]["parse_failures"], 0)

    def test_adapter_parses_scheduling_actions_from_actions_object(self) -> None:
        client = MockLLMClient(raw_output='{"actions":[{"action_type":"schedule","job":"J1","machine":"M0"}]}')
        adapter = RealLLMProposerAdapter(client=client, model="mock-model", prompt_template_version="scheduling-v1")
        actions = adapter.propose(task={}, node={}, k=2)
        self.assertEqual(actions, [ScheduleAction(job="J1", machine="M0")])
        self.assertEqual(adapter.call_records[-1]["parse_failures"], 0)

    def test_adapter_records_parse_failure_without_network(self) -> None:
        client = MockLLMClient(raw_output="not json", input_tokens=4, output_tokens=2, latency=0.0)
        adapter = RealLLMProposerAdapter(client=client, model="mock-model", prompt_template_version="v1")
        actions = adapter.propose(task={}, node={}, k=2)
        self.assertEqual(actions, [])
        self.assertEqual(adapter.call_records[-1]["parse_failures"], 1)

    def test_adapter_charges_budget_for_success_and_parse_failure(self) -> None:
        budget = BudgetState(proposal_call_budget=2, generated_token_budget=100)
        ok = RealLLMProposerAdapter(
            client=MockLLMClient(raw_output='[{"action_type":"assign","task":"T1","resource":"R2"}]', input_tokens=5, output_tokens=7),
            model="mock",
            prompt_template_version="v1",
        )
        bad = RealLLMProposerAdapter(
            client=MockLLMClient(raw_output='not json', input_tokens=3, output_tokens=2),
            model="mock",
            prompt_template_version="v1",
        )
        ok.propose(task={}, node={}, k=1, budget_state=budget)
        bad.propose(task={}, node={}, k=1, budget_state=budget)
        self.assertEqual(budget.cost.proposal_calls, 2)
        self.assertEqual(budget.cost.input_tokens, 8)
        self.assertEqual(budget.cost.output_tokens, 9)

    def test_env_configured_client_reads_credentials_only_from_environment(self) -> None:
        env = {"CGR_LLM_API_KEY": "secret", "CGR_LLM_BASE_URL": "https://example.invalid/v1"}
        client = EnvConfiguredLLMClient.from_env(env, provider="openai-compatible")
        self.assertEqual(client.provider, "openai-compatible")
        self.assertEqual(client.base_url, "https://example.invalid/v1")
        self.assertNotIn("secret", client.to_safe_metadata().values())
        with self.assertRaises(RuntimeError):
            client.generate_typed_actions(task={}, node={}, k=1)

    def test_openai_compatible_http_client_builds_request_and_extracts_usage(self) -> None:
        requests = []

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return json.dumps(
                    {
                        "choices": [{"message": {"content": '[{"action_type":"assign","task":"T1","resource":"R2"}]'}}],
                        "usage": {"prompt_tokens": 13, "completion_tokens": 7},
                    }
                ).encode("utf-8")

        def fake_urlopen(request, timeout):
            requests.append((request, timeout))
            return FakeResponse()

        client = OpenAICompatibleHTTPClient(
            base_url="https://llm.example/v1",
            api_key="secret-key",
            model="test-model",
            urlopen=fake_urlopen,
        )
        response = client.generate_typed_actions(task={"id": "task"}, node={"id": "node"}, k=2)
        self.assertEqual(response["raw_output"], '[{"action_type":"assign","task":"T1","resource":"R2"}]')
        self.assertEqual(response["input_tokens"], 13)
        self.assertEqual(response["output_tokens"], 7)
        request = requests[0][0]
        self.assertEqual(request.full_url, "https://llm.example/v1/chat/completions")
        self.assertEqual(request.headers["Authorization"], "Bearer secret-key")
        body = json.loads(request.data.decode("utf-8"))
        self.assertEqual(body["model"], "test-model")
        self.assertEqual(body["response_format"], {"type": "json_object"})
        self.assertIn("actions", body["messages"][1]["content"])

    def test_openai_compatible_http_client_uses_state_local_game24_prompt(self) -> None:
        requests = []

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return json.dumps(
                    {
                        "choices": [{"message": {"content": '{"actions":[]}'}}],
                        "usage": {"prompt_tokens": 13, "completion_tokens": 7},
                    }
                ).encode("utf-8")

        def fake_urlopen(request, timeout):
            requests.append((request, timeout))
            return FakeResponse()

        task = Game24Task("game24-test", (6, 7, 10, 13))
        child = task.apply(task.root_node(), Game24Action(left=10, right=13, operator="+"))
        client = OpenAICompatibleHTTPClient(
            base_url="https://llm.example/v1",
            api_key="secret-key",
            model="test-model",
            urlopen=fake_urlopen,
        )
        client.generate_typed_actions(task=task, node=child, k=4)
        body = json.loads(requests[0][0].data.decode("utf-8"))
        prompt = json.loads(body["messages"][1]["content"])
        self.assertEqual(prompt["current_values"], [6.0, 7.0, 23.0])
        self.assertNotIn("semantic_features", prompt)
        self.assertIn("Do not use numbers from prior states", prompt["instruction"])

    def test_openai_compatible_http_client_uses_state_local_scheduling_prompt(self) -> None:
        requests = []

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return json.dumps(
                    {
                        "choices": [{"message": {"content": '{"actions":[]}'}}],
                        "usage": {"prompt_tokens": 13, "completion_tokens": 7},
                    }
                ).encode("utf-8")

        def fake_urlopen(request, timeout):
            requests.append((request, timeout))
            return FakeResponse()

        task = generate_scheduling_fixture(random.Random(25), "sched-prompt", job_count=4, machine_count=2)
        client = OpenAICompatibleHTTPClient(
            base_url="https://llm.example/v1",
            api_key="secret-key",
            model="test-model",
            urlopen=fake_urlopen,
        )
        client.generate_typed_actions(task=task, node=task.root_node(), k=4)
        body = json.loads(requests[0][0].data.decode("utf-8"))
        prompt = json.loads(body["messages"][1]["content"])
        self.assertEqual(prompt["task"], "precedence_constrained_scheduling")
        self.assertIn("eligible_jobs", prompt)
        self.assertEqual(prompt["response_schema"]["actions"][0]["action_type"], "schedule")
        self.assertNotIn("best_descendant_objective", prompt)

    def test_openai_compatible_http_client_scores_state(self) -> None:
        requests = []

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return json.dumps(
                    {
                        "choices": [{"message": {"content": '{"score":0.75}'}}],
                        "usage": {"prompt_tokens": 11, "completion_tokens": 3},
                    }
                ).encode("utf-8")

        def fake_urlopen(request, timeout):
            requests.append((request, timeout))
            return FakeResponse()

        client = OpenAICompatibleHTTPClient(
            base_url="https://llm.example/v1",
            api_key="secret-key",
            model="test-model",
            urlopen=fake_urlopen,
        )
        response = client.score_state(node={"values": [4, 5, 6, 10]}, model="test-model")
        self.assertEqual(response["score"], 0.75)
        self.assertEqual(response["input_tokens"], 11)
        self.assertEqual(response["output_tokens"], 3)
        body = json.loads(requests[0][0].data.decode("utf-8"))
        self.assertIn("score", body["messages"][1]["content"])

    def test_openai_compatible_http_client_scores_scheduling_state_with_visible_features_only(self) -> None:
        requests = []

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return json.dumps(
                    {
                        "choices": [{"message": {"content": '{"score":0.25}'}}],
                        "usage": {"prompt_tokens": 19, "completion_tokens": 3},
                    }
                ).encode("utf-8")

        def fake_urlopen(request, timeout):
            requests.append((request, timeout))
            return FakeResponse()

        task = generate_scheduling_fixture(random.Random(26), "sched-score", job_count=4, machine_count=2)
        client = OpenAICompatibleHTTPClient(
            base_url="https://llm.example/v1",
            api_key="secret-key",
            model="test-model",
            urlopen=fake_urlopen,
        )
        response = client.score_state(task=task, node=task.root_node(), model="test-model")
        self.assertEqual(response["score"], 0.25)
        self.assertEqual(response["input_tokens"], 19)
        body = json.loads(requests[0][0].data.decode("utf-8"))
        prompt = json.loads(body["messages"][1]["content"])
        self.assertEqual(prompt["task"], "precedence_constrained_scheduling_state_value")
        self.assertIn("eligible_jobs", prompt)
        self.assertNotIn("best_descendant_objective", prompt)

    def test_openai_compatible_http_client_scores_scheduling_frontier_ranking(self) -> None:
        requests = []

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return json.dumps(
                    {
                        "choices": [{"message": {"content": '{"ranking":["b","a"]}'}}],
                        "usage": {"prompt_tokens": 31, "completion_tokens": 5},
                    }
                ).encode("utf-8")

        def fake_urlopen(request, timeout):
            requests.append((request, timeout))
            return FakeResponse()

        task = generate_scheduling_fixture(random.Random(27), "sched-frontier", job_count=4, machine_count=2)
        root = task.root_node()
        a = root.with_updates(node_id="a")
        b = root.with_updates(node_id="b")
        client = OpenAICompatibleHTTPClient(
            base_url="https://llm.example/v1",
            api_key="secret-key",
            model="test-model",
            urlopen=fake_urlopen,
        )
        response = client.score_frontier_states(task=task, nodes=[a, b], model="test-model")
        self.assertEqual(response["scores"], [1.0, 2.0])
        self.assertEqual(response["input_tokens"], 31)
        body = json.loads(requests[0][0].data.decode("utf-8"))
        prompt = json.loads(body["messages"][1]["content"])
        self.assertEqual(prompt["task"], "precedence_constrained_scheduling_frontier_ranking")
        self.assertEqual(len(prompt["candidates"]), 2)
        self.assertNotIn("best_descendant_objective", prompt)

    def test_openai_compatible_http_client_generates_complete_scheduling_solution(self) -> None:
        requests = []

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return json.dumps(
                    {
                        "choices": [{"message": {"content": '{"actions":[{"action_type":"schedule","job":"J0","machine":"M0"}]}'}}],
                        "usage": {"prompt_tokens": 41, "completion_tokens": 9},
                    }
                ).encode("utf-8")

        def fake_urlopen(request, timeout):
            requests.append((request, timeout))
            return FakeResponse()

        task = generate_scheduling_fixture(random.Random(28), "sched-native", job_count=4, machine_count=2)
        client = OpenAICompatibleHTTPClient(
            base_url="https://llm.example/v1",
            api_key="secret-key",
            model="test-model",
            urlopen=fake_urlopen,
        )
        response = client.generate_scheduling_solution(task=task, model="test-model", method="direct", sample_index=0)
        self.assertEqual(response["input_tokens"], 41)
        body = json.loads(requests[0][0].data.decode("utf-8"))
        prompt = json.loads(body["messages"][1]["content"])
        self.assertEqual(prompt["task"], "precedence_constrained_scheduling_complete_solution")
        self.assertEqual(prompt["objective"], "minimize makespan + total tardiness")
        self.assertIn("reasoning", prompt["response_schema"])
        self.assertIn("Think step by step", prompt["instruction"])
        self.assertNotIn("best_descendant_objective", prompt)

    def test_openai_compatible_http_client_retries_timeout(self) -> None:
        calls = {"count": 0}

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return json.dumps(
                    {
                        "choices": [{"message": {"content": '{"actions":[]}'}}],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                    }
                ).encode("utf-8")

        def flaky_urlopen(request, timeout):
            del request, timeout
            calls["count"] += 1
            if calls["count"] == 1:
                raise TimeoutError("simulated timeout")
            return FakeResponse()

        client = OpenAICompatibleHTTPClient(
            base_url="https://llm.example/v1",
            api_key="secret-key",
            model="test-model",
            urlopen=flaky_urlopen,
            max_retries=2,
        )
        response = client.generate_typed_actions(task={}, node={}, k=1)
        self.assertEqual(response["raw_output"], '{"actions":[]}')
        self.assertEqual(calls["count"], 2)

    def test_openai_compatible_http_client_retries_remote_disconnect(self) -> None:
        calls = {"count": 0}

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return json.dumps(
                    {
                        "choices": [{"message": {"content": '{"score":0.25}'}}],
                        "usage": {"prompt_tokens": 2, "completion_tokens": 1},
                    }
                ).encode("utf-8")

        def flaky_urlopen(request, timeout):
            del request, timeout
            calls["count"] += 1
            if calls["count"] == 1:
                raise http.client.RemoteDisconnected("simulated disconnect")
            return FakeResponse()

        client = OpenAICompatibleHTTPClient(
            base_url="https://llm.example/v1",
            api_key="secret-key",
            model="test-model",
            urlopen=flaky_urlopen,
            max_retries=2,
        )
        response = client.score_state(node={"values": [24]}, model="test-model")
        self.assertEqual(response["score"], 0.25)
        self.assertEqual(calls["count"], 2)


if __name__ == "__main__":
    unittest.main()
