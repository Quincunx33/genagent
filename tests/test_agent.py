import unittest
from unittest.mock import patch
import json
from agent_loop import AgentLoop

class FakeClient:
    def __init__(self):
        self.calls = 0

    def generate(self, *args, **kwargs):
        self.calls += 1
        return {"type": "text", "text": "done", "tool_calls": []}

class AgentTests(unittest.TestCase):
    def test_loop(self):
        self.assertEqual(AgentLoop(FakeClient(), output=lambda _value: None).run("hello"), "done")

    def test_read_file_result_is_redacted_before_model_context(self):
        loop = AgentLoop(FakeClient(), output=lambda _value: None)
        with patch("agent_loop.read_file", return_value="GEMINI_API_KEY=super-secret\nvisible=value"):
            result = loop.execute("read_file", {"path": "config.txt"})
        self.assertNotIn("super-secret", result["content"])
        self.assertIn("[REDACTED]", result["content"])

    def test_terminate_process_does_not_double_prompt(self):
        loop = AgentLoop(FakeClient(), output=lambda _value: None)
        with patch("builtins.input", return_value="y") as prompt, patch("agent_loop.terminate_process", return_value={"ok": True}) as terminate:
            loop.execute("terminate_process", {"pid": 1234})
        prompt.assert_called_once()
        terminate.assert_called_once_with(1234, approved=True)

    def test_circuit_breaker_stops_infinite_loop(self):
        class LoopingClient:
            def __init__(self):
                self.count = 0
            def generate(self, prompt, tools=None, history=None, cancel_event=None):
                self.count += 1
                return {"type": "tool_call", "text": "", "tool_calls": [{"name": "list_directory", "args": {"path": "workflow"}}]}

        client = LoopingClient()
        loop = AgentLoop(client, output=lambda _value: None)
        res = loop.run("test loop breaker")
        self.assertIn("Action loop detected", res)
        self.assertLessEqual(client.count, 5)

    def test_autonomous_conversational_response(self):
        class DirectTextClient:
            def generate(self, prompt, tools=None, history=None, cancel_event=None):
                self.prompt = prompt
                return {"type": "text", "text": "Hello! How can I help you?", "usage": {"prompt_tokens": 450, "candidates_tokens": 20, "total_tokens": 470}}

        client = DirectTextClient()
        loop = AgentLoop(client, output=lambda _val: None)
        reply = loop.run("hi")
        self.assertEqual(reply, "Hello! How can I help you?")
        self.assertIn("Tool Usage Rule", client.prompt)
        self.assertEqual(loop.last_task_tokens["total"], 470)

    def test_plain_json_tool_request_is_not_executed_or_marked_complete(self):
        class JsonToolClient:
            def __init__(self):
                self.calls = 0
            def generate(self, prompt, tools=None, history=None, cancel_event=None):
                self.calls += 1
                return {"type": "text", "text": json.dumps({"text": "", "tool_calls": [{"name": "write_file", "args": {"path": "marker.txt", "content": "made"}}]}), "tool_calls": []}

        class RecordingStore:
            def __init__(self):
                self.status = None
            def create(self, task):
                return "task-test"
            def save(self, task_id, task, status, step, history):
                self.status = status

        class MemoryStub:
            def context(self, _task):
                return ""
            def save(self, *_args, **_kwargs):
                pass

        client = JsonToolClient()
        store = RecordingStore()
        loop = AgentLoop(client, output=lambda _value: None, store=store, memory=MemoryStub())
        with patch.object(loop, "execute") as execute:
            result = loop.run("Create marker.txt with the word made and verify it.")

        self.assertIn("Task not completed", result)
        self.assertEqual(store.status, "failed")
        self.assertEqual(client.calls, 2)
        execute.assert_not_called()

    def test_retryable_model_error_pauses_task_without_fake_assistant_turn(self):
        class UnavailableClient:
            def generate(self, prompt, tools=None, history=None, cancel_event=None):
                return {"type": "error", "text": "Gemini temporarily unavailable", "tool_calls": [], "error": {"retryable": True}}

        class RecordingStore:
            def __init__(self):
                self.status = None
                self.history = []
            def create(self, task):
                return "task-retry"
            def save(self, task_id, task, status, step, history):
                self.status = status
                self.history = history

        class MemoryStub:
            def context(self, _task):
                return ""
            def save(self, *_args, **_kwargs):
                pass

        store = RecordingStore()
        loop = AgentLoop(UnavailableClient(), output=lambda _value: None, store=store, memory=MemoryStub())
        result = loop.run("Create a small file and verify it")

        self.assertIn("resume with task_id=task-retry", result)
        self.assertEqual(store.status, "paused")
        self.assertEqual([item["role"] for item in store.history], ["user"])

    def test_continuation_after_tool_sends_no_extra_user_prompt(self):
        class TwoTurnClient:
            def __init__(self):
                self.prompts = []
                self.histories = []
            def generate(self, prompt, tools=None, history=None, cancel_event=None):
                self.prompts.append(prompt)
                self.histories.append(list(history or []))
                if len(self.prompts) == 1:
                    return {"type": "tool_call", "text": "", "tool_calls": [{"name": "list_directory", "args": {"path": "."}, "id": "call-list"}], "gemini_content": {"role": "model", "parts": [{"functionCall": {"name": "list_directory", "args": {"path": "."}, "id": "call-list"}}]}}
                return {"type": "text", "text": "The directory listing is complete.", "tool_calls": []}

        client = TwoTurnClient()
        loop = AgentLoop(client, output=lambda _value: None)
        with patch.object(loop, "execute", return_value={"entries": []}):
            result = loop.run("List the project directory and report what is inside.")

        self.assertEqual(result, "The directory listing is complete.")
        self.assertTrue(client.prompts[0])
        self.assertEqual(client.prompts[1], "")
        self.assertEqual([item["role"] for item in client.histories[1]], ["user", "assistant", "tool"])

if __name__ == "__main__":
    unittest.main()
