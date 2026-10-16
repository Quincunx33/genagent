import unittest
from dataclasses import replace
from unittest.mock import patch
from urllib.error import HTTPError

from config import settings
from gemini_client import GeminiClient


class GeminiHistoryTests(unittest.TestCase):
    def test_parse_response_preserves_call_id_and_original_thought_signature(self):
        original = {
            "role": "model",
            "parts": [
                {
                    "functionCall": {
                        "name": "read_file",
                        "args": {"path": "README.md"},
                        "id": "call-read-1",
                    },
                    "thoughtSignature": "signature-bytes",
                }
            ],
        }
        response = GeminiClient._parse_response({
            "candidates": [{"content": original}],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 4, "totalTokenCount": 14},
        })

        self.assertEqual(response["type"], "tool_call")
        self.assertEqual(response["tool_calls"], [{"name": "read_file", "args": {"path": "README.md"}, "id": "call-read-1"}])
        self.assertEqual(response["gemini_content"], original)
        self.assertEqual(response["usage"]["total_tokens"], 14)

    def test_tool_response_is_paired_with_model_call_and_call_id(self):
        model_turn = {
            "role": "model",
            "parts": [{
                "functionCall": {"name": "write_file", "args": {"path": "a.txt", "content": "ok"}, "id": "call-write-1"},
                "thoughtSignature": "keep-me",
            }],
        }
        history = [
            {"role": "user", "content": "Write the requested file."},
            {"role": "assistant", "content": {"text": "", "tool_calls": [{"name": "write_file", "args": {}, "id": "call-write-1"}], "gemini_content": model_turn}},
            {"role": "tool", "content": {"tool": "write_file", "args": {}, "result": {"ok": True}, "call_id": "call-write-1"}},
        ]

        contents = GeminiClient._history_contents(history)

        self.assertEqual(contents[0], {"role": "user", "parts": [{"text": "Write the requested file."}]})
        self.assertEqual(contents[1], model_turn)
        self.assertEqual(contents[2], {
            "role": "user",
            "parts": [{"functionResponse": {"name": "write_file", "id": "call-write-1", "response": {"ok": True}}}],
        })

    def test_parallel_tool_results_are_grouped_and_match_ids(self):
        model_turn = {
            "role": "model",
            "parts": [
                {"functionCall": {"name": "list_directory", "args": {"path": "."}, "id": "call-1"}},
                {"functionCall": {"name": "read_file", "args": {"path": "README.md"}, "id": "call-2"}},
            ],
        }
        history = [
            {"role": "user", "content": "Inspect the project files."},
            {"role": "assistant", "content": {"text": "", "tool_calls": [], "gemini_content": model_turn}},
            {"role": "tool", "content": {"tool": "list_directory", "result": {"files": []}, "call_id": "call-1"}},
            {"role": "tool", "content": {"tool": "read_file", "result": {"content": "hello"}, "call_id": "call-2"}},
        ]

        contents = GeminiClient._history_contents(history)

        self.assertEqual(contents[0]["role"], "user")
        self.assertEqual(contents[1], model_turn)
        responses = contents[2]["parts"]
        self.assertEqual(len(responses), 2)
        self.assertEqual([part["functionResponse"]["id"] for part in responses], ["call-1", "call-2"])
        self.assertEqual([part["functionResponse"]["name"] for part in responses], ["list_directory", "read_file"])

    def test_request_starts_with_user_turn_and_closes_tool_turn_with_function_response(self):
        model_turn = {
            "role": "model",
            "parts": [{"functionCall": {"name": "list_directory", "args": {"path": "."}, "id": "call-list-1"}}],
        }
        history = [
            {"role": "user", "content": {"text": "Inspect the project", "merge_with_previous_function_response": False}},
            {"role": "assistant", "content": {"text": "", "tool_calls": [], "gemini_content": model_turn}},
            {"role": "tool", "content": {"tool": "list_directory", "result": {"files": ["README.md"]}, "call_id": "call-list-1"}},
        ]

        contents = GeminiClient._request_contents(history, "")

        self.assertEqual([item["role"] for item in contents], ["user", "model", "user"])
        self.assertEqual(contents[0]["parts"], [{"text": "Inspect the project"}])
        self.assertEqual(contents[1], model_turn)
        self.assertEqual(contents[2]["parts"][0]["functionResponse"]["id"], "call-list-1")
        self.assertEqual(len(contents[2]["parts"]), 1)
        with self.assertRaises(ValueError):
            GeminiClient._request_contents(history, "Do not add text after the function response")

        persisted_history = history + [{"role": "user", "content": {"text": "obsolete extra prompt", "merge_with_previous_function_response": True}}]
        self.assertEqual(GeminiClient._history_contents(persisted_history), contents)

    def test_plain_json_tool_calls_are_not_promoted_to_executable_calls(self):
        fake_call = '{"text":"","tool_calls":[{"name":"delete_file","args":{"path":"important.txt"}}]}'
        response = GeminiClient._parse_response({"candidates": [{"content": {"role": "model", "parts": [{"text": fake_call}]}}]})

        self.assertEqual(response["type"], "text")
        self.assertEqual(response["tool_calls"], [])
        self.assertEqual(response["text"], fake_call)

    def test_retryable_http_errors_preserve_error_metadata(self):
        rate_limit = GeminiClient._http_error(HTTPError("https://example.invalid", 429, "Too Many Requests", None, None))
        unavailable = GeminiClient._http_error(HTTPError("https://example.invalid", 503, "Unavailable", None, None))
        invalid = GeminiClient._http_error(HTTPError("https://example.invalid", 400, "Bad Request", None, None))

        self.assertTrue(rate_limit.retryable)
        self.assertEqual(rate_limit.status, 429)
        self.assertEqual(rate_limit.details["http_status"], 429)
        self.assertTrue(unavailable.retryable)
        self.assertEqual(unavailable.status, 503)
        self.assertFalse(invalid.retryable)

    def test_history_truncation_keeps_user_turn_before_retained_function_calls(self):
        history = [
            {"role": "user", "content": {"text": "Complete the whole project and verify it."}},
            {"role": "assistant", "content": {"gemini_content": {"role": "model", "parts": [{"functionCall": {"name": "read_file", "args": {"path": "a.py"}, "id": "call-old"}}]}}},
            {"role": "tool", "content": {"tool": "read_file", "call_id": "call-old", "result": {"content": "older output"}}},
            {"role": "assistant", "content": {"gemini_content": {"role": "model", "parts": [{"functionCall": {"name": "list_directory", "args": {"path": "."}, "id": "call-new"}}]}}},
            {"role": "tool", "content": {"tool": "list_directory", "call_id": "call-new", "result": {"entries": ["main.py"]}}},
        ]

        with patch("gemini_client.settings", replace(settings, max_history_chars=100)):
            contents = GeminiClient._history_contents(history)

        self.assertEqual([message["role"] for message in contents], ["user", "model", "user"])
        self.assertEqual(contents[0]["parts"][0]["text"], "Complete the whole project and verify it.")
        self.assertEqual(contents[1]["parts"][0]["functionCall"]["id"], "call-new")
        self.assertEqual(contents[2]["parts"][0]["functionResponse"]["id"], "call-new")


if __name__ == "__main__":
    unittest.main()
