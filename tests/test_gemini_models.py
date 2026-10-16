import json
import unittest
from unittest.mock import patch

from gemini_client import list_gemini_models


class _Response:
    def __init__(self, data):
        self.data = json.dumps(data).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self.data


class GeminiModelDiscoveryTests(unittest.TestCase):
    @patch("gemini_client.urlopen")
    def test_lists_only_generation_models_across_pages(self, urlopen):
        urlopen.side_effect = [
            _Response({
                "models": [
                    {"name": "models/gemini-a", "displayName": "Gemini A", "supportedGenerationMethods": ["generateContent"], "inputTokenLimit": 100},
                    {"name": "models/embed-a", "supportedGenerationMethods": ["embedContent"]},
                ],
                "nextPageToken": "next-token",
            }),
            _Response({
                "models": [
                    {"baseModelId": "gemini-b", "name": "models/gemini-b-v001", "displayName": "Gemini B", "supportedGenerationMethods": ["generateContent"]}
                ]
            }),
        ]
        result = list_gemini_models("test-key")
        self.assertEqual([item["id"] for item in result], ["gemini-a", "gemini-b"])
        self.assertEqual(result[0]["input_token_limit"], 100)
        self.assertEqual(urlopen.call_count, 2)
        self.assertIn("pageToken=next-token", urlopen.call_args_list[1].args[0].full_url)

    def test_requires_api_key(self):
        with self.assertRaises(ValueError):
            list_gemini_models("")


if __name__ == "__main__":
    unittest.main()
