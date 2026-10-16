from __future__ import annotations

import json
import hashlib
import random
import threading
import time
from urllib.parse import urlencode
from collections import OrderedDict, deque
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from config import settings
from errors import AgentError, ErrorCategory, normalize_exception
from logger import get_logger
from quota import RequestQuota


log = get_logger("genagent.gemini")


class _RequestGate:
    def __init__(self):
        self._lock = threading.Lock()
        self._events: deque[float] = deque()
        self._last = 0.0

    def acquire(self, cancel_event=None) -> bool:
        while True:
            now = time.monotonic()
            with self._lock:
                window = max(1, settings.gemini_rate_window)
                while self._events and now - self._events[0] >= window:
                    self._events.popleft()
                interval_wait = max(0.0, settings.gemini_min_interval - (now - self._last))
                quota_wait = max(0.0, window - (now - self._events[0])) if len(self._events) >= settings.gemini_rate_limit else 0.0
                wait = max(interval_wait, quota_wait)
                if wait <= 0:
                    self._last = now
                    self._events.append(now)
                    return True
            if cancel_event is not None and cancel_event.is_set():
                return False
            time.sleep(min(wait, 0.25))


_REQUEST_GATE = _RequestGate()
_PERSISTENT_QUOTA = RequestQuota(settings.db_path)


def list_gemini_models(api_key: str, *, timeout: float = 10.0) -> list[dict[str, Any]]:
    """Return models accessible to this API key that support generateContent."""
    if not api_key:
        raise ValueError("Gemini API key is required")
    models: dict[str, dict[str, Any]] = {}
    page_token = ""
    while True:
        params = {"key": api_key, "pageSize": 1000}
        if page_token:
            params["pageToken"] = page_token
        request = Request(
            "https://generativelanguage.googleapis.com/v1beta/models?" + urlencode(params),
            headers={"Accept": "application/json"},
        )
        with urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        for item in data.get("models", []):
            if not isinstance(item, dict) or "generateContent" not in item.get("supportedGenerationMethods", []):
                continue
            name = item.get("baseModelId") or item.get("name", "").removeprefix("models/")
            if not name:
                continue
            models[name] = {
                "id": name,
                "display_name": item.get("displayName") or name,
                "description": item.get("description", ""),
                "input_token_limit": item.get("inputTokenLimit"),
                "output_token_limit": item.get("outputTokenLimit"),
            }
        page_token = data.get("nextPageToken", "")
        if not page_token:
            break
    return sorted(models.values(), key=lambda item: (item["display_name"].lower(), item["id"]))


class GeminiClient:
    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or settings.gemini_api_key
        self.model = model or settings.gemini_model
        self.models = tuple(dict.fromkeys((self.model, *settings.gemini_fallback_models)))
        self._cache: OrderedDict[str, tuple[float, dict[str, Any]]] = OrderedDict()
        self._cache_lock = threading.Lock()
        self._model_failures: dict[str, tuple[int, float]] = {}
        self.session_tokens: dict[str, int] = {"prompt": 0, "candidates": 0, "total": 0}

    @staticmethod
    def _cache_key(model: str, payload: dict[str, Any]) -> str:
        encoded = json.dumps({"model": model, "payload": payload}, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def _cached(self, key: str) -> dict[str, Any] | None:
        if not settings.response_cache_enabled:
            return None
        now = time.time()
        with self._cache_lock:
            item = self._cache.get(key)
            if not item:
                return None
            created, value = item
            if now - created >= settings.response_cache_ttl:
                self._cache.pop(key, None)
                return None
            self._cache.move_to_end(key)
            return json.loads(json.dumps(value, ensure_ascii=False))

    def _store_cache(self, key: str, value: dict[str, Any]) -> None:
        if not settings.response_cache_enabled or value.get("type") != "text" or value.get("tool_calls"):
            return
        with self._cache_lock:
            self._cache[key] = (time.time(), json.loads(json.dumps(value, ensure_ascii=False)))
            self._cache.move_to_end(key)
            while len(self._cache) > settings.response_cache_size:
                self._cache.popitem(last=False)

    def available(self) -> bool:
        return bool(self.api_key)

    def set_task_route(self, task: str) -> None:
        self.models = tuple(dict.fromkeys((self.model, *settings.gemini_fallback_models)))

    @staticmethod
    def _history_contents(history: list | None) -> list[dict[str, Any]]:
        turns: list[list[dict[str, Any]]] = []
        items = history or []
        index = 0

        def copy_json(value: Any) -> Any:
            return json.loads(json.dumps(value, ensure_ascii=False, default=str))

        def response_part(content: Any) -> dict[str, Any] | None:
            if not isinstance(content, dict):
                return None
            name = content.get("tool")
            if not isinstance(name, str) or not name:
                return None
            response = content.get("result")
            if isinstance(response, (dict, list, str, int, float, bool)) or response is None:
                response_value = response if isinstance(response, dict) else {"result": response}
            else:
                response_value = {"result": str(response)}
            encoded = json.dumps(response_value, ensure_ascii=False, default=str)
            limit = settings.max_tool_result_chars
            if len(encoded) > limit:
                response_value = {"truncated": True, "preview": encoded[:max(0, limit - 40)]}
            function_response = {"name": name, "response": response_value}
            call_id = content.get("call_id") or content.get("id")
            if call_id is not None:
                function_response["id"] = str(call_id)
            return {"functionResponse": function_response}

        while index < len(items):
            item = items[index]
            if not isinstance(item, dict):
                turns.append([{"role": "user", "parts": [{"text": str(item)[:settings.max_tool_result_chars]}]}])
                index += 1
                continue
            role = item.get("role", "user")
            value = item.get("content", "")

            if role == "assistant":
                raw_content = value.get("gemini_content") if isinstance(value, dict) else None
                if isinstance(raw_content, dict) and isinstance(raw_content.get("parts"), list):
                    model_turn = copy_json(raw_content)
                    model_turn["role"] = "model"
                else:
                    text = value.get("text", "") if isinstance(value, dict) else str(value)
                    calls = value.get("tool_calls", []) if isinstance(value, dict) else []
                    parts = ([{"text": str(text)}] if text else [])
                    for call in calls if isinstance(calls, list) else []:
                        if not isinstance(call, dict) or not call.get("name"):
                            continue
                        function_call = {"name": str(call["name"]), "args": call.get("args", {}) if isinstance(call.get("args", {}), dict) else {}}
                        if call.get("id") is not None:
                            function_call["id"] = str(call["id"])
                        parts.append({"functionCall": function_call})
                    if not parts:
                        parts = [{"text": ""}]
                    model_turn = {"role": "model", "parts": parts}

                group = [model_turn]
                index += 1
                response_parts = []
                while index < len(items) and isinstance(items[index], dict) and items[index].get("role") == "tool":
                    part = response_part(items[index].get("content"))
                    if part:
                        response_parts.append(part)
                    index += 1
                if response_parts:
                    group.append({"role": "user", "parts": response_parts})
                turns.append(group)
                continue

            if role == "tool":
                response_parts = []
                while index < len(items) and isinstance(items[index], dict) and items[index].get("role") == "tool":
                    part = response_part(items[index].get("content"))
                    if part:
                        response_parts.append(part)
                    index += 1
                if response_parts:
                    turns.append([{"role": "user", "parts": response_parts}])
                continue

            if role == "user" and isinstance(value, dict) and "text" in value:
                text = value.get("text", "")
                if value.get("merge_with_previous_function_response"):
                    # Older saved tasks may contain a continuation prompt appended after a
                    # function response. Gemini requires the function response to close the turn.
                    index += 1
                    continue
            else:
                text = value
            if isinstance(text, (dict, list)):
                text = json.dumps(text, ensure_ascii=False, separators=(",", ":"), default=str)
            turns.append([{"role": "user", "parts": [{"text": str(text)[:settings.max_tool_result_chars]}]}])
            index += 1

        selected: list[list[dict[str, Any]]] = []
        used = 0
        budget = settings.max_history_chars
        for group in reversed(turns):
            cost = sum(len(json.dumps(message, ensure_ascii=False, separators=(",", ":"), default=str)) for message in group)
            if selected and used + cost > budget:
                break
            selected.append(group)
            used += cost
        chronological = list(reversed(selected))
        if chronological and chronological[0] and chronological[0][0].get("role") == "model":
            opening_user = next((
                group for group in turns
                if group and group[0].get("role") == "user"
                and not any("functionResponse" in part for part in group[0].get("parts", []) if isinstance(part, dict))
            ), None)
            if opening_user is None:
                opening_user = [{"role": "user", "parts": [{"text": "Continue the original task from the retained tool-call and tool-response history; earlier context was trimmed."}]}]
            chronological.insert(0, opening_user)
        return [message for group in chronological for message in group]

    @staticmethod
    def _request_contents(history: list | None, prompt: str) -> list[dict[str, Any]]:
        contents = GeminiClient._history_contents(history)
        if contents and contents[-1].get("role") == "user":
            parts = contents[-1].get("parts", [])
            if any(isinstance(part, dict) and "functionResponse" in part for part in parts):
                if prompt.strip():
                    raise ValueError("Do not append a new user prompt after a Gemini functionResponse turn")
                return contents
        if not prompt.strip():
            raise ValueError("An empty Gemini prompt is valid only after a functionResponse turn")
        text_part = {"text": prompt[:settings.max_prompt_chars]}
        contents.append({"role": "user", "parts": [text_part]})
        return contents

    @staticmethod
    def _http_error(exc: HTTPError) -> AgentError:
        code = getattr(exc, "code", 0)
        if code == 429:
            return AgentError(
                "API_RATE_LIMIT", "Gemini HTTP 429 (Rate limit / Too many requests)",
                "AI API rate limit reached (too many requests). Please wait a moment before trying again.",
                category=ErrorCategory.RATE_LIMIT, retryable=True, status=429,
                details={"http_status": code},
            )
        if code in {408, 425} or code >= 500:
            return AgentError(
                "API_HTTP_ERROR", f"Gemini HTTP {code}", "The AI service is temporarily unavailable.",
                category=ErrorCategory.NETWORK, retryable=True, status=503,
                details={"http_status": code},
            )
        if code in {401, 403}:
            return AgentError(
                "API_AUTH_ERROR", f"Gemini authentication failed (HTTP {code})",
                "The AI API key was rejected. Check your configuration.",
                category=ErrorCategory.SYSTEM, retryable=False, status=502,
                details={"http_status": code},
            )
        return AgentError(
            "API_REQUEST_ERROR", f"Gemini rejected the request (HTTP {code})",
            "The AI service rejected the request.", category=ErrorCategory.VALIDATION,
            retryable=False, status=502, details={"http_status": code},
        )

    @staticmethod
    def _parse_response(data: Any) -> dict[str, Any]:
        if not isinstance(data, dict):
            raise ValueError("Gemini response must be a JSON object")
        candidates = data.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            api_error = data.get("error")
            raise ValueError(f"Gemini response has no candidates: {api_error or 'unknown response'}")
        content = candidates[0].get("content", {})
        parts = content.get("parts", []) if isinstance(content, dict) else []
        if not isinstance(parts, list):
            raise ValueError("Gemini response has invalid content parts")
        text_parts = [p.get("text", "") for p in parts if isinstance(p, dict) and p.get("text")]
        calls = []
        for part in parts:
            call = part.get("functionCall") if isinstance(part, dict) else None
            if isinstance(call, dict) and call.get("name"):
                args = call.get("args", {})
                parsed_call = {"name": str(call["name"]), "args": args if isinstance(args, dict) else {}}
                if call.get("id") is not None:
                    parsed_call["id"] = str(call["id"])
                calls.append(parsed_call)
        text = "".join(text_parts)
        model_content = json.loads(json.dumps(content, ensure_ascii=False, default=str)) if isinstance(content, dict) else {"parts": parts}
        model_content["role"] = "model"
        usage = data.get("usageMetadata", {})
        return {
            "type": "tool_call" if calls else "text",
            "text": text,
            "tool_calls": calls,
            "gemini_content": model_content,
            "usage": {
                "prompt_tokens": usage.get("promptTokenCount", 0),
                "candidates_tokens": usage.get("candidatesTokenCount", 0),
                "total_tokens": usage.get("totalTokenCount", 0),
            }
        }

    def generate(self, prompt: str, tools: list[dict] | None = None, history: list | None = None, cancel_event=None) -> dict[str, Any]:
        if not self.api_key:
            return {"type": "error", "text": "Gemini is not configured. Run 'python setup.py'.", "tool_calls": [], "error": AgentError("NOT_CONFIGURED", "Gemini API key is missing", "Gemini is not configured.", category=ErrorCategory.SYSTEM, retryable=False, status=503).to_dict()["error"]}
        if not isinstance(prompt, str) or (not prompt.strip() and not any(isinstance(item, dict) and item.get("role") == "tool" for item in (history or []))):
            raise AgentError(
                "INVALID_REQUEST", "Prompt must be non-empty unless continuing from a function response",
                "The agent prompt was empty.", category=ErrorCategory.VALIDATION,
                retryable=False, status=400,
            )

        payload: dict[str, Any] = {
            "systemInstruction": {"parts": [{"text": "You are an autonomous, highly capable computer assistant. You have full terminal and filesystem autonomy via core tools (run_command, read_file, write_file, patch_file, inspect_image). Take initiative, make your own decisions, and solve tasks directly using bash, Python, Git, and package managers without needing rigid hardcoded tools."}]},
            "contents": self._request_contents(history, prompt),
        }
        if tools:
            payload["tools"] = [{"functionDeclarations": tools}]

        last_error: AgentError | None = None
        attempts = max(1, min(settings.api_retries, 10))
        now = time.time()
        healthy_models = [model for model in self.models if now >= self._model_failures.get(model, (0, 0))[1]] or list(self.models)
        for model in healthy_models:
            cache_key = self._cache_key(model, payload)
            cached = self._cached(cache_key)
            if cached is not None:
                log.debug("Gemini response cache hit model=%s", model)
                self.model = model
                return cached
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.api_key}"
            for attempt in range(attempts):
                if cancel_event is not None and cancel_event.is_set():
                    return {"type": "cancelled", "text": "Gemini request cancelled.", "tool_calls": [], "error": {"code": "CANCELLED"}}
                if not _PERSISTENT_QUOTA.allow(settings.gemini_persistent_limit, settings.gemini_persistent_window):
                    return {"type": "error", "text": "Gemini request budget reached. Please wait before starting more work.", "tool_calls": [], "error": {"code": "API_QUOTA_EXCEEDED"}}
                if not _REQUEST_GATE.acquire(cancel_event):
                    return {"type": "cancelled", "text": "Gemini request cancelled.", "tool_calls": [], "error": {"code": "CANCELLED"}}
                request = Request(url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json", "Accept": "application/json"}, method="POST")
                retry_after = None
                try:
                    with urlopen(request, timeout=settings.command_timeout) as response:
                        try:
                            raw = response.read(settings.max_api_response_bytes)
                        except TypeError:
                            raw = response.read()
                        if len(raw) > settings.max_api_response_bytes:
                            raise ValueError("Gemini response exceeds configured size limit")
                    data = json.loads(raw.decode("utf-8"))
                    result = self._parse_response(data)
                    self.model = model
                    self._model_failures.pop(model, None)
                    self._store_cache(cache_key, result)
                    if isinstance(result, dict) and "usage" in result:
                        u = result["usage"]
                        self.session_tokens["prompt"] += u.get("prompt_tokens", 0)
                        self.session_tokens["candidates"] += u.get("candidates_tokens", 0)
                        self.session_tokens["total"] += u.get("total_tokens", 0)
                    return result
                except HTTPError as exc:
                    err = self._http_error(exc)
                    try:
                        retry_after = float(exc.headers.get("Retry-After", "")) if exc.headers else None
                    except (TypeError, ValueError):
                        retry_after = None
                except (URLError, TimeoutError, OSError, json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
                    err = normalize_exception(exc, operation="Gemini API request")
                last_error = err
                if err.status == 429:
                    failures, _ = self._model_failures.get(model, (0, 0))
                    self._model_failures[model] = (failures + 1, time.time() + 45.0)
                    log.warning("Gemini model=%s rate limited (429), switching immediately to fallback model", model)
                    break

                if err.retryable:
                    failures, _ = self._model_failures.get(model, (0, 0))
                    self._model_failures[model] = (failures + 1, time.time() + min(settings.gemini_backoff_max, settings.gemini_backoff_base * (2 ** min(failures, 5))))
                log.warning("Gemini request failed model=%s attempt=%s/%s code=%s http_status=%s", model, attempt + 1, attempts, err.code, err.details.get("http_status", err.status))
                if not err.retryable:
                    break
                if attempt + 1 < attempts:
                    base_delay = 2.0 if err.status == 429 else settings.gemini_backoff_base
                    delay = retry_after if retry_after is not None else min(settings.gemini_backoff_max, base_delay * (2 ** attempt))
                    end = time.monotonic() + delay + random.uniform(0, 0.25)
                    while time.monotonic() < end:
                        if cancel_event is not None and cancel_event.is_set():
                            return {"type": "cancelled", "text": "Gemini request cancelled.", "tool_calls": [], "error": {"code": "CANCELLED"}}
                        time.sleep(min(0.25, max(0.01, end - time.monotonic())))

        err = last_error or AgentError(
            "API_UNAVAILABLE", "No Gemini model was available", "The AI service is unavailable.",
            category=ErrorCategory.NETWORK, retryable=True, status=503,
        )
        return {"type": "error", "text": f"{err.public_message} (error_id={err.error_id})", "tool_calls": [], "error": err.to_dict()["error"]}
