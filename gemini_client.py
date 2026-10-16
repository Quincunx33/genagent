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
from errors import AgentError, normalize_exception
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
        selected: list[Any] = []
        used = 0
        for item in reversed(history or []):
            value = item.get("content", "") if isinstance(item, dict) else str(item)
            if isinstance(value, (dict, list)):
                value = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            value = str(value)[:settings.max_tool_result_chars]
            cost = len(value)
            if selected and used + cost > settings.max_history_chars:
                break
            selected.append((item, value))
            used += cost

        contents = []
        for item, compact_value in reversed(selected):
            role = item.get("role", "user") if isinstance(item, dict) else "user"
            api_role = "model" if role == "assistant" else "user"
            contents.append({"role": api_role, "parts": [{"text": compact_value}]})
        return contents

    @staticmethod
    def _http_error(exc: HTTPError) -> AgentError:
        code = getattr(exc, "code", 0)
        if code == 429:
            return AgentError("API_RATE_LIMIT", f"Gemini HTTP 429 (Rate limit / Too many requests)", "AI API rate limit reached (too many requests). Please wait a moment before trying again.", True, 429, {"http_status": code})
        if code in {408, 425} or code >= 500:
            return AgentError("API_HTTP_ERROR", f"Gemini HTTP {code}", "The AI service is temporarily unavailable.", True, 503, {"http_status": code})
        if code in {401, 403}:
            return AgentError("API_AUTH_ERROR", f"Gemini authentication failed (HTTP {code})", "The AI API key was rejected. Check your configuration.", False, 502, {"http_status": code})
        return AgentError("API_REQUEST_ERROR", f"Gemini rejected the request (HTTP {code})", "The AI service rejected the request.", False, 502, {"http_status": code})

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
                calls.append({"name": str(call["name"]), "args": args if isinstance(args, dict) else {}})
        text = "".join(text_parts)
        if not calls and text.lstrip().startswith("{"):
            try:
                encoded, _ = json.JSONDecoder().raw_decode(text.lstrip())
                encoded_calls = encoded.get("tool_calls", []) if isinstance(encoded, dict) else []
                if isinstance(encoded_calls, list) and all(isinstance(item, dict) and item.get("name") for item in encoded_calls):
                    calls = [{"name": str(item["name"]), "args": item.get("args", {}) if isinstance(item.get("args", {}), dict) else {}} for item in encoded_calls]
                    text = str(encoded.get("text", ""))
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
        usage = data.get("usageMetadata", {})
        return {
            "type": "tool_call" if calls else "text",
            "text": text,
            "tool_calls": calls,
            "usage": {
                "prompt_tokens": usage.get("promptTokenCount", 0),
                "candidates_tokens": usage.get("candidatesTokenCount", 0),
                "total_tokens": usage.get("totalTokenCount", 0),
            }
        }

    def generate(self, prompt: str, tools: list[dict] | None = None, history: list | None = None, cancel_event=None) -> dict[str, Any]:
        if not self.api_key:
            return {"type": "error", "text": "Gemini is not configured. Run 'python setup.py'.", "tool_calls": [], "error": AgentError("NOT_CONFIGURED", "Gemini API key is missing", "Gemini is not configured.", False, 503).to_dict()["error"]}
        if not isinstance(prompt, str) or not prompt.strip():
            raise AgentError("INVALID_REQUEST", "Prompt must be non-empty", "The agent prompt was empty.", False, 400)

        payload: dict[str, Any] = {
            "systemInstruction": {"parts": [{"text": "You are an autonomous, highly capable computer assistant. You have full terminal and filesystem autonomy via core tools (run_command, read_file, write_file, patch_file, inspect_image). Take initiative, make your own decisions, and solve tasks directly using bash, Python, Git, and package managers without needing rigid hardcoded tools."}]},
            "contents": self._history_contents(history) + [{"role": "user", "parts": [{"text": prompt[:settings.max_prompt_chars]}]}],
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
                if getattr(err, "http_status", 0) == 429:
                    failures, _ = self._model_failures.get(model, (0, 0))
                    self._model_failures[model] = (failures + 1, time.time() + 45.0)
                    log.warning("Gemini model=%s rate limited (429), switching immediately to fallback model", model)
                    break

                if err.retryable:
                    failures, _ = self._model_failures.get(model, (0, 0))
                    self._model_failures[model] = (failures + 1, time.time() + min(settings.gemini_backoff_max, settings.gemini_backoff_base * (2 ** min(failures, 5))))
                log.warning("Gemini request failed model=%s attempt=%s/%s code=%s", model, attempt + 1, attempts, err.code)
                if not err.retryable:
                    break
                if attempt + 1 < attempts:
                    base_delay = 2.0 if getattr(err, "http_status", 0) == 429 else settings.gemini_backoff_base
                    delay = retry_after if retry_after is not None else min(settings.gemini_backoff_max, base_delay * (2 ** attempt))
                    end = time.monotonic() + delay + random.uniform(0, 0.25)
                    while time.monotonic() < end:
                        if cancel_event is not None and cancel_event.is_set():
                            return {"type": "cancelled", "text": "Gemini request cancelled.", "tool_calls": [], "error": {"code": "CANCELLED"}}
                        time.sleep(min(0.25, max(0.01, end - time.monotonic())))

        err = last_error or AgentError("API_UNAVAILABLE", "No Gemini model was available", "The AI service is unavailable.", True, 503)
        return {"type": "error", "text": f"{err.public_message} (error_id={err.error_id})", "tool_calls": [], "error": err.to_dict()["error"]}
