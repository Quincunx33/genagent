from dataclasses import dataclass
import os
from pathlib import Path


def _load_dotenv(path: Path | None = None) -> None:
    path = path or Path(__file__).resolve().parent / ".env"
    if not path.exists():
        return
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if key:
            os.environ[key] = value


def _int_env(name: str, default: int, *, minimum: int = 1, maximum: int = 10_000_000) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(value, maximum))


def _bool_env(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).lower() in {"1", "true", "yes", "on"}


def _float_env(name: str, default: float, *, minimum: float, maximum: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(value, maximum))


_load_dotenv()


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    llm_provider: str = os.getenv("LLM_PROVIDER", "gemini").lower()
    llm_fallback_providers: tuple[str, ...] = tuple(p.strip().lower() for p in os.getenv("LLM_FALLBACK_PROVIDERS", "openai,xai,deepseek").split(",") if p.strip())
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-6-astra")
    openai_fallback_models: tuple[str, ...] = tuple(m.strip() for m in os.getenv("OPENAI_FALLBACK_MODELS", "gpt-5.6-terra").split(",") if m.strip())
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    xai_api_key: str = os.getenv("XAI_API_KEY", "")
    xai_model: str = os.getenv("XAI_MODEL", "grok-4.7")
    xai_fallback_models: tuple[str, ...] = tuple(m.strip() for m in os.getenv("XAI_FALLBACK_MODELS", "grok-4.6").split(",") if m.strip())
    xai_base_url: str = os.getenv("XAI_BASE_URL", "https://api.x.ai/v1")
    deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
    deepseek_model: str = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
    deepseek_fallback_models: tuple[str, ...] = tuple(m.strip() for m in os.getenv("DEEPSEEK_FALLBACK_MODELS", "deepseek-v4-pro").split(",") if m.strip())
    deepseek_base_url: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
    gemini_fallback_models: tuple[str, ...] = tuple(m.strip() for m in os.getenv("GEMINI_FALLBACK_MODELS", "gemini-3.5-flash-lite,gemini-3.1-flash-lite").split(",") if m.strip())
    workspace: Path = Path(os.getenv("AGENT_WORKSPACE", str(Path.cwd()))).expanduser().resolve()
    host_mode: bool = _bool_env("AGENT_HOST_MODE", False)
    allowed_paths: str = os.getenv("AGENT_ALLOWED_PATHS", "")
    denied_paths: str = os.getenv("AGENT_DENIED_PATHS", "")
    dry_run: bool = _bool_env("AGENT_DRY_RUN", False)
    skills_enabled: bool = _bool_env("SKILLS_ENABLED", True)
    skills_path: Path = Path(os.getenv("AGENT_SKILLS_PATH", str(Path(__file__).resolve().parent / "skills"))).expanduser().resolve()
    debug_mode: bool = _bool_env("AGENT_DEBUG", False)
    gemini_min_interval: float = _float_env("GEMINI_MIN_INTERVAL", 0.2, minimum=0.0, maximum=60.0)
    gemini_rate_limit: int = _int_env("GEMINI_RATE_LIMIT", 30, maximum=100_000)
    gemini_rate_window: int = _int_env("GEMINI_RATE_WINDOW", 60, maximum=86_400)
    gemini_backoff_base: float = _float_env("GEMINI_BACKOFF_BASE", 1.0, minimum=0.1, maximum=60.0)
    gemini_backoff_max: float = _float_env("GEMINI_BACKOFF_MAX", 16.0, minimum=1.0, maximum=600.0)
    gemini_persistent_limit: int = _int_env("GEMINI_PERSISTENT_LIMIT", 2000, maximum=100_000)
    gemini_persistent_window: int = _int_env("GEMINI_PERSISTENT_WINDOW", 86_400, maximum=7 * 86_400)
    max_agent_steps: int = _int_env("MAX_AGENT_STEPS", 50, maximum=1000)
    command_timeout: int = _int_env("COMMAND_TIMEOUT", 120, maximum=3600)
    max_retries: int = _int_env("MAX_RETRIES", 3, maximum=10)
    api_retries: int = _int_env("API_RETRIES", 3, maximum=10)
    max_file_size: int = _int_env("MAX_FILE_SIZE", 2 * 1024 * 1024, maximum=100 * 1024 * 1024)
    max_output_chars: int = _int_env("MAX_OUTPUT_CHARS", 20000, maximum=1_000_000)
    max_api_response_bytes: int = _int_env("MAX_API_RESPONSE_BYTES", 4 * 1024 * 1024, maximum=50 * 1024 * 1024)
    max_prompt_chars: int = _int_env("MAX_PROMPT_CHARS", 16000, maximum=200_000)
    max_history_chars: int = _int_env("MAX_HISTORY_CHARS", 12000, maximum=200_000)
    max_memory_chars: int = _int_env("MAX_MEMORY_CHARS", 3000, maximum=50_000)
    max_tool_result_chars: int = _int_env("MAX_TOOL_RESULT_CHARS", 5000, maximum=100_000)
    response_cache_enabled: bool = _bool_env("RESPONSE_CACHE_ENABLED", True)
    response_cache_ttl: int = _int_env("RESPONSE_CACHE_TTL", 300, maximum=86_400)
    response_cache_size: int = _int_env("RESPONSE_CACHE_SIZE", 128, maximum=10_000)
    task_tool_filtering: bool = _bool_env("TASK_TOOL_FILTERING", False)
    pure_primitives_only: bool = _bool_env("PURE_PRIMITIVES_ONLY", True)
    fast_model: str = os.getenv("GEMINI_FAST_MODEL", "gemini-3.5-flash-lite")
    log_rotate_hours: float = _float_env("LOG_ROTATE_HOURS", 24.0, minimum=1.0, maximum=24 * 30)
    log_retention: int = _int_env("LOG_RETENTION", 7, maximum=365)
    mobile_rate_limit: int = _int_env("MOBILE_RATE_LIMIT", 30, maximum=100_000)
    mobile_rate_window: int = _int_env("MOBILE_RATE_WINDOW", 60, maximum=86_400)
    mobile_max_body: int = _int_env("MOBILE_MAX_BODY", 64 * 1024, maximum=10 * 1024 * 1024)
    require_confirmation: bool = _bool_env("REQUIRE_CONFIRMATION", True)
    autonomy_mode: str = os.getenv("AUTONOMY_MODE", "supervised").lower() if os.getenv("AUTONOMY_MODE", "supervised").lower() in {"safe", "supervised", "trusted"} else "supervised"
    auto_install: bool = _bool_env("AUTO_INSTALL", False)
    install_timeout: int = _int_env("INSTALL_TIMEOUT", 300, maximum=3600)
    web_search_enabled: bool = _bool_env("WEB_SEARCH_ENABLED", True)
    plugin_enabled: bool = _bool_env("PLUGIN_ENABLED", True)
    max_parallel_agents: int = _int_env("MAX_PARALLEL_AGENTS", 3, maximum=4)
    db_path: Path = Path(os.getenv("AGENT_DB", str(Path.home() / ".gemini_computer_agent.sqlite3"))).expanduser()
    audit_log_path: Path = Path(os.getenv("AGENT_AUDIT_LOG", str(Path.home() / ".gemini_computer_agent_audit.jsonl"))).expanduser()


settings = Settings()


def is_configured() -> bool:
    """Check if any valid LLM provider API key is configured."""
    return bool(settings.gemini_api_key or settings.openai_api_key or settings.xai_api_key or settings.deepseek_api_key)


def reload_settings() -> Settings:
    """Reload configuration from .env and return fresh settings instance."""
    global settings
    _load_dotenv()
    settings = Settings()
    return settings


__all__ = ["Settings", "settings", "is_configured", "reload_settings"]
