#!/usr/bin/env python3
"""Dedicated Gemini setup wizard for GenAgent (with full iOS a-Shell & iSH support)."""
import os
import sys
import json
import re
import subprocess
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

from config import settings, is_configured, reload_settings


class ConsoleTheme:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.bold = "\033[1m" if enabled else ""
        self.cyan = "\033[36m" if enabled else ""
        self.green = "\033[32m" if enabled else ""
        self.yellow = "\033[33m" if enabled else ""
        self.red = "\033[31m" if enabled else ""
        self.gray = "\033[90m" if enabled else ""
        self.reset = "\033[0m" if enabled else ""

    def paint(self, color: str, text: str) -> str:
        return f"{color}{text}{self.reset}" if self.enabled else str(text)


def sanitize_key(key: str) -> str:
    """Clean invisible characters, iOS carriage returns, and smart quotes from pasted key."""
    if not key:
        return ""
    cleaned = key.strip().replace("\r", "").replace("\n", "").replace('"', '').replace("'", "")
    # Remove zero-width spaces or non-ascii artifacts
    cleaned = re.sub(r"[\u200b-\u200d\uFEFF]", "", cleaned)
    return cleaned.strip()


def get_clipboard_key() -> str | None:
    """Read API key from iOS clipboard (pbpaste in a-Shell / macOS)."""
    try:
        res = subprocess.run(["pbpaste"], capture_output=True, text=True, timeout=2)
        text = sanitize_key(res.stdout)
        if (text.startswith("AIza") or text.startswith("AQ.") or len(text) >= 25) and " " not in text:
            return text
    except Exception:
        pass
    return None


def get_file_key() -> str | None:
    """Read API key from key.txt or .key file in workspace."""
    base = Path(__file__).resolve().parent
    for fname in ("key.txt", ".key", "api_key.txt"):
        p = base / fname
        if p.exists():
            try:
                text = sanitize_key(p.read_text(encoding="utf-8"))
                if text:
                    return text
            except Exception:
                pass
    return None


def get_cli_key() -> str | None:
    """Read --key argument from command-line."""
    for idx, arg in enumerate(sys.argv):
        if arg in {"--key", "-k"} and idx + 1 < len(sys.argv):
            return sanitize_key(sys.argv[idx + 1])
        elif arg.startswith("--key="):
            return sanitize_key(arg.split("=", 1)[1])
    return None


def verify_gemini_key(api_key: str, model: str = "gemini-2.5-flash") -> tuple[bool, str]:
    """Test Gemini API key with a fast 1-second ping."""
    key = sanitize_key(api_key)
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
    payload = json.dumps({
        "contents": [{"role": "user", "parts": [{"text": "ping"}]}],
        "generationConfig": {"maxOutputTokens": 5}
    }).encode("utf-8")
    req = Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(req, timeout=8) as res:
            if res.status == 200:
                return True, "Key verified! Connected to Google Gemini."
    except HTTPError as exc:
        code = getattr(exc, "code", 0)
        if code in {401, 403}:
            return False, f"Invalid API Key (HTTP {code}). Check your key at https://aistudio.google.com"
        if code in {404, 429, 503}:
            return True, "Key authenticated successfully."
        return False, f"API rejected request (HTTP {code})."
    except (URLError, TimeoutError, OSError) as exc:
        return True, f"Network check skipped ({exc}). Key saved."
    return False, "Unknown verification error."


def save_env(key: str) -> bool:
    """Save Gemini key to .env file securely."""
    clean = sanitize_key(key)
    env_path = Path(__file__).resolve().parent / ".env"
    env_content = f"""# GenAgent Configuration (Gemini Powered)
LLM_PROVIDER=gemini
GEMINI_API_KEY={clean}
GEMINI_MODEL=gemini-3.6-flash
GEMINI_FALLBACK_MODELS=gemini-3.5-flash-lite,gemini-3.1-flash-lite
GEMINI_FAST_MODEL=gemini-3.5-flash-lite

AGENT_WORKSPACE=.
AGENT_HOST_MODE=false
AUTONOMY_MODE=supervised
REQUIRE_CONFIRMATION=true
AUTO_INSTALL=true
WEB_SEARCH_ENABLED=true
PLUGIN_ENABLED=true
SKILLS_ENABLED=true
MAX_AGENT_STEPS=50
COMMAND_TIMEOUT=120
"""
    try:
        env_path.write_text(env_content, encoding="utf-8")
        if os.name != "nt":
            try:
                os.chmod(env_path, 0o600)
            except OSError:
                pass
        reload_settings()
        return True
    except OSError as exc:
        print(f"Error saving .env file: {exc}")
        return False


def run_setup(interactive: bool = True, provided_key: str | None = None) -> bool:
    t = ConsoleTheme(enabled=True)
    banner_line = "=" * 60
    print(t.paint(t.cyan, "\n" + banner_line))
    print(t.paint(t.bold + t.cyan, "       GENAGENT  /  GOOGLE GEMINI SETUP"))
    print(t.paint(t.cyan, banner_line))

    # Priority 1: Passed key argument
    key = sanitize_key(provided_key or "")

    # Priority 2: CLI argument --key
    if not key:
        cli_k = get_cli_key()
        if cli_k:
            print(t.paint(t.gray, "  Found API key from command line argument."))
            key = cli_k

    # Priority 3: Environment variable GEMINI_API_KEY
    if not key and os.getenv("GEMINI_API_KEY"):
        env_k = sanitize_key(os.getenv("GEMINI_API_KEY", ""))
        if env_k:
            print(t.paint(t.gray, "  Found GEMINI_API_KEY in shell environment."))
            key = env_k

    # Priority 4: key.txt file (iOS safe method)
    if not key:
        file_k = get_file_key()
        if file_k:
            print(t.paint(t.gray, "  Found API key in local key.txt file."))
            key = file_k

    # Priority 5: Clipboard (iOS a-Shell / pbpaste)
    if not key:
        clip_k = get_clipboard_key()
        if clip_k:
            print(t.paint(t.green, f"  Detected Gemini key in iOS clipboard ({clip_k[:6]}...{clip_k[-4:]})!"))
            key = clip_k

    # Priority 6: Interactive Terminal Input (with iOS fallback)
    if not key and interactive:
        print(t.paint(t.yellow, "  Tip for iOS/iPad: You can also pass the key directly:"))
        print(t.paint(t.gray, "  python agent.py --key YOUR_GEMINI_KEY\n"))
        print(t.paint(t.cyan, "  Enter your Google Gemini API Key:"))
        print(t.paint(t.gray, "  (Get your free key from https://aistudio.google.com)"))
        try:
            raw_input = input(t.paint(t.green, "  API Key: ")).strip()
            key = sanitize_key(raw_input)
        except (KeyboardInterrupt, EOFError):
            print(t.paint(t.gray, "\nSetup cancelled."))
            return False

    if not key:
        print(t.paint(t.red, "\n✕ Error: No API key was provided."))
        print(t.paint(t.gray, "  On iOS a-Shell or iSH, you can easily run:"))
        print(t.paint(t.cyan, "  python setup.py --key YOUR_GEMINI_KEY\n"))
        return False

    print(t.paint(t.gray, "  Verifying key connection with Gemini..."))
    ok, msg = verify_gemini_key(key)
    if ok:
        print(t.paint(t.green, f"  ✓ {msg}"))
    else:
        print(t.paint(t.red, f"  ✕ {msg}"))
        if interactive:
            try:
                proceed = input("  Save key anyway? (y/n) [n]: ").strip().lower()
                if proceed not in {"y", "yes"}:
                    return False
            except (KeyboardInterrupt, EOFError):
                return False

    saved = save_env(key)
    if saved:
        print(t.paint(t.green, "  ✓ Configuration saved to .env."))
        print(t.paint(t.cyan, banner_line + "\n"))
        return True
    return False


if __name__ == "__main__":
    success = run_setup(interactive=True)
    if success:
        print("Setup completed successfully! Run 'python agent.py' to start.")
    else:
        sys.exit(1)
