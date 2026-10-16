<div align="center">

# GenAgent 🤖

### Autonomous, Grounded Local Computer Agent with Zero Third-Party Runtime Dependencies

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Gemini](https://img.shields.io/badge/AI-Google%20Gemini-8E75B2?style=for-the-badge&logo=googlegemini&logoColor=white)](https://ai.google.dev/)
[![Tests](https://img.shields.io/badge/tests-passing-22C55E?style=for-the-badge)](tests/)
[![Dependencies](https://img.shields.io/badge/dependencies-0%20runtime-blue?style=for-the-badge)](#zero-external-dependencies)
[![Safety](https://img.shields.io/badge/safety-bounded%20%26%20audited-F59E0B?style=for-the-badge)](#safety-first)

</div>

`genagent` is a dependency-free, terminal-based autonomous computer assistant for local development, coding, and controlled workspace automation. It uses Google's Gemini models (with Gemini-only model fallbacks) to plan, observe, and execute safe operations inside your workspace.

## Highlights & Capabilities

- **Zero Third-Party Runtime Dependencies:** Runs purely on Python 3's Standard Library (`urllib`, `sqlite3`, `json`, `math`, `subprocess`, `difflib`). No heavy frameworks, no pip dependency bloat, and instantaneous startup.
- **Smart Surgical File Patching (`patch_file`):** Search-and-replace exact code sections without rewriting entire files, preventing code truncation and hallucinations on large files.
- **Multimodal Vision (`inspect_image`):** Direct image and screenshot understanding via Gemini Vision with zero external libraries (no OpenCV, PIL, or Tesseract required).
- **Dynamic Task Planning & Self-Correction:** Tracks step progression (`[DONE]`, `[ACTIVE]`, `[PENDING]`), detects tool errors, and dynamically reorganizes plans with alternative recovery steps.
- **Autonomous Background Daemon (`autonomous_daemon.py`):** Headless, queue-based background task runner for scheduled or unattended jobs without hanging on interactive prompts.
- **Bounded Safety & Workspace Sandboxing:** Strict workspace boundaries, command risk classification (Normal, Low Risk, Privileged, Destructive), and automated credential redaction.

## Available Tools

| Tool | Category | Description |
| --- | --- | --- |
| `read_file` | Filesystem | Read workspace files safely |
| `write_file` | Filesystem | Write full files with automated backup checkpointing |
| `patch_file` | Filesystem | Surgically search and replace sections without rewriting |
| `create_file` | Filesystem | Create new files safely (preventing unintended overwrite) |
| `delete_file` | Filesystem | Delete workspace files (protected action) |
| `list_directory` | Filesystem | Inspect directory contents |
| `run_command` | Terminal | Execute shell commands with timeout and risk validation |
| `verify_python` | Verification | Bytecode syntax compiler without running untrusted code |
| `verify_project` | Verification | Automatically discovers and runs project `unittest` suite |
| `inspect_image` | Vision | Multimodal image and UI inspection via Gemini Vision |
| `preview_diff` | Git | Non-destructive Git diff preview |
| `git_checkpoint` | Git | Create reversible git diff patch artifacts |
| `search_web` | Web | Dependency-free DuckDuckGo public web search fallback |

## Quick Start

### 1. Setup Environment

Clone the repository and prepare your configuration:

```bash
git clone https://github.com/<your-username>/genagent.git
cd genagent
cp .env.example .env
```

### 2. Configure Credentials

Add your Gemini API key to `.env`:

```env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3.6-flash
GEMINI_FALLBACK_MODELS=gemini-3.5-flash-lite,gemini-3.1-flash-lite
AUTONOMY_MODE=supervised
REQUIRE_CONFIRMATION=true
```

### 3. Run the Agent

**Interactive Terminal Mode:**
```bash
python3 agent.py
```

**Autonomous Background Daemon Mode:**
```bash
AUTONOMY_MODE=trusted python3 autonomous_daemon.py --once
```

## Running Tests

The test suite runs with Python's built-in `unittest` runner:

```bash
python3 -m unittest discover -s tests -p 'test_*.py' -v
```

## Project Structure

```text
├── agent.py                 # Interactive terminal CLI entrypoint
├── agent_loop.py            # Bounded observe-plan-act-verify loop
├── planning.py              # Dynamic plan state machine & self-recovery
├── planner.py               # Tool definitions and schema registry
├── gemini_client.py         # Dependency-free Gemini REST client
├── autonomous_daemon.py     # Headless background task worker
├── config.py                # Environment and configuration loader
├── permissions.py           # Risk classification & secret redaction
├── memory.py                # SQLite continuity memory
├── tools/
│   ├── filesystem.py        # File operations + patch_file
│   ├── vision.py            # Multimodal image analysis
│   ├── terminal.py          # Command execution engine
│   ├── verification.py      # Syntax and compile checks
│   └── ...
└── tests/                   # Standard library test suite
```

## License

MIT License. See [LICENCE](LICENCE) for details.
