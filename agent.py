#!/usr/bin/env python3
"""Interactive terminal entrypoint for genagent."""
import os
import sys

from text_safety import configure_terminal
from config import settings, is_configured, reload_settings
from agent_loop import AgentLoop
from memory import Memory
from platform_support import detect
from ui import EventRenderer, format_response


class Console:
    def __init__(self):
        self.ui = EventRenderer(color=True)
        self.enabled = self.ui.theme.enabled
        self.t = self.ui.theme

    def c(self, color: str, text: str) -> str:
        return self.t.paint(color, text)

    def width(self) -> int:
        try:
            return max(56, min(100, os.get_terminal_size().columns))
        except OSError:
            return 72

    def fit(self, value: str, limit: int | None = None) -> str:
        limit = limit or max(28, self.width() - 16)
        value = str(value)
        if len(value) <= limit:
            return value
        return "..." + value[-(limit - 3):]

    def clear(self) -> None:
        if os.getenv("NO_CLEAR") is None:
            print("\033[2J\033[H", end="")

    def header(self) -> None:
        width = self.width()
        line = "-" * width
        print(self.c(self.t.blue, "+" + line + "+"))
        title = " GENAGENT  /  GROUNDED LOCAL ASSISTANT "
        print(self.c(self.t.blue, "|") + self.c(self.t.bold + self.t.cyan, title.center(width)) + self.c(self.t.blue, "|"))
        print(self.c(self.t.blue, "+" + line + "+"))
        print(f"  {self.c(self.t.gray, 'Model')}     {self.c(self.t.yellow, settings.gemini_model)}")
        print(f"  {self.c(self.t.gray, 'Workspace')} {self.c(self.t.yellow, self.fit(settings.workspace, min(48, width - 20)))}")
        print(f"  {self.c(self.t.gray, 'Mode')}      {self.c(self.t.green, settings.autonomy_mode)}  {self.c(self.t.gray, '- type /help for commands')}")
        print()

    def help(self) -> None:
        print(self.ui.divider())
        print(self.c(self.t.bold + self.t.cyan, "  COMMANDS"))
        rows = [
            ("/status", "Show model, mode, platform and step limits"),
            ("/tools", "List available automation tools"),
            ("/model", "Show active model"),
            ("/workspace", "Show current workspace"),
            ("/files", "Show workspace file analytics & summary"),
            ("/tree", "Show visual workspace directory tree"),
            ("/approvals", "Show protected-action approvals"),
            ("/setup", "Launch interactive setup wizard"),
            ("/clear-memory", "Clear saved task memory"),
            ("/debug on|off", "Toggle structured developer events"),
            ("/cancel", "Stop the current task safely"),
            ("/exit", "Close the agent"),
        ]
        for command, description in rows:
            print(f"  {self.c(self.t.green, command):<24} {self.c(self.t.gray, description)}")
        print(self.ui.divider())


def main() -> None:
    configure_terminal()
    console = Console()

    # Priority 1: Check if API key was passed via CLI (e.g. python agent.py --key YOUR_KEY)
    from setup import get_cli_key, run_setup, save_env
    cli_key = get_cli_key()
    if cli_key:
        save_env(cli_key)
        reload_settings()

    # Priority 2: Auto setup if unconfigured
    if not is_configured():
        console.clear()
        print(console.c(console.t.yellow, "\n  [Notice] GenAgent is not configured yet."))
        print(console.c(console.t.cyan, "  Starting automatic setup wizard...\n"))
        success = run_setup(interactive=True)
        if not success:
            print(console.c(console.t.red, "\n  Setup incomplete. Run 'python agent.py --key YOUR_GEMINI_KEY' when ready.\n"))
            return
        reload_settings()
        console.clear()

    memory = Memory()
    console.clear()
    console.header()
    loop = AgentLoop(output=print, memory=memory, debug=settings.debug_mode)

    # Optional: CLI single-task execution (e.g. python agent.py "run calculator")
    cli_tasks = [arg for arg in sys.argv[1:] if not arg.startswith("-") and arg != cli_key]
    if cli_tasks:
        single_task = " ".join(cli_tasks).strip()
        if single_task:
            print(f"\n{console.ui.divider()}")
            print(f"  {console.c(console.t.cyan, 'TASK')}  {single_task[:160]}")
            response = loop.run(single_task)
            print(f"\n{console.ui.divider()}")
            clean_response = format_response(response)
            print(f"  {console.c(console.t.bold + console.t.green, 'AGENT')}")
            print("  " + clean_response.replace("\n", "\n  "))
            return

    while True:
        try:
            text = input(f"{console.c(console.t.green, '  You')} {console.c(console.t.cyan, '>')} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{console.c(console.t.gray, 'Session closed.')}")
            break
        if not text:
            continue
        if text == "/exit":
            print(f"{console.c(console.t.gray, 'Session closed. Goodbye.')}")
            break
        if text == "/help":
            console.help()
            continue
        if text == "/status":
            info = detect()
            print(console.ui.divider())
            print(f"  {console.c(console.t.cyan, 'STATUS')}")
            print(f"  Model      {settings.gemini_model}")
            print(f"  Mode       {settings.autonomy_mode}")
            print(f"  Steps      {settings.max_agent_steps}")
            print(f"  Platform   {info.profile}  ·  {info.shell_family}")
            print(f"  Debug      {'on' if loop.ui.debug else 'off'}")
            print(console.ui.divider())
            continue
        if text == "/model":
            active_m = getattr(loop.client, 'model', settings.gemini_model)
            print(f"  {console.c(console.t.cyan, 'MODEL')}  {active_m} (use /model fast or /model primary to switch)")
            continue
        if text in {"/model fast", "/fast"}:
            if hasattr(loop.client, 'model'):
                loop.client.model = settings.fast_model
                loop.client.models = (settings.fast_model, *settings.gemini_fallback_models)
            print(f"  {console.c(console.t.green, '✓')} Switched to Light/Fast model: {settings.fast_model}")
            continue
        if text in {"/model primary", "/model default"}:
            if hasattr(loop.client, 'model'):
                loop.client.model = settings.gemini_model
                loop.client.models = (settings.gemini_model, *settings.gemini_fallback_models)
            print(f"  {console.c(console.t.green, '✓')} Switched to Primary capable model: {settings.gemini_model}")
            continue
        if text in {"/files", "/summary"}:
            from file_manager import AIFileManager, main as fm_main
            old_argv = sys.argv
            sys.argv = ["file_manager.py", "summary"]
            fm_main()
            sys.argv = old_argv
            continue
        if text in {"/tree", "/ls"}:
            from file_manager import AIFileManager, main as fm_main
            old_argv = sys.argv
            sys.argv = ["file_manager.py", "tree", "-d", "2"]
            fm_main()
            sys.argv = old_argv
            continue
        if text == "/workspace":
            print(f"  {console.c(console.t.cyan, 'WORKSPACE')}  {settings.workspace}")
            continue
        if text == "/tools":
            from planner import tools_for_task
            t_names = [t["name"] for t in tools_for_task("")]
            print(f"  {console.c(console.t.cyan, 'TOOLS')} ({len(t_names)} available):")
            print(f"  {console.c(console.t.gray, ', '.join(t_names))}")
            continue
        if text == "/setup":
            from setup import run_setup
            if run_setup(interactive=True):
                reload_settings()
                loop = AgentLoop(output=print, memory=memory, debug=settings.debug_mode)
                print(f"  {console.c(console.t.green, '✓')} Configuration reloaded.")
            continue
        if text == "/clear-memory":
            memory.clear()
            print(f"  {console.c(console.t.green, '✓')} Memory cleared")
            continue
        if text == "/cancel":
            loop.cancel()
            print(f"  {console.c(console.t.yellow, '⚠')} Current task will stop at the next safe boundary")
            continue
        if text == "/approvals":
            print(f"  {console.c(console.t.cyan, 'APPROVALS')}  {loop.approvals.list()}")
            continue
        if text.startswith("/debug"):
            parts = text.split(maxsplit=1)
            if len(parts) == 1:
                print(f"  {console.c(console.t.cyan, 'DEBUG')}  {'on' if loop.ui.debug else 'off'}")
            else:
                enabled = parts[1].strip().lower() in {"on", "true", "1", "yes"}
                loop.ui.set_debug(enabled)
                print(f"  {console.c(console.t.cyan, 'DEBUG')}  {'on' if enabled else 'off'}")
            continue

        print(f"\n{console.ui.divider()}")
        print(f"  {console.c(console.t.cyan, 'TASK')}  {text[:160]}")
        try:
            response = loop.run(text)
            print(f"\n{console.ui.divider()}")
            clean_response = format_response(response)
            print(f"  {console.c(console.t.bold + console.t.green, 'AGENT')}")
            print("  " + clean_response.replace("\n", "\n  "))
            print()
        except Exception as exc:
            print(f"\n  {console.c(console.t.red, '✕ Error')}  {exc}\n")


if __name__ == "__main__":
    main()
