from __future__ import annotations

import json
import re
import time
import threading
import sys
from typing import Any

from config import settings
from errors import AgentError, ErrorCategory, error_text, normalize_exception, traceback_text
from gemini_client import GeminiClient
from logger import get_logger
from memory import Memory
from planner import tools_for_task
from state import AgentState
from task_store import TaskStore
from tools.terminal import run_command
from tools.filesystem import read_file, write_file, patch_file, create_file, move_file, delete_file, list_directory
from tools.verification import verify_python
from tools.control import platform_info, open_app, list_processes, terminate_process
from tools.gui_control import gui_capabilities, screenshot, ocr, mouse_click, type_text, press_key
from tools.vision import inspect_image
from tools.self_update import self_update
from text_safety import safe_text, safe_value
from platform_support import command_environment, detect as detect_platform, supported_tool_names, tool_support
from permissions import redact_secrets
from skill_registry import SkillRegistry
from ui import EventRenderer
from approval import ApprovalBroker
from tool_validation import validate_tool_args
from planning import make_plan, as_prompt, TaskPlan
from checkpoints import checkpoint
from tools.filesystem import safe_path
from tools.fallbacks import find_alternatives, verify_tool, install_and_verify
from web_search import search_web
from plugin_manager import PluginManager
from tools.project_verify import verify_project
from git_tools import preview_diff, git_checkpoint


log = get_logger("genagent.loop")




class AgentLoop:
    def __init__(self, client=None, output=print, store=None, memory=None, debug=None, non_interactive=False):
        self.client = client or GeminiClient()
        self.output = output
        self.store = store or TaskStore(settings.db_path)
        self.memory = memory or Memory()
        self.skills = SkillRegistry(settings.skills_path) if settings.skills_enabled else None
        self.ui = EventRenderer(settings.debug_mode if debug is None else debug)
        self.cancel_event = threading.Event()
        self.approvals = ApprovalBroker()
        self.plan: TaskPlan | None = None
        self.last_task_id: str | None = None
        self.non_interactive = non_interactive

    def cancel(self) -> None:
        self.cancel_event.set()

    def _cancelled(self, task_id: str, task: str, state: AgentState) -> str | None:
        if not self.cancel_event.is_set():
            return None
        state.cancelled = True
        self._save(task_id, task, "cancelled", state)
        self.cancel_event.clear()
        self.output(safe_text("  [STOPPED] Task stopped safely. Changes made so far are intact."))
        return f"Task cancelled at step {state.step}. Resume with task_id={task_id} if needed."

    def _save(self, task_id: str, task: str, status: str, state: AgentState) -> None:
        try:
            self.store.save(task_id, task, status, state.step, state.history)
        except Exception as exc:
            log.error("task persistence failed status=%s error=%s", status, error_text(exc, operation="task persistence"), exc_info=True)

    def execute(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        from permissions import classify_command, Risk

        if not isinstance(name, str) or not name:
            raise AgentError("INVALID_TOOL_CALL", "Tool name is missing", "The model returned an invalid tool call.", category=ErrorCategory.VALIDATION, retryable=False, status=422)
        if not isinstance(args, dict):
            raise AgentError("INVALID_TOOL_ARGS", f"Arguments for {name} must be an object", "The model returned invalid tool arguments.", category=ErrorCategory.VALIDATION, retryable=False, status=422)
        if name == "search_web" and not settings.web_search_enabled:
            return {"error": "web search is disabled", "code": "FEATURE_DISABLED"}
        if name == "install_plugin" and not settings.plugin_enabled:
            return {"error": "plugin installation is disabled", "code": "FEATURE_DISABLED"}
        valid, reason = validate_tool_args(name, args)
        if not valid:
            return {"error": f"Invalid arguments: {reason}", "code": "INVALID_TOOL_ARGS", "retryable": False}
        supported, reason = tool_support(name)
        if not supported:
            return {"error": f"Tool '{name}' is unavailable on this platform: {reason}", "code": "UNSUPPORTED_PLATFORM", "platform": detect_platform().profile}
        if settings.dry_run and name in {"run_command", "write_file", "patch_file", "create_file", "move_file", "delete_file", "self_update", "open_app", "terminate_process", "mouse_click", "type_text", "press_key"}:
            return {"dry_run": True, "tool": name, "args": redact_secrets(json.dumps(args, ensure_ascii=False, default=str)), "message": "No side effect executed because AGENT_DRY_RUN=true"}

        requires_confirm = settings.require_confirmation
        if settings.autonomy_mode == "safe" and name not in {"list_skills", "find_alternatives", "verify_tool", "search_web", "parallel_analysis", "verify_project", "preview_diff", "list_directory", "read_file", "platform_info", "gui_capabilities", "inspect_image"}:
            requires_confirm = True
        elif settings.autonomy_mode == "trusted":
            if name in {"write_file", "patch_file", "create_file", "move_file", "self_update", "git_checkpoint"}:
                requires_confirm = False
            elif name == "run_command" and classify_command(str(args.get("command", ""))) != Risk.DESTRUCTIVE:
                requires_confirm = False
        if name in {"delete_file", "terminate_process"}:
            requires_confirm = True
        if name in {"list_skills", "find_alternatives", "verify_tool", "verify_python", "search_web", "parallel_analysis", "verify_project", "preview_diff", "list_directory", "read_file", "platform_info", "gui_capabilities", "inspect_image"}:
            requires_confirm = False
        elif name == "run_command" and classify_command(str(args.get("command", ""))) == Risk.NORMAL:
            requires_confirm = False

        if requires_confirm:
            approval = self.approvals.create(name, args)
            self.output(safe_text(f"  Approval ID: {approval.approval_id}" if self.ui.debug else ""))
            self.output(safe_text(self.ui.permission(name, args)))
            if self.non_interactive:
                self.approvals.resolve(approval.approval_id, False)
                return {"error": "Permission denied: non-interactive mode cannot prompt user for tool: " + name, "code": "PERMISSION_DENIED_HEADLESS"}
            try:
                confirm = input("Allow execution? [y/N]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                if settings.autonomy_mode == "trusted":
                    self.approvals.resolve(approval.approval_id, True)
                    confirm = "y"
                else:
                    self.approvals.resolve(approval.approval_id, False)
                    return {"error": "User confirmation was interrupted or unavailable", "code": "PERMISSION_DENIED"}
            if confirm not in {"y", "yes"}:
                self.approvals.resolve(approval.approval_id, False)
                return {"error": "Permission denied by user for tool: " + name, "code": "PERMISSION_DENIED"}
            self.approvals.resolve(approval.approval_id, True)
        else:
            notice = self.ui.auto(name)
            if notice:
                self.output(safe_text(notice))

        dispatch: dict[str, Any] = {
            "run_command": lambda: run_command(command=args.get("command"), cwd=args.get("cwd"), timeout=args.get("timeout"), approved=not requires_confirm),
            "read_file": lambda: {"content": read_file(args["path"])},
            "write_file": lambda: {"checkpoint": str(checkpoint(safe_path(args["path"]), settings.workspace) or ""), "written": write_file(args["path"], args["content"])},
            "patch_file": lambda: {"checkpoint": str(checkpoint(safe_path(args["path"]), settings.workspace) or ""), "patched": patch_file(args["path"], args["search"], args["replace"], args.get("count", 1))},
            "create_file": lambda: {"created": create_file(args["path"], args["content"])},
            "move_file": lambda: {"moved": move_file(args["source"], args["destination"])},
            "delete_file": lambda: {"deleted": delete_file(args["path"], approved=True)},
            "self_update": lambda: self_update(args["path"], args["content"]),
            "list_directory": lambda: {"files": list_directory(args.get("path", "."))},
            "verify_python": lambda: verify_python(args["path"]),
            "verify_tool": lambda: verify_tool(args["requested"]),
            "install_and_verify": lambda: install_and_verify(args["requested"]),
            "install_package": lambda: __import__("tools.fallbacks", fromlist=["install_package"]).install_package(args["package"], args.get("manager", "auto")),
            "inspect_code": lambda: __import__("tools.code_intel", fromlist=["inspect_code"]).inspect_code(args["path"], args.get("symbol")),
            "create_snapshot": lambda: __import__("tools.snapshot", fromlist=["create_snapshot"]).create_snapshot(args.get("label", "")),
            "restore_snapshot": lambda: __import__("tools.snapshot", fromlist=["restore_snapshot"]).restore_snapshot(args["snapshot_id"]),
            "create_skill": lambda: __import__("skill_registry", fromlist=["create_skill"]).create_skill(args["name"], args["description"], args["instructions"]),
            "synthesize_tool": lambda: __import__("tools.tool_synthesis", fromlist=["synthesize_tool"]).synthesize_tool(args["name"], args["code"], args.get("test_code"), args.get("description", "")),
            "execute_synthesized_tool": lambda: __import__("tools.tool_synthesis", fromlist=["execute_synthesized_tool"]).execute_synthesized_tool(args["name"], args["function"], args.get("args")),
            "analyze_logs": lambda: __import__("tools.log_analyzer", fromlist=["analyze_logs"]).analyze_logs(args.get("log_text"), args.get("path"), args.get("max_lines", 50)),
            "check_port": lambda: __import__("tools.watchdog", fromlist=["check_port"]).check_port(args.get("host", "127.0.0.1"), args["port"], args.get("timeout", 2.0)),
            "check_process_resources": lambda: __import__("tools.watchdog", fromlist=["check_process_resources"]).check_process_resources(args.get("pid")),
            "preview_impact": lambda: __import__("tools.safety_preview", fromlist=["preview_impact"]).preview_impact(args.get("command"), args.get("tool"), args.get("args")),
            "search_web": lambda: search_web(args["query"], args.get("limit", 5)),
            "install_plugin": lambda: PluginManager().install_zip(args["archive"]),
            "parallel_analysis": lambda: __import__("subagents").run_parallel(args["tasks"], min(args.get("max_workers", settings.max_parallel_agents), settings.max_parallel_agents)),
            "verify_project": verify_project,
            "preview_diff": lambda: preview_diff(args.get("path")),
            "git_checkpoint": lambda: git_checkpoint(args.get("label", "checkpoint")),
            "platform_info": platform_info,
            "list_skills": lambda: {"skills": self.skills.list() if self.skills else []},
            "find_alternatives": lambda: find_alternatives(args["requested"]),
            "open_app": lambda: open_app(args["target"]),
            "list_processes": list_processes,
            "terminate_process": lambda: terminate_process(args["pid"], approved=True),
            "gui_capabilities": gui_capabilities,
            "screenshot": lambda: screenshot(args.get("path", "screen.png")),
            "ocr": lambda: ocr(args["path"]),
            "inspect_image": lambda: inspect_image(args["path"], args.get("prompt", "Describe this image")),
            "mouse_click": lambda: mouse_click(args["x"], args["y"], args.get("clicks", 1)),
            "type_text": lambda: type_text(args["text"]),
            "press_key": lambda: press_key(args["key"]),
        }
        if name not in dispatch:
            raise AgentError("UNKNOWN_TOOL", f"Unknown tool: {name}", "The requested tool is not available.", category=ErrorCategory.VALIDATION, retryable=False, status=422)
        try:
            result = dispatch[name]()
            if name == "read_file" and isinstance(result, dict) and isinstance(result.get("content"), str):
                result = {**result, "content": redact_secrets(result["content"])}
            return safe_value(result)
        except Exception as exc:
            err = normalize_exception(exc, operation=f"tool {name}")
            log.warning("tool failed name=%s code=%s error_id=%s", name, err.code, err.error_id, exc_info=True)
            err_dict = {
                "error": err.public_message,
                "code": err.code,
                "category": err.category,
                "error_id": err.error_id,
                "retryable": err.retryable,
            }
            if err.suggestion:
                err_dict["suggestion"] = err.suggestion
            if err.root_cause:
                err_dict["root_cause"] = err.root_cause
            return err_dict

    @staticmethod
    def _call_key(call: dict[str, Any]) -> str:
        return json.dumps({"name": call.get("name"), "args": call.get("args", {})}, sort_keys=True, ensure_ascii=False, default=str)

    @staticmethod
    def _unstructured_tool_request(text: str, tools: list[dict[str, Any]]) -> bool:
        """Detect tool-call-shaped JSON text without treating it as executable authority."""
        candidate = (text or "").strip()
        if candidate.startswith("```"):
            candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", candidate, flags=re.IGNORECASE).strip()
        try:
            payload = json.loads(candidate)
        except (TypeError, ValueError):
            return False
        calls = payload.get("tool_calls") if isinstance(payload, dict) else None
        if not isinstance(calls, list) or not calls:
            return False
        available = {item.get("name") for item in tools if isinstance(item, dict)}
        return all(isinstance(call, dict) and call.get("name") in available for call in calls)

    def _prompt(self, task: str, state: AgentState, previous_result: str | None = None) -> str:
        if state.step > 1:
            sections = [
                f"Task: {task}",
                f"Latest progress:\n{previous_result or 'None'}"
            ]
            if self.plan and getattr(self.plan, "steps", None):
                p_text = as_prompt(self.plan)
                if p_text:
                    sections.append(f"Plan status:\n{p_text}")
            sections.append(f"This is step {state.step} of a bounded run. For action tasks, do not claim completion until every requested deliverable exists and the requested verification is confirmed by tool results. If work remains, continue with tools; if blocked, say so plainly. If everything is verified, provide the final answer.")
            return "\n\n".join(sections)[:settings.max_prompt_chars]

        # Step 1: Initial concise system prompt
        platform_info = getattr(self, "_platform_info", detect_platform())
        sections = [
            f"You are GenAgent, an autonomous computer assistant.\nWorkspace: {settings.workspace}\nPlatform: {platform_info.profile} ({platform_info.shell_family})",
            "User request:\n" + str(task),
        ]

        if self.skills:
            skill_context = self.skills.context(task)
            if skill_context and "no specialized" not in skill_context.lower() and "disabled" not in skill_context.lower():
                sections.append("Skill guidance:\n" + skill_context)

        if self.plan and getattr(self.plan, "steps", None):
            plan_context = as_prompt(self.plan)
            if plan_context:
                sections.append("Task plan:\n" + plan_context)

        memory_context = self.memory.context(task)[:settings.max_memory_chars].strip()
        if memory_context and "no relevant" not in memory_context.lower():
            sections.append("Prior context:\n" + memory_context)

        contract = (
            "Working contract:\n"
            "- Tool Usage Rule: Call tools only when the request requires reading/modifying files or executing commands. For conversation, questions, or code explanations, answer directly in natural language without calling tools.\n"
            "- To create or edit files, always call write_file or patch_file with actual file content.\n"
            "- Before finalizing an action task, compare every requested deliverable and acceptance criterion with actual tool results. Run requested tests/checks and fix failures; never claim completion or passing tests from intent alone. If blocked or incomplete, report that accurately.\n"
            "- When tools were executed, conclude with: 1. What changed 2. Verification evidence 3. Remaining risk. (For direct conversational replies, answer naturally).\n\n"
            "This is step 1 of a bounded run. Use an available tool only if action is needed; otherwise give the final answer."
        )
        sections.append(contract)

        return "\n\n".join(sections)[:settings.max_prompt_chars]

    def run(self, task: str, task_id: str | None = None) -> str:
        if task_id:
            try:
                saved = self.store.load(task_id)
            except Exception as exc:
                return error_text(exc, operation="load task")
            if not saved:
                return f"Task not found: {task_id}"
            state = AgentState(task=saved["task"], step=saved["step"], history=saved["history"])
            task = saved["task"]
        else:
            state = AgentState(task=task)
            try:
                task_id = self.store.create(task)
            except Exception as exc:
                return error_text(exc, operation="create task")
        self.last_task_id = task_id

        cancelled = self._cancelled(task_id, task, state)
        if cancelled:
            return cancelled


        self.plan = make_plan(task)
        self._platform_info = detect_platform()
        tools = tools_for_task(task, settings.task_tool_filtering, self._platform_info)
        if hasattr(self.client, "set_task_route"):
            self.client.set_task_route(task)

        previous_result: str | None = None
        recent_calls: list[str] = []
        seen_results: dict[str, str] = {}
        consecutive_no_progress = 0
        self.last_task_tokens = {"prompt": 0, "candidates": 0, "total": 0}
        unstructured_tool_retries = 0
        for step_idx in range(settings.max_agent_steps):
            state.step += 1
            if step_idx > 0:
                time.sleep(max(0.15, settings.gemini_min_interval))
            cancelled = self._cancelled(task_id, task, state)
            if cancelled:
                return cancelled
            self.output(safe_text(self.ui.thinking()))
            continuing_tool_response = bool(state.history and state.history[-1].get("role") == "tool")
            request_prompt = "" if continuing_tool_response else self._prompt(task, state, previous_result)
            try:
                result = self.client.generate(request_prompt, tools, state.history, cancel_event=self.cancel_event)
            except TypeError:
                result = self.client.generate(request_prompt, tools, state.history)
            except KeyboardInterrupt:
                self.cancel_event.set()
                return self._cancelled(task_id, task, state) or "Task cancelled."
            except Exception as exc:
                err = normalize_exception(exc, operation="agent model request")
                self._save(task_id, task, "paused" if err.retryable else "failed", state)
                log.error("model request failed error_id=%s", err.error_id, exc_info=True)
                suffix = f" Task paused; resume with task_id={task_id}." if err.retryable else ""
                return f"{err.public_message} (error_id={err.error_id}, task_id={task_id}){suffix}"

            if not isinstance(result, dict):
                err = AgentError("INVALID_MODEL_RESPONSE", "Model client returned a non-object", "The AI service returned an invalid response.", category=ErrorCategory.VALIDATION, retryable=False, status=502)
                self._save(task_id, task, "failed", state)
                return f"{err.public_message} (error_id={err.error_id}, task_id={task_id})"
            if request_prompt:
                state.add("user", request_prompt)
            if not result.get("tool_calls") and self._unstructured_tool_request(str(result.get("text", "")), tools):
                unstructured_tool_retries += 1
                message = "The previous response represented a tool call as plain JSON text. No tool was executed. Continue the task using a native Gemini function call, or give a truthful final response if no action remains."
                self.output(safe_text("  [RECOVERY] Model returned a tool request as plain text; no action was executed."))
                if unstructured_tool_retries >= 2:
                    failure = "Task not completed: Gemini repeatedly returned tool-call JSON as plain text instead of a structured function call. No such text was executed."
                    self._save(task_id, task, "failed", state)
                    return failure
                previous_result = message
                state.add("assistant", {"text": "[unstructured tool-call text withheld]", "tool_calls": []})
                self._save(task_id, task, "running", state)
                continue
            unstructured_tool_retries = 0
            if result.get("type") == "error":
                text = str(result.get("text") or "The model request failed.")
                error_info = result.get("error") if isinstance(result.get("error"), dict) else {}
                resumable = bool(error_info.get("retryable"))
                self._save(task_id, task, "paused" if resumable else "failed", state)
                self.output(safe_text(text))
                if resumable:
                    return f"{text} Task paused; resume with task_id={task_id}."
                return text
            if result.get("type") == "cancelled" or self.cancel_event.is_set():
                return self._cancelled(task_id, task, state) or "Task cancelled."
            state.add("assistant", {"text": result.get("text", ""), "tool_calls": result.get("tool_calls", []), "gemini_content": result.get("gemini_content")})
            if isinstance(result, dict) and "usage" in result:
                u = result["usage"]
                p_tokens = u.get("prompt_tokens", 0)
                c_tokens = u.get("candidates_tokens", 0)
                # In multi-turn chat, prompt_tokens represents the cumulative context of that turn.
                # Use peak context for prompt size and sum generated output tokens:
                self.last_task_tokens["prompt"] = max(self.last_task_tokens.get("prompt", 0), p_tokens)
                self.last_task_tokens["candidates"] += c_tokens
                self.last_task_tokens["total"] = self.last_task_tokens["prompt"] + self.last_task_tokens["candidates"]
                self.last_task_tokens["api_billed"] = self.last_task_tokens.get("api_billed", 0) + u.get("total_tokens", 0)
            calls = result.get("tool_calls", [])
            if result.get("type") == "text" or not calls:
                text = str(result.get("text", "")).strip() or "I could not produce a useful final summary."
                self._save(task_id, task, "completed", state)
                self.memory.save("completed_task", {"task": task, "summary": text})
                if self.last_task_tokens["total"] > 0:
                    tok_msg = f"  [Tokens] {self.last_task_tokens['total']:,} tokens (Prompt: {self.last_task_tokens['prompt']:,} · Output: {self.last_task_tokens['candidates']:,})"
                    self.output(safe_text(self.ui.theme.paint(self.ui.theme.gray, tok_msg)))
                return text

            if not isinstance(calls, list):
                err = AgentError("INVALID_MODEL_RESPONSE", "Model tool_calls must be a list", "The AI service returned invalid tool calls.", category=ErrorCategory.VALIDATION, retryable=False, status=502)
                self._save(task_id, task, "failed", state)
                return f"{err.public_message} (error_id={err.error_id}, task_id={task_id})"
            tool_results = []
            for call in calls:
                cancelled = self._cancelled(task_id, task, state)
                if cancelled:
                    return cancelled
                if not isinstance(call, dict):
                    tool_results.append({"error": "Invalid tool call", "code": "INVALID_TOOL_CALL"})
                    continue
                key = self._call_key(call)
                recent_calls = (recent_calls + [key])[-8:]
                is_repetition = (len(recent_calls) >= 3 and len(set(recent_calls[-3:])) == 1)
                if not is_repetition and len(recent_calls) >= 4:
                    if recent_calls[-1] == recent_calls[-3] and recent_calls[-2] == recent_calls[-4]:
                        is_repetition = True
                    elif recent_calls[-6:].count(key) >= 3:
                        is_repetition = True

                if is_repetition:
                    consecutive_no_progress += 1
                    message = {"error": "The same tool call was requested repeatedly without progress. Choose a different safe approach or explain the blocker.", "code": "NO_PROGRESS"}
                    if self.plan:
                        self.plan.record_failure_and_replan(f"Repeated tool {call.get('name')}", "Repeated identical or cyclic call with no progress")
                    self.output(safe_text(self.ui.result("recovery", message, 0)))
                    if consecutive_no_progress >= 2:
                        blocker_msg = "Action loop detected: Unable to complete operation after repeated attempts without progress. Please provide specific file names, exact paths, or check command arguments."
                        self._save(task_id, task, "paused", state)
                        self.output(safe_text(blocker_msg))
                        return blocker_msg
                    tool_results.append(message)
                    continue
                else:
                    consecutive_no_progress = 0
                name, args = call.get("name", ""), call.get("args", {})
                self.output(safe_text(self.ui.start(name, args)))
                started = time.perf_counter()
                try:
                    tool_result = self.execute(name, args)
                except KeyboardInterrupt:
                    self.cancel_event.set()
                    return self._cancelled(task_id, task, state) or "Task cancelled."
                tool_result = safe_value(tool_result)
                self.output(safe_text(self.ui.result(name, tool_result, (time.perf_counter() - started) * 1000)))
                
                # Check for error and update dynamic plan
                if isinstance(tool_result, dict) and tool_result.get("error"):
                    if self.plan:
                        self.plan.record_failure_and_replan(f"Tool {name}", str(tool_result.get("error")))
                elif self.plan:
                    self.plan.mark_completed(f"Used {name}")

                if name in {"find_alternatives", "verify_tool", "install_and_verify", "search_web"}:
                    self.memory.save("tool_outcome", {"tool": name, "args": redact_secrets(json.dumps(args, ensure_ascii=False)), "result": tool_result})
                result_key = json.dumps(tool_result, sort_keys=True, ensure_ascii=False, default=str)
                if result_key in seen_results:
                    compact_result = {"deduplicated": True, "same_as": seen_results[result_key], "note": "identical result already present in recent context"}
                    tool_results.append(compact_result)
                    state.add("tool", {"tool": name, "args": args, "result": compact_result, "call_id": call.get("id")})
                else:
                    seen_results[result_key] = name
                    state.add("tool", {"tool": name, "args": args, "result": tool_result, "call_id": call.get("id")})
                    tool_results.append(tool_result)

            previous_result = json.dumps(safe_value(tool_results), ensure_ascii=False, default=str)[:settings.max_tool_result_chars]
            self._save(task_id, task, "running", state)

        self._save(task_id, task, "paused", state)
        return f"Task paused at step limit. Resume with task_id={task_id}."

    def resume(self, task_id: str) -> str:
        return self.run("", task_id=task_id)
