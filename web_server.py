#!/usr/bin/env python3
"""
GenAgent Multi-Device Responsive Web Dashboard & HTML Server.
100% Python Standard Library (zero external pip/npm dependencies).
Ultra-responsive across all devices: iPhone, iPad, Android, Windows, Mac, Linux.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import queue
import re
import socket
import sys
import threading
import time
import urllib.parse
import uuid
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any, Dict, List, Optional

from config import settings, reload_settings
from platform_support import detect
from memory import Memory
from agent_loop import AgentLoop
from errors import normalize_exception

# Global agent instance & task synchronization
_memory = Memory()
_loop_lock = threading.Lock()
_current_task_id: Optional[str] = None
_task_events: Dict[str, queue.Queue] = {}
_is_busy = False

def get_local_ip() -> str:
    """Discover the local network IP address for cross-device LAN access."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip

def mask_key(k: str) -> str:
    if not k:
        return "Not Configured"
    if len(k) <= 10:
        return "****"
    return k[:6] + "..." + k[-4:]

def update_env_config(api_key: Optional[str] = None, model: Optional[str] = None, autonomy_mode: Optional[str] = None) -> bool:
    env_path = os.path.join(settings.workspace or ".", ".env")
    lines = []
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

    config_dict = {}
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            k, v = stripped.split("=", 1)
            config_dict[k.strip()] = v.strip()

    if api_key:
        config_dict["GEMINI_API_KEY"] = api_key.strip()
    if model:
        config_dict["GEMINI_MODEL"] = model.strip()
    if autonomy_mode:
        config_dict["AUTONOMY_MODE"] = autonomy_mode.strip()
        if autonomy_mode == "trusted":
            config_dict["REQUIRE_CONFIRMATION"] = "false"
        else:
            config_dict["REQUIRE_CONFIRMATION"] = "true"

    config_dict.setdefault("LLM_PROVIDER", "gemini")
    config_dict.setdefault("GEMINI_FAST_MODEL", "gemini-3.5-flash-lite")
    config_dict.setdefault("AGENT_WORKSPACE", ".")
    config_dict.setdefault("MAX_AGENT_STEPS", "50")

    out_lines = ["# GenAgent Configuration\n"]
    for k, v in config_dict.items():
        out_lines.append(str(k) + "=" + str(v) + "\n")

    with open(env_path, "w", encoding="utf-8") as f:
        f.writelines(out_lines)

    if os.name != "nt":
        try:
            os.chmod(env_path, 0o600)
        except OSError:
            pass

    reload_settings()
    return True


INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no, viewport-fit=cover">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="GenAgent Studio">
<meta name="theme-color" content="#07090e">
<title>GENAGENT Studio · Autonomous AI IDE</title>
<style>
:root {
  --bg-base: #07090e;
  --bg-surface: #0d111a;
  --bg-elevated: #131826;
  --bg-card: #182032;
  --bg-hover: #1f2b44;
  --border: rgba(255, 255, 255, 0.08);
  --border-focus: rgba(0, 229, 255, 0.4);
  --text-main: #f8fafc;
  --text-muted: #94a3b8;
  --text-dim: #64748b;
  --cyan: #00e5ff;
  --cyan-glow: rgba(0, 229, 255, 0.25);
  --purple: #8b5cf6;
  --purple-glow: rgba(139, 92, 246, 0.25);
  --emerald: #10b981;
  --amber: #f59e0b;
  --rose: #f43f5e;
  --code-bg: #04060a;
  --font-sans: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", Roboto, sans-serif;
  --font-mono: "JetBrains Mono", "Fira Code", ui-monospace, Menlo, Monaco, Consolas, monospace;
  --safe-top: env(safe-area-inset-top, 0px);
  --safe-bottom: env(safe-area-inset-bottom, 0px);
}

* { box-sizing: border-box; margin: 0; padding: 0; -webkit-tap-highlight-color: transparent; }

html, body {
  height: 100%;
  height: 100vh;
  height: 100dvh;
  background-color: var(--bg-base);
  color: var(--text-main);
  font-family: var(--font-sans);
  overflow: hidden;
  display: flex;
  flex-direction: column;
}

/* Ambient Background Glow */
body::before {
  content: "";
  position: fixed;
  top: -100px;
  left: 20%;
  width: 500px;
  height: 300px;
  background: radial-gradient(circle, var(--cyan-glow) 0%, transparent 70%);
  pointer-events: none;
  z-index: 0;
  filter: blur(60px);
  opacity: 0.6;
}

/* Top Glass Navigation Bar */
header.top-bar {
  height: calc(54px + var(--safe-top));
  padding-top: var(--safe-top);
  background: rgba(13, 17, 26, 0.85);
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding-left: 16px;
  padding-right: 16px;
  z-index: 100;
  flex-shrink: 0;
}

.brand-section {
  display: flex;
  align-items: center;
  gap: 12px;
}

.brand-logo {
  width: 32px;
  height: 32px;
  border-radius: 9px;
  background: linear-gradient(135deg, var(--cyan), var(--purple));
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 800;
  font-size: 16px;
  color: #000;
  box-shadow: 0 0 16px var(--cyan-glow);
}

.brand-title {
  display: flex;
  flex-direction: column;
}

.brand-title h1 {
  font-size: 14.5px;
  font-weight: 700;
  letter-spacing: -0.2px;
  display: flex;
  align-items: center;
  gap: 6px;
}

.brand-title .badge {
  font-size: 10px;
  padding: 1px 6px;
  border-radius: 10px;
  background: rgba(0, 229, 255, 0.15);
  color: var(--cyan);
  border: 1px solid rgba(0, 229, 255, 0.3);
  font-weight: 600;
}

/* Status & Mode Indicators */
.status-pill {
  font-size: 11.5px;
  padding: 4px 10px;
  border-radius: 20px;
  background: var(--bg-card);
  border: 1px solid var(--border);
  color: var(--text-muted);
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-weight: 500;
}

.status-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: var(--emerald);
  box-shadow: 0 0 8px var(--emerald);
  animation: pulseDot 2s infinite ease-in-out;
}

.status-dot.busy {
  background: var(--amber);
  box-shadow: 0 0 8px var(--amber);
}

@keyframes pulseDot {
  0%, 100% { opacity: 1; transform: scale(1); }
  50% { opacity: 0.5; transform: scale(0.85); }
}

/* Segmented View Switcher Tabs */
.view-tabs {
  display: flex;
  background: var(--bg-surface);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 3px;
  gap: 2px;
}

.tab-btn {
  background: transparent;
  border: none;
  color: var(--text-muted);
  font-size: 12.5px;
  font-weight: 600;
  padding: 5px 12px;
  border-radius: 7px;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 6px;
  transition: all 0.18s ease;
}

.tab-btn:hover {
  color: var(--text-main);
  background: rgba(255, 255, 255, 0.05);
}

.tab-btn.active {
  color: #fff;
  background: var(--bg-card);
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.4);
}

/* Action Icon Buttons */
.nav-actions {
  display: flex;
  align-items: center;
  gap: 8px;
}

.btn-icon {
  background: var(--bg-elevated);
  border: 1px solid var(--border);
  color: var(--text-muted);
  height: 32px;
  padding: 0 10px;
  border-radius: 8px;
  font-size: 12px;
  font-weight: 600;
  cursor: pointer;
  display: inline-flex;
  align-items: center;
  gap: 5px;
  transition: all 0.15s ease;
  text-decoration: none;
}

.btn-icon:hover {
  background: var(--bg-hover);
  color: var(--text-main);
  border-color: rgba(255, 255, 255, 0.15);
}

/* Studio Workspace Layout */
main.studio-stage {
  flex: 1;
  display: flex;
  overflow: hidden;
  position: relative;
  z-index: 10;
}

/* Chat Pane */
.chat-pane {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  border-right: 1px solid var(--border);
  transition: all 0.25s ease;
  min-width: 0;
}

.chat-history {
  flex: 1;
  overflow-y: auto;
  padding: 20px 18px;
  display: flex;
  flex-direction: column;
  gap: 16px;
  -webkit-overflow-scrolling: touch;
}

/* Message Rows */
.msg-card {
  display: flex;
  gap: 12px;
  max-width: 900px;
  width: 100%;
  margin: 0 auto;
  animation: fadeIn 0.2s ease-out;
}

@keyframes fadeIn {
  from { opacity: 0; transform: translateY(6px); }
  to { opacity: 1; transform: translateY(0); }
}

.msg-card.user {
  justify-content: flex-end;
}

.msg-card.user .msg-bubble {
  background: linear-gradient(135deg, #0284c7, #2563eb);
  color: #fff;
  border-radius: 16px 16px 4px 16px;
  padding: 12px 18px;
  font-size: 14.5px;
  line-height: 1.55;
  max-width: 82%;
  box-shadow: 0 4px 16px rgba(2, 132, 199, 0.25);
  word-break: break-word;
}

.msg-card.agent .msg-bubble {
  background: var(--bg-surface);
  border: 1px solid var(--border);
  border-radius: 16px 16px 16px 4px;
  padding: 18px;
  font-size: 14px;
  line-height: 1.65;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
  word-break: break-word;
  width: 100%;
}

.avatar-icon {
  width: 34px;
  height: 34px;
  border-radius: 10px;
  background: var(--bg-card);
  border: 1px solid var(--border);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 16px;
  flex-shrink: 0;
}

/* Tool Execution Step Cards */
.execution-card {
  background: var(--code-bg);
  border: 1px solid var(--border);
  border-left: 3px solid var(--cyan);
  border-radius: 8px;
  margin: 10px 0;
  padding: 10px 14px;
  font-family: var(--font-mono);
  font-size: 12px;
  color: #bae6fd;
}

.execution-card.ok {
  border-left-color: var(--emerald);
  color: #a7f3d0;
}

.execution-card.err {
  border-left-color: var(--rose);
  color: #fecdd3;
}

.execution-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}

.execution-badge {
  font-size: 10px;
  padding: 2px 6px;
  border-radius: 4px;
  background: rgba(255, 255, 255, 0.08);
  color: var(--text-muted);
}

/* Code Snippet & Markdown Styling */
.markdown-body pre {
  background: var(--code-bg);
  border: 1px solid var(--border);
  border-radius: 8px;
  margin: 12px 0;
  padding: 14px;
  overflow-x: auto;
  font-family: var(--font-mono);
  font-size: 13px;
  line-height: 1.5;
  position: relative;
}

.markdown-body pre code {
  color: #e2e8f0;
  background: transparent;
  padding: 0;
}

.copy-btn {
  position: absolute;
  top: 8px;
  right: 8px;
  background: var(--bg-card);
  border: 1px solid var(--border);
  color: var(--text-muted);
  font-size: 11px;
  padding: 3px 8px;
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.15s;
}

.copy-btn:hover {
  background: var(--bg-hover);
  color: var(--text-main);
  border-color: var(--cyan);
}

.markdown-body code {
  background: rgba(255, 255, 255, 0.08);
  font-family: var(--font-mono);
  padding: 2px 5px;
  border-radius: 4px;
  font-size: 13px;
}

.markdown-body p { margin-bottom: 10px; }
.markdown-body ul, .markdown-body ol { margin-left: 20px; margin-bottom: 10px; }

/* Floating Prompt Input Dock */
.input-dock {
  padding: 12px 16px calc(14px + var(--safe-bottom));
  background: rgba(13, 17, 26, 0.9);
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  border-top: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  gap: 8px;
  flex-shrink: 0;
}

.dock-inner {
  max-width: 900px;
  margin: 0 auto;
  width: 100%;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.chip-bar {
  display: flex;
  gap: 6px;
  overflow-x: auto;
  scrollbar-width: none;
  padding-bottom: 2px;
}

.chip-bar::-webkit-scrollbar { display: none; }

.quick-chip {
  background: var(--bg-surface);
  border: 1px solid var(--border);
  color: var(--text-muted);
  font-size: 11.5px;
  font-weight: 500;
  padding: 5px 12px;
  border-radius: 14px;
  cursor: pointer;
  white-space: nowrap;
  transition: all 0.15s ease;
}

.quick-chip:hover {
  background: var(--bg-card);
  color: var(--cyan);
  border-color: var(--border-focus);
}

.input-container {
  display: flex;
  align-items: flex-end;
  background: var(--bg-elevated);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 8px 10px;
  gap: 8px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
  transition: border-color 0.2s, box-shadow 0.2s;
}

.input-container:focus-within {
  border-color: var(--cyan);
  box-shadow: 0 0 0 2px var(--cyan-glow);
}

textarea#prompt-input {
  flex: 1;
  background: transparent;
  border: none;
  color: var(--text-main);
  font-family: inherit;
  font-size: 15px;
  line-height: 1.45;
  resize: none;
  outline: none;
  max-height: 160px;
  min-height: 24px;
}

.dock-actions {
  display: flex;
  align-items: center;
  gap: 6px;
}

.btn-circle {
  width: 34px;
  height: 34px;
  border-radius: 50%;
  border: none;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: all 0.2s ease;
}

.btn-voice {
  background: var(--bg-card);
  color: var(--text-muted);
  border: 1px solid var(--border);
}

.btn-voice.recording {
  background: var(--rose);
  color: #fff;
  animation: pulseDot 1s infinite;
}

.btn-send {
  background: linear-gradient(135deg, var(--cyan), #0284c7);
  color: #000;
  font-weight: 700;
}

.btn-send:hover {
  transform: scale(1.05);
  box-shadow: 0 0 14px var(--cyan-glow);
}

/* Right Studio Panel (File Explorer & Code Editor) */
.studio-side-panel {
  width: 450px;
  background: var(--bg-surface);
  border-left: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  overflow: hidden;
  transition: transform 0.25s ease;
}

.panel-header {
  height: 44px;
  padding: 0 14px;
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  justify-content: space-between;
  background: var(--bg-elevated);
}

.panel-header-title {
  font-size: 13px;
  font-weight: 600;
  display: flex;
  align-items: center;
  gap: 8px;
}

.split-content {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

/* File Tree */
.file-tree-container {
  height: 180px;
  overflow-y: auto;
  border-bottom: 1px solid var(--border);
  background: var(--bg-surface);
  padding: 8px;
}

.tree-node {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 10px;
  border-radius: 6px;
  font-size: 12.5px;
  color: var(--text-muted);
  cursor: pointer;
  transition: all 0.12s;
  user-select: none;
}

.tree-node:hover {
  background: var(--bg-elevated);
  color: var(--text-main);
}

.tree-node.active {
  background: rgba(0, 229, 255, 0.12);
  color: var(--cyan);
  font-weight: 600;
}

/* In-Browser Code Editor */
.editor-wrapper {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.editor-toolbar {
  padding: 6px 12px;
  background: var(--bg-elevated);
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  justify-content: space-between;
}

textarea#code-editor {
  flex: 1;
  width: 100%;
  background: var(--code-bg);
  color: #38bdf8;
  font-family: var(--font-mono);
  font-size: 13px;
  line-height: 1.55;
  padding: 14px;
  border: none;
  outline: none;
  resize: none;
  white-space: pre;
  overflow: auto;
}

/* Settings Modal */
.modal-overlay {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.7);
  backdrop-filter: blur(8px);
  -webkit-backdrop-filter: blur(8px);
  z-index: 1000;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 16px;
  animation: fadeIn 0.15s ease-out;
}

.modal-dialog {
  background: var(--bg-surface);
  border: 1px solid var(--border);
  border-radius: 16px;
  width: 100%;
  max-width: 520px;
  box-shadow: 0 20px 40px rgba(0, 0, 0, 0.6);
  overflow: hidden;
}

.modal-head {
  padding: 16px 20px;
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.modal-head h3 {
  font-size: 15px;
  font-weight: 700;
  display: flex;
  align-items: center;
  gap: 8px;
}

.modal-close {
  background: transparent;
  border: none;
  color: var(--text-muted);
  font-size: 20px;
  cursor: pointer;
}

.modal-body {
  padding: 20px;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.form-item label {
  display: block;
  font-size: 12px;
  font-weight: 600;
  color: var(--text-muted);
  margin-bottom: 6px;
}

.form-item input, .form-item select {
  width: 100%;
  background: var(--bg-elevated);
  border: 1px solid var(--border);
  color: var(--text-main);
  padding: 10px 12px;
  border-radius: 8px;
  font-size: 13.5px;
  outline: none;
}

.form-item input:focus, .form-item select:focus {
  border-color: var(--cyan);
}

.input-btn-group {
  display: flex;
  gap: 6px;
}

.modal-foot {
  padding: 14px 20px;
  background: var(--bg-elevated);
  border-top: 1px solid var(--border);
  display: flex;
  justify-content: flex-end;
  gap: 8px;
}

/* Responsive Media Queries */
@media (max-width: 900px) {
  .studio-side-panel {
    display: none;
    position: absolute;
    inset: 0;
    width: 100%;
    z-index: 80;
  }
  .studio-side-panel.open-mobile {
    display: flex;
  }
}
</style>
</head>
<body>

<!-- Header -->
<header class="top-bar">
  <div class="brand-section">
    <div class="brand-logo">G</div>
    <div class="brand-title">
      <h1>GENAGENT <span class="badge">Studio</span></h1>
    </div>
    <div class="status-pill">
      <span class="status-dot" id="global-status-dot"></span>
      <span id="global-status-text">Online</span>
    </div>
  </div>

  <nav class="view-tabs" id="view-tabs-nav">
    <button class="tab-btn active" onclick="setStudioView('chat')" id="tab-btn-chat">💬 Chat</button>
    <button class="tab-btn" onclick="setStudioView('files')" id="tab-btn-files">📁 Files</button>
    <button class="tab-btn" onclick="setStudioView('split')" id="tab-btn-split">💻 Split Studio</button>
  </nav>

  <div class="nav-actions">
    <button class="btn-icon" onclick="openSettingsModal()">⚙️ Settings</button>
    <a href="/api/download" class="btn-icon" download>📦 Zip</a>
  </div>
</header>

<!-- Main Stage -->
<main class="studio-stage">
  <!-- Left / Main Chat -->
  <section class="chat-pane" id="chat-pane">
    <div class="chat-history" id="chat-history">
      <div class="msg-card agent">
        <div class="avatar-icon">🤖</div>
        <div class="msg-bubble">
          <strong>GenAgent Studio · AI Computer Assistant</strong><br>
          Autonomous developer agent running directly on your workspace.<br>
          <span style="font-size:12px; color:var(--text-muted);">Fully optimized for iPad, Android, Mac, Linux, and Windows.</span>
        </div>
      </div>
    </div>

    <!-- Floating Bottom Dock -->
    <div class="input-dock">
      <div class="dock-inner">
        <div class="chip-bar">
          <span class="quick-chip" onclick="applyQuickTask('Check system status and platform health')">⚡ System Status</span>
          <span class="quick-chip" onclick="applyQuickTask('List files in the workspace')">📂 List Files</span>
          <span class="quick-chip" onclick="applyQuickTask('Run unittest test suite')">🧪 Run Tests</span>
          <span class="quick-chip" onclick="applyQuickTask('Explain this project from README.md')">📖 Explain Project</span>
          <span class="quick-chip" onclick="clearChat()">🧹 Clear</span>
        </div>
        <div class="input-container">
          <textarea id="prompt-input" placeholder="Message GenAgent or instruct an action..." rows="1" onkeydown="handleInputKey(event)" oninput="autoExpand(this)"></textarea>
          <div class="dock-actions">
            <button class="btn-circle btn-voice" id="voice-btn" onclick="toggleVoiceInput()" title="Voice Input">🎙️</button>
            <button class="btn-circle btn-send" id="send-btn" onclick="submitUserTask()" title="Send">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg>
            </button>
          </div>
        </div>
      </div>
    </div>
  </section>

  <!-- Right / Live Workspace & File Editor -->
  <aside class="studio-side-panel" id="side-panel">
    <div class="panel-header">
      <div class="panel-header-title">
        <span>📁 Workspace Explorer</span>
      </div>
      <button class="btn-icon" onclick="refreshWorkspaceFiles()">🔄 Refresh</button>
    </div>
    <div class="split-content">
      <div class="file-tree-container" id="file-tree-list">
        <div style="padding:10px; font-size:12px; color:var(--text-dim);">Loading workspace tree...</div>
      </div>
      <div class="editor-wrapper">
        <div class="editor-toolbar">
          <span id="active-file-label" style="font-size:12px; font-weight:600; color:var(--text-muted);">No file selected</span>
          <button class="btn-icon" onclick="saveActiveFile()">💾 Save</button>
        </div>
        <textarea id="code-editor" placeholder="Click any file above to view or edit source code..."></textarea>
      </div>
    </div>
  </aside>
</main>

<!-- Settings Modal -->
<div id="settings-modal" class="modal-overlay" style="display:none;" onclick="handleModalBackdrop(event)">
  <div class="modal-dialog">
    <div class="modal-head">
      <h3>⚙️ Studio Settings</h3>
      <button class="modal-close" onclick="closeSettingsModal()">&times;</button>
    </div>
    <div class="modal-body">
      <div class="form-item">
        <label>Google Gemini API Key:</label>
        <div class="input-btn-group">
          <input type="password" id="cfg-api-key" placeholder="Enter API Key (AQ.Ab8... / AIzaSy...)">
          <button class="btn-icon" onclick="toggleKeyMask()" id="btn-mask">👁️</button>
          <button class="btn-icon" onclick="verifyKeyLive()">🔍 Test</button>
        </div>
        <small id="key-status-msg" style="display:block; margin-top:4px; font-size:11px; color:var(--text-dim);">Checking credentials...</small>
      </div>

      <div class="form-item">
        <label>Primary AI Model:</label>
        <div class="input-btn-group">
          <select id="cfg-model" style="flex:1"><option value="">Enter an API key and load models</option></select>
          <button class="btn-icon" onclick="loadGeminiModels()">↻ Load models</button>
        </div>
        <small style="display:block; margin-top:4px; font-size:11px; color:var(--text-dim);">Shows models available to your Gemini API key that support text generation.</small>
      </div>

      <div class="form-item">
        <label>Autonomy Mode:</label>
        <select id="cfg-autonomy">
          <option value="trusted">Trusted Mode (Execute safe tools autonomously)</option>
          <option value="supervised">Supervised Mode (Confirm dangerous actions)</option>
        </select>
      </div>
    </div>
    <div class="modal-foot">
      <button class="btn-icon" onclick="closeSettingsModal()">Cancel</button>
      <button class="btn-icon" style="background:var(--cyan); color:#000; font-weight:700;" onclick="saveSettings()">Save & Apply</button>
    </div>
  </div>
</div>

<script>
let activeTaskId = null;
let eventSource = null;
let currentActiveFile = null;
let isBusy = false;
let speechRec = null;

// Initialization
document.addEventListener("DOMContentLoaded", () => {
  fetchStatus();
  refreshWorkspaceFiles();
  fetchConfig();
  initVoice();
});

// Auto-expand textarea
function autoExpand(el) {
  el.style.height = "auto";
  el.style.height = Math.min(el.scrollHeight, 160) + "px";
}

function handleInputKey(e) {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    submitUserTask();
  }
}

function applyQuickTask(text) {
  const input = document.getElementById("prompt-input");
  input.value = text;
  autoExpand(input);
  submitUserTask();
}

function setStudioView(mode) {
  document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
  const btn = document.getElementById("tab-btn-" + mode);
  if (btn) btn.classList.add("active");

  const side = document.getElementById("side-panel");
  const chat = document.getElementById("chat-pane");

  if (mode === "chat") {
    side.style.display = "none";
    chat.style.display = "flex";
  } else if (mode === "files") {
    side.style.display = "flex";
    side.style.width = "100%";
    chat.style.display = "none";
  } else if (mode === "split") {
    side.style.display = "flex";
    side.style.width = "460px";
    chat.style.display = "flex";
  }
}

async function fetchStatus() {
  try {
    const res = await fetch("/api/status");
    const data = await res.json();
    const dot = document.getElementById("global-status-dot");
    const txt = document.getElementById("global-status-text");
    if (data.is_busy) {
      dot.className = "status-dot busy";
      txt.innerText = "Running Task...";
      isBusy = true;
    } else {
      dot.className = "status-dot";
      txt.innerText = "Online";
      isBusy = false;
    }
  } catch (err) {
    console.error("fetchStatus error", err);
  }
}

async function fetchConfig() {
  try {
    const res = await fetch("/api/config");
    const data = await res.json();
    if (data.api_key_masked) {
      document.getElementById("cfg-api-key").placeholder = "Key Configured: " + data.api_key_masked;
      document.getElementById("key-status-msg").innerText = "Key is active and configured.";
    }
    if (data.model) document.getElementById("cfg-model").value = data.model;
    if (data.autonomy_mode) document.getElementById("cfg-autonomy").value = data.autonomy_mode;
    if (data.is_configured) loadGeminiModels(data.model);
  } catch (e) {}
}

async function loadGeminiModels(preferredModel = "") {
  const key = document.getElementById("cfg-api-key").value.trim();
  const select = document.getElementById("cfg-model");
  const selected = preferredModel || select.value;
  select.innerHTML = '<option value="">Loading models...</option>';
  try {
    const res = await fetch("/api/models", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ api_key: key })
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.message || "Could not load Gemini models");
    select.innerHTML = "";
    (data.models || []).forEach(model => {
      const option = document.createElement("option");
      option.value = model.id;
      option.textContent = model.display_name + " — " + model.id;
      option.title = model.description || model.id;
      select.appendChild(option);
    });
    if (!data.models || !data.models.length) throw new Error("No text-generation models were returned for this API key");
    if ([...select.options].some(option => option.value === selected)) select.value = selected;
  } catch (err) {
    select.innerHTML = '<option value="">Model list unavailable — check API key/network</option>';
    document.getElementById("key-status-msg").innerText = err.message;
  }
}

async function refreshWorkspaceFiles() {
  try {
    const res = await fetch("/api/files");
    const data = await res.json();
    const list = document.getElementById("file-tree-list");
    list.innerHTML = "";
    (data.files || []).forEach(f => {
      const item = document.createElement("div");
      item.className = "tree-node" + (currentActiveFile === f.path ? " active" : "");
      item.innerHTML = (f.is_dir ? "📁 " : "📄 ") + f.name;
      if (!f.is_dir) {
        item.onclick = () => openFileInEditor(f.path);
      }
      list.appendChild(item);
    });
  } catch (err) {
    console.error("refreshWorkspaceFiles error", err);
  }
}

async function openFileInEditor(path) {
  currentActiveFile = path;
  document.getElementById("active-file-label").innerText = path;
  refreshWorkspaceFiles();
  try {
    const res = await fetch("/api/file?path=" + encodeURIComponent(path));
    const data = await res.json();
    document.getElementById("code-editor").value = data.content || "";
  } catch (err) {
    alert("Could not load file: " + err);
  }
}

async function saveActiveFile() {
  if (!currentActiveFile) return alert("No file selected to save.");
  const content = document.getElementById("code-editor").value;
  try {
    const res = await fetch("/api/file", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: currentActiveFile, content })
    });
    const d = await res.json();
    if (d.ok) alert("File saved successfully!");
  } catch (err) {
    alert("Save error: " + err);
  }
}

// Submit Task via API
async function submitUserTask() {
  const input = document.getElementById("prompt-input");
  const task = input.value.trim();
  if (!task || isBusy) return;

  input.value = "";
  autoExpand(input);
  appendUserMessage(task);

  const card = createAgentResponseCard();
  const stepContainer = card.querySelector(".step-logs");
  const contentBody = card.querySelector(".content-body");

  try {
    isBusy = true;
    document.getElementById("global-status-dot").className = "status-dot busy";
    document.getElementById("global-status-text").innerText = "Executing...";

    const res = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ task })
    });

    if (res.status === 429) {
      contentBody.innerText = "Agent is currently busy with another operation. Please wait a moment.";
      isBusy = false;
      fetchStatus();
      return;
    }

    const data = await res.json();
    activeTaskId = data.task_id;
    listenToStream(activeTaskId, stepContainer, contentBody);
  } catch (err) {
    contentBody.innerText = "Connection error: " + err;
    isBusy = false;
    fetchStatus();
  }
}

function listenToStream(taskId, stepContainer, contentBody) {
  if (eventSource) eventSource.close();
  eventSource = new EventSource("/api/stream?task_id=" + taskId);

  eventSource.onmessage = (e) => {
    try {
      const ev = JSON.parse(e.data);
      if (ev.type === "step") {
        const step = document.createElement("div");
        step.className = "execution-card";
        step.innerText = ev.text;
        stepContainer.appendChild(step);
        scrollToBottom();
      } else if (ev.type === "tool_result") {
        const r = document.createElement("div");
        r.className = "execution-card " + (ev.ok ? "ok" : "err");
        r.innerText = ev.text;
        stepContainer.appendChild(r);
        scrollToBottom();
      } else if (ev.type === "done") {
        eventSource.close();
        isBusy = false;
        fetchStatus();
        refreshWorkspaceFiles();
        renderFormattedResponse(contentBody, ev.response || "");
        scrollToBottom();
      }
    } catch (err) {}
  };

  eventSource.onerror = () => {
    eventSource.close();
    isBusy = false;
    fetchStatus();
  };
}

function appendUserMessage(text) {
  const history = document.getElementById("chat-history");
  const row = document.createElement("div");
  row.className = "msg-card user";
  row.innerHTML = `<div class="msg-bubble">${escapeHtml(text)}</div>`;
  history.appendChild(row);
  scrollToBottom();
}

function createAgentResponseCard() {
  const history = document.getElementById("chat-history");
  const row = document.createElement("div");
  row.className = "msg-card agent";
  row.innerHTML = `
    <div class="avatar-icon">🤖</div>
    <div class="msg-bubble">
      <div class="step-logs"></div>
      <div class="content-body" style="font-size:14px; margin-top:8px;">
        <span style="color:var(--cyan); animation:pulseDot 1.5s infinite;">● Planning & executing...</span>
      </div>
    </div>
  `;
  history.appendChild(row);
  scrollToBottom();
  return row;
}

function renderFormattedResponse(container, text) {
  let html = escapeHtml(text);
  // Code block matching
  html = html.replace(/```([a-zA-Z0-9_+-]*)\n([\s\S]*?)```/g, (m, lang, code) => {
    return `<pre><button class="copy-btn" onclick="copySnippet(this)">Copy</button><code>${code.trim()}</code></pre>`;
  });
  // Inline code
  html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
  // Bold
  html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  // Newlines
  html = html.replace(/\n/g, "<br>");
  container.innerHTML = `<div class="markdown-body">${html}</div>`;
}

function copySnippet(btn) {
  const code = btn.nextElementSibling.innerText;
  navigator.clipboard.writeText(code).then(() => {
    const orig = btn.innerText;
    btn.innerText = "Copied!";
    setTimeout(() => btn.innerText = orig, 1800);
  });
}

function scrollToBottom() {
  const h = document.getElementById("chat-history");
  h.scrollTop = h.scrollHeight;
}

function clearChat() {
  const h = document.getElementById("chat-history");
  h.innerHTML = `
    <div class="msg-card agent">
      <div class="avatar-icon">🤖</div>
      <div class="msg-bubble">Chat cleared. Ready for your next command!</div>
    </div>
  `;
}

function escapeHtml(str) {
  const d = document.createElement("div");
  d.innerText = str;
  return d.innerHTML;
}

// Voice Input
function initVoice() {
  const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRec) {
    document.getElementById("voice-btn").style.display = "none";
    return;
  }
  speechRec = new SpeechRec();
  speechRec.continuous = false;
  speechRec.interimResults = false;
  speechRec.onresult = (e) => {
    const transcript = e.results[0][0].transcript;
    const input = document.getElementById("prompt-input");
    input.value = (input.value ? input.value + " " : "") + transcript;
    autoExpand(input);
  };
  speechRec.onend = () => {
    document.getElementById("voice-btn").classList.remove("recording");
  };
}

function toggleVoiceInput() {
  if (!speechRec) return;
  const btn = document.getElementById("voice-btn");
  if (btn.classList.contains("recording")) {
    speechRec.stop();
    btn.classList.remove("recording");
  } else {
    speechRec.start();
    btn.classList.add("recording");
  }
}

// Settings Modal
function openSettingsModal() {
  document.getElementById("settings-modal").style.display = "flex";
}

function closeSettingsModal() {
  document.getElementById("settings-modal").style.display = "none";
}

function handleModalBackdrop(e) {
  if (e.target.id === "settings-modal") closeSettingsModal();
}

function toggleKeyMask() {
  const input = document.getElementById("cfg-api-key");
  input.type = input.type === "password" ? "text" : "password";
}

async function verifyKeyLive() {
  const key = document.getElementById("cfg-api-key").value.trim();
  const model = document.getElementById("cfg-model").value;
  const msg = document.getElementById("key-status-msg");
  msg.innerText = "Testing API connection...";
  try {
    const res = await fetch("/api/verify_key", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ api_key: key, model })
    });
    const d = await res.json();
    msg.style.color = d.ok ? "var(--emerald)" : "var(--rose)";
    msg.innerText = (d.ok ? "✓ " : "✕ ") + d.message;
  } catch (err) {
    msg.style.color = "var(--rose)";
    msg.innerText = "Error: " + err;
  }
}

async function saveSettings() {
  const key = document.getElementById("cfg-api-key").value.trim();
  const model = document.getElementById("cfg-model").value;
  const autonomy = document.getElementById("cfg-autonomy").value;
  try {
    const res = await fetch("/api/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ api_key: key, model, autonomy_mode: autonomy })
    });
    const d = await res.json();
    alert(d.message || "Settings saved!");
    closeSettingsModal();
    fetchStatus();
  } catch (err) {
    alert("Save failed: " + err);
  }
}
</script>
</body>
</html>"""

class GenAgentWebHandler(BaseHTTPRequestHandler):
    def _send_json(self, status: int, data: Any):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        # 1. Main HTML Dashboard
        if path in {"/", "/index.html"}:
            body = INDEX_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        # Zip Download API
        if path in {"/api/download", "/download"}:
            zip_candidates = ["/workspace/genagent_github_ready.zip", "/workspace/genagent.zip", "genagent.zip"]
            found_zip = None
            for z in zip_candidates:
                if os.path.exists(z):
                    found_zip = z
                    break
            if found_zip:
                with open(found_zip, "rb") as zf:
                    data = zf.read()
                self.send_response(200)
                self.send_header("Content-Type", "application/zip")
                self.send_header("Content-Disposition", "attachment; filename=genagent.zip")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            self._send_json(404, {"error": "Zip file not ready yet."})
            return

        # Config API
        if path == "/api/config":
            k = settings.gemini_api_key or ""
            masked = mask_key(k)
            self._send_json(200, {
                "is_configured": bool(k),
                "api_key_masked": masked,
                "model": settings.gemini_model,
                "autonomy_mode": settings.autonomy_mode
            })
            return

        # 2. Status API
        if path == "/api/status":
            plat = detect()
            data = {
                "model": settings.gemini_model,
                "workspace": os.path.abspath(settings.workspace),
                "autonomy_mode": settings.autonomy_mode,
                "platform": {
                    "system": plat.system,
                    "profile": plat.profile,
                    "is_ios": plat.is_ios_shell,
                    "is_termux": plat.is_termux,
                },
                "is_busy": _is_busy
            }
            self._send_json(200, data)
            return

        # 3. Files List API
        if path == "/api/files":
            files = []
            ws = settings.workspace or "."
            for item in sorted(os.listdir(ws)):
                if item.startswith(".") and item not in {".env.example"}:
                    continue
                p = os.path.join(ws, item)
                files.append({
                    "name": item,
                    "path": item,
                    "is_dir": os.path.isdir(p),
                    "size": os.path.getsize(p) if os.path.isfile(p) else 0
                })
            self._send_json(200, {"files": files})
            return

        # 4. Read File API
        if path == "/api/file":
            target_path = query.get("path", [""])[0]
            if not target_path:
                self._send_json(400, {"error": "Missing 'path' parameter"})
                return
            try:
                with open(target_path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read(50000)
                self._send_json(200, {"path": target_path, "content": content})
            except Exception as e:
                self._send_json(500, {"error": str(e)})
            return

        # 5. Logs API
        if path == "/api/logs":
            log_p = "agent.log"
            logs = ""
            if os.path.exists(log_p):
                with open(log_p, "r", encoding="utf-8", errors="replace") as f:
                    lines = f.readlines()
                    logs = "".join(lines[-100:])
            body = logs.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        # 6. SSE Real-Time Event Stream
        if path == "/api/stream":
            task_id = query.get("task_id", [""])[0]
            if not task_id or task_id not in _task_events:
                self.send_response(404)
                self.end_headers()
                return

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            q = _task_events[task_id]
            while True:
                try:
                    event = q.get(timeout=25.0)
                    msg = f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                    self.wfile.write(msg.encode("utf-8"))
                    self.wfile.flush()
                    if event.get("type") == "done":
                        break
                except queue.Empty:
                    self.wfile.write(b": heartbeat\n\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    break
            return

        self._send_json(404, {"error": "Not Found"})

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(length) if length > 0 else b"{}"
        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except Exception:
            payload = {}

        # Verify Key API
        if path == "/api/verify_key":
            k = payload.get("api_key", "").strip() or settings.gemini_api_key
            m = payload.get("model", "gemini-flash-lite-latest")
            if not k:
                self._send_json(400, {"ok": False, "message": "API key cannot be empty."})
                return
            from setup import verify_gemini_key
            ok, msg = verify_gemini_key(k, m)
            self._send_json(200, {"ok": ok, "message": msg})
            return

        # Gemini Models API (the API key is accepted in the body, never the URL)
        if path == "/api/models":
            k = payload.get("api_key", "").strip() or settings.gemini_api_key
            if not k:
                self._send_json(400, {"ok": False, "message": "Enter or configure a Gemini API key first."})
                return
            try:
                from gemini_client import list_gemini_models
                models = list_gemini_models(k)
                self._send_json(200, {"ok": True, "models": models})
            except Exception as exc:
                self._send_json(502, {"ok": False, "message": f"Could not load models: {exc}"})
            return

        # Save Config API
        if path == "/api/config":
            k = payload.get("api_key", "").strip()
            m = payload.get("model", "").strip()
            auto = payload.get("autonomy_mode", "").strip()
            update_env_config(api_key=k if k else None, model=m if m else None, autonomy_mode=auto if auto else None)
            self._send_json(200, {"ok": True, "message": "Settings updated and applied successfully!"})
            return

        # Run Task API
        if path == "/api/run":
            task_text = payload.get("task", "").strip()
            if not task_text:
                self._send_json(400, {"error": "Task must not be empty"})
                return

            global _is_busy
            if _is_busy:
                self._send_json(429, {"error": "Agent is currently busy with another task"})
                return

            task_id = uuid.uuid4().hex[:8]
            q = queue.Queue()
            _task_events[task_id] = q

            def worker():
                global _is_busy
                _is_busy = True
                try:
                    def output_logger(msg: str):
                        clean = msg.strip()
                        if clean.startswith(">"):
                            q.put({"type": "step", "text": clean})
                        elif clean.startswith("OK"):
                            q.put({"type": "tool_result", "text": clean, "ok": True})
                        elif clean.startswith("X"):
                            q.put({"type": "tool_result", "text": clean, "ok": False})

                    mem = Memory()
                    loop = AgentLoop(output=output_logger, memory=mem, debug=False)
                    response = loop.run(task_text)
                    q.put({"type": "done", "response": response})
                except Exception as exc:
                    err = normalize_exception(exc)
                    q.put({"type": "done", "response": f"[Error]: {err.public_message} ({err.suggestion})"})
                finally:
                    _is_busy = False

            threading.Thread(target=worker, daemon=True).start()
            self._send_json(200, {"task_id": task_id, "status": "started"})
            return

        # Save File API
        if path == "/api/file":
            target_path = payload.get("path", "")
            content = payload.get("content", "")
            if not target_path:
                self._send_json(400, {"error": "Missing path"})
                return
            try:
                with open(target_path, "w", encoding="utf-8") as f:
                    f.write(content)
                self._send_json(200, {"ok": True, "path": target_path})
            except Exception as exc:
                self._send_json(500, {"ok": False, "error": str(exc)})
            return

        self._send_json(404, {"error": "Not Found"})

    def log_message(self, format, *args):
        # Silence default terminal request logs to keep terminal clean
        pass


def run_web_server(host: str = "0.0.0.0", port: int = 8080):
    """Start the responsive HTML Web Dashboard server with smart auto-port selection."""
    HTTPServer.allow_reuse_address = True
    start_port = port
    server = None
    for p in range(start_port, start_port + 50):
        try:
            server = HTTPServer((host, p), GenAgentWebHandler)
            port = p
            break
        except OSError as e:
            if getattr(e, "errno", None) in (98, 48, 10048) or "address already in use" in str(e).lower():
                continue
            raise
    if not server:
        raise RuntimeError(f"Could not bind to any port between {start_port} and {start_port + 50}")

    local_ip = get_local_ip()
    plat = detect()
    print("=" * 66)
    print("       GENAGENT MOBILE & MULTI-DEVICE HTML WEB SERVER       ")
    print("=" * 66)
    print(f"  Platform Profile : {plat.profile} ({plat.shell_family})")
    print(f"  Local Device URL : http://localhost:{port}")
    print(f"  Network / Phone  : http://{local_ip}:{port}")
    print("=" * 66)
    print("  Open the Network URL above on any iPhone, iPad, Android phone,")
    print("  tablet, or other PC connected to the same Wi-Fi/network!")
    print("  Press Ctrl+C to stop the web server.")
    print("=" * 66)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nWeb server stopped safely.")
    finally:
        server.server_close()


def main():
    parser = argparse.ArgumentParser(description="GenAgent Multi-Device HTML Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host address (default: 0.0.0.0 for LAN access)")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on (default: 8080)")
    args = parser.parse_args()

    run_web_server(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
