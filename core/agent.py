"""Agentic capabilities – web search, PC control, tool use and MCP scaffolding."""
import os
import json
import logging
import subprocess
import platform
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool definitions (returned to the frontend for display)
# ---------------------------------------------------------------------------

TOOLS = {
    "web_search": {
        "label": "Web Search",
        "icon": "🌐",
        "description": "Search the web using DuckDuckGo and return results.",
    },
    "read_file": {
        "label": "Read File",
        "icon": "📄",
        "description": "Read the contents of a file on the local machine.",
    },
    "write_file": {
        "label": "Write File",
        "icon": "💾",
        "description": "Write content to a file on the local machine.",
    },
    "run_command": {
        "label": "Run Command",
        "icon": "⚡",
        "description": "Execute a shell command and return output (sandboxed).",
    },
    "list_dir": {
        "label": "List Directory",
        "icon": "📁",
        "description": "List files and folders in a directory.",
    },
    "open_browser": {
        "label": "Open Browser",
        "icon": "🌍",
        "description": "Open a URL in the system browser.",
    },
    "screenshot": {
        "label": "Take Screenshot",
        "icon": "📸",
        "description": "Capture the current screen.",
    },
}

# Commands that are blocked for safety
_BLOCKED_COMMANDS = {"rm -rf", "del /f", "format", "shutdown", "reboot", "dd if=", "mkfs"}


class AgentTools:
    """Executes agent tool calls requested by the AI."""

    def __init__(self, pc_control_enabled: bool = False, web_search_enabled: bool = False):
        self.pc_control_enabled = pc_control_enabled
        self.web_search_enabled = web_search_enabled

    def dispatch(self, tool_name: str, params: dict) -> dict:
        """Dispatch a tool call and return result."""
        handlers = {
            "web_search": self._web_search,
            "read_file": self._read_file,
            "write_file": self._write_file,
            "run_command": self._run_command,
            "list_dir": self._list_dir,
            "open_browser": self._open_browser,
        }
        fn = handlers.get(tool_name)
        if fn is None:
            return {"error": f"Unknown tool: {tool_name}"}
        try:
            return fn(params)
        except Exception as e:
            logger.error("Tool %s failed: %s", tool_name, e)
            return {"error": str(e)}

    # ------------------------------------------------------------------
    # Web search
    # ------------------------------------------------------------------

    def _web_search(self, params: dict) -> dict:
        if not self.web_search_enabled:
            return {"error": "Web search is not enabled for this model."}
        query = params.get("query", "")
        if not query:
            return {"error": "No query provided."}
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=5))
            return {"results": results}
        except ImportError:
            return {"error": "duckduckgo_search not installed."}
        except Exception as e:
            return {"error": str(e)}

    # ------------------------------------------------------------------
    # PC control (file system)
    # ------------------------------------------------------------------

    def _read_file(self, params: dict) -> dict:
        if not self.pc_control_enabled:
            return {"error": "PC control is not enabled for this model."}
        path = params.get("path", "")
        if not path or not os.path.exists(path):
            return {"error": f"File not found: {path}"}
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                content = f.read(100_000)
            return {"content": content, "path": path}
        except Exception as e:
            return {"error": str(e)}

    def _write_file(self, params: dict) -> dict:
        if not self.pc_control_enabled:
            return {"error": "PC control is not enabled for this model."}
        path = params.get("path", "")
        content = params.get("content", "")
        if not path:
            return {"error": "No path provided."}
        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            return {"success": True, "path": path}
        except Exception as e:
            return {"error": str(e)}

    def _list_dir(self, params: dict) -> dict:
        if not self.pc_control_enabled:
            return {"error": "PC control is not enabled for this model."}
        path = params.get("path", os.path.expanduser("~"))
        try:
            entries = os.listdir(path)
            return {"path": path, "entries": sorted(entries)[:200]}
        except Exception as e:
            return {"error": str(e)}

    def _run_command(self, params: dict) -> dict:
        if not self.pc_control_enabled:
            return {"error": "PC control is not enabled for this model."}
        cmd = params.get("command", "")
        if not cmd:
            return {"error": "No command provided."}
        # Safety check
        for blocked in _BLOCKED_COMMANDS:
            if blocked in cmd.lower():
                return {"error": f"Blocked command for safety: {cmd}"}
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=30
            )
            return {
                "stdout": result.stdout[:10_000],
                "stderr": result.stderr[:2_000],
                "returncode": result.returncode,
            }
        except subprocess.TimeoutExpired:
            return {"error": "Command timed out (30s)."}
        except Exception as e:
            return {"error": str(e)}

    def _open_browser(self, params: dict) -> dict:
        if not self.pc_control_enabled:
            return {"error": "PC control is not enabled for this model."}
        url = params.get("url", "")
        if not url:
            return {"error": "No URL provided."}
        try:
            import webbrowser
            webbrowser.open(url)
            return {"success": True, "url": url}
        except Exception as e:
            return {"error": str(e)}
