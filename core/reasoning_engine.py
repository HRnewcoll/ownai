"""Reasoning engine – Think-Act-Observe-Verify agentic loop with Chain-of-Thought.

Implements the core cognitive loop described in the project brief:
  THINK   → generate a structured plan / scratchpad
  ACT     → produce code or tool call
  OBSERVE → execute in a sandboxed subprocess and capture output/errors
  VERIFY  → run the test suite or assertion; loop back if failing
  TDD     → write failing tests BEFORE writing the solution
"""
from __future__ import annotations

import ast
import json
import logging
import os
import resource
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Safety constants
# ---------------------------------------------------------------------------

_TIMEOUT_SECONDS = 60
_MAX_OUTPUT_CHARS = 20_000
_BLOCKED_PATTERNS = [
    "import os; os.system",
    "subprocess.call",
    "__import__('os').system",
    "shutil.rmtree",
    "os.remove",
    "os.unlink",
    "open('/etc/",
    "open(\"/etc/",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ThinkStep:
    """One complete Think-Act-Observe-Verify cycle."""
    step_index: int
    plan: str = ""
    code: str = ""
    test_code: str = ""
    observation: str = ""
    test_result: str = ""
    verified: bool = False
    error: Optional[str] = None
    elapsed_ms: int = 0


@dataclass
class ReasoningTrace:
    """Full reasoning trace for a single task."""
    task: str
    steps: List[ThinkStep] = field(default_factory=list)
    final_answer: str = ""
    success: bool = False
    total_steps: int = 0
    tdd_enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "success": self.success,
            "total_steps": self.total_steps,
            "tdd_enabled": self.tdd_enabled,
            "final_answer": self.final_answer,
            "steps": [
                {
                    "step": s.step_index,
                    "plan": s.plan,
                    "code": s.code,
                    "test_code": s.test_code,
                    "observation": s.observation,
                    "test_result": s.test_result,
                    "verified": s.verified,
                    "error": s.error,
                    "elapsed_ms": s.elapsed_ms,
                }
                for s in self.steps
            ],
        }


# ---------------------------------------------------------------------------
# Sandbox execution
# ---------------------------------------------------------------------------

class CodeSandbox:
    """Executes Python code in a restricted subprocess.

    Safety measures:
    - Working directory restricted to a temp folder
    - Hard timeout enforced by a kill thread
    - Blocked dangerous import patterns checked at source level
    - stdout/stderr captured and truncated
    """

    def __init__(self, timeout: int = _TIMEOUT_SECONDS, workspace: Optional[str] = None):
        self.timeout = timeout
        self.workspace = workspace or tempfile.mkdtemp(prefix="ownai_sandbox_")

    # ------------------------------------------------------------------
    def run(self, code: str, extra_files: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Run code string and return result dict."""
        # Static safety scan
        safety_error = self._static_check(code)
        if safety_error:
            return {"stdout": "", "stderr": safety_error, "returncode": -1, "timed_out": False}

        with tempfile.TemporaryDirectory(prefix="ownai_exec_") as tmpdir:
            # Write helper files
            if extra_files:
                for fname, content in extra_files.items():
                    with open(os.path.join(tmpdir, fname), "w", encoding="utf-8") as fh:
                        fh.write(content)
            # Write main script
            script_path = os.path.join(tmpdir, "solution.py")
            with open(script_path, "w", encoding="utf-8") as fh:
                fh.write(code)

            t0 = time.time()
            timed_out = False
            try:
                proc = subprocess.Popen(
                    [sys.executable, script_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=tmpdir,
                    text=True,
                )
                # Kill thread
                def _kill():
                    nonlocal timed_out
                    if proc.poll() is None:
                        timed_out = True
                        proc.kill()

                timer = threading.Timer(self.timeout, _kill)
                timer.start()
                try:
                    stdout, stderr = proc.communicate()
                finally:
                    timer.cancel()

                elapsed_ms = int((time.time() - t0) * 1000)
                return {
                    "stdout": stdout[:_MAX_OUTPUT_CHARS],
                    "stderr": stderr[:4_000],
                    "returncode": proc.returncode,
                    "timed_out": timed_out,
                    "elapsed_ms": elapsed_ms,
                }
            except Exception as exc:
                return {
                    "stdout": "",
                    "stderr": str(exc),
                    "returncode": -1,
                    "timed_out": False,
                    "elapsed_ms": int((time.time() - t0) * 1000),
                }

    # ------------------------------------------------------------------
    def run_tests(self, solution_code: str, test_code: str) -> Dict[str, Any]:
        """Run pytest-style tests against a solution module."""
        combined = solution_code.rstrip() + "\n\n# --- TESTS ---\n" + test_code
        combined += textwrap.dedent("""

            # --- auto-runner ---
            import sys as _sys
            _passed = _failed = 0
            for _name, _fn in list(globals().items()):
                if _name.startswith('test_') and callable(_fn):
                    try:
                        _fn()
                        _passed += 1
                        print(f'  ✅ {_name}')
                    except Exception as _e:
                        _failed += 1
                        print(f'  ❌ {_name}: {_e}')
            print(f'\\nPASSED: {_passed}  FAILED: {_failed}')
            _sys.exit(0 if _failed == 0 else 1)
        """)
        return self.run(combined)

    # ------------------------------------------------------------------
    @staticmethod
    def _static_check(code: str) -> Optional[str]:
        """Return an error string if dangerous patterns are found."""
        code_lower = code.lower()
        for pattern in _BLOCKED_PATTERNS:
            if pattern.lower() in code_lower:
                return f"🔒 Blocked: code contains unsafe pattern '{pattern}'"
        # Try AST parse to catch syntax errors early
        try:
            ast.parse(code)
        except SyntaxError as exc:
            return f"SyntaxError: {exc}"
        return None


# ---------------------------------------------------------------------------
# Chain-of-Thought builder
# ---------------------------------------------------------------------------

class ChainOfThought:
    """Builds structured CoT plans and extracts code blocks from LLM text."""

    PLAN_TEMPLATE = textwrap.dedent("""\
        ## 🧠 THINK: Planning step {step}

        Task: {task}

        Previous context:
        {context}

        Plan:
        1. Understand what is being asked
        2. Identify what the function/solution must do
        3. Consider edge cases
        4. Write failing tests FIRST (TDD)
        5. Implement the solution
        6. Verify tests pass
    """)

    # ------------------------------------------------------------------
    @staticmethod
    def extract_code_blocks(text: str) -> List[str]:
        """Extract ```python ... ``` code blocks from LLM output."""
        blocks: List[str] = []
        in_block = False
        current: List[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("```") and not in_block:
                in_block = True
                current = []
            elif stripped == "```" and in_block:
                in_block = False
                if current:
                    blocks.append("\n".join(current))
                current = []
            elif in_block:
                current.append(line)
        return blocks

    # ------------------------------------------------------------------
    @staticmethod
    def build_tdd_tests(task: str, function_signature: str = "") -> str:
        """Generate skeleton TDD tests for a task."""
        fn_name = "solution"
        if "def " in function_signature:
            fn_name = function_signature.split("def ")[1].split("(")[0].strip()

        return textwrap.dedent(f"""\
            # TDD: Failing tests written BEFORE implementation
            # Task: {task}

            def test_basic_case():
                # TODO: fill in expected values for your task
                result = {fn_name}()
                assert result is not None, "Should return a value"

            def test_edge_case_empty():
                # Edge case: empty/None input
                try:
                    result = {fn_name}(None)
                    assert result is not None
                except (TypeError, ValueError):
                    pass  # Graceful error handling is acceptable

            def test_correctness():
                # Replace with task-specific assertion
                assert True, "Placeholder – add real assertions"
        """)

    # ------------------------------------------------------------------
    @staticmethod
    def format_scratchpad(steps: List[ThinkStep]) -> str:
        """Format the reasoning trace as a human-readable scratchpad."""
        lines: List[str] = ["# 🧠 Reasoning Scratchpad\n"]
        for step in steps:
            lines.append(f"## Step {step.step_index}")
            if step.plan:
                lines.append(f"**Plan:**\n{step.plan}")
            if step.code:
                lines.append(f"**Code:**\n```python\n{step.code}\n```")
            if step.test_code:
                lines.append(f"**Tests:**\n```python\n{step.test_code}\n```")
            if step.observation:
                lines.append(f"**Observation:**\n{step.observation}")
            if step.test_result:
                lines.append(f"**Test result:**\n{step.test_result}")
            status = "✅ Verified" if step.verified else "🔄 Needs fix"
            lines.append(f"**Status:** {status}\n")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Think-Act-Observe-Verify loop
# ---------------------------------------------------------------------------

class ReasoningEngine:
    """Orchestrates the TAOV loop.

    Usage::

        engine = ReasoningEngine(tdd_enabled=True)
        trace = engine.solve(
            task="Write a factorial function",
            initial_code="",         # optional seed code
            max_steps=5,
        )
        print(trace.final_answer)
    """

    def __init__(
        self,
        tdd_enabled: bool = True,
        max_steps: int = 5,
        sandbox: Optional[CodeSandbox] = None,
    ):
        self.tdd_enabled = tdd_enabled
        self.max_steps = max_steps
        self.sandbox = sandbox or CodeSandbox()
        self.cot = ChainOfThought()

    # ------------------------------------------------------------------
    def solve(
        self,
        task: str,
        initial_code: str = "",
        hint_tests: str = "",
        on_step: Optional[callable] = None,
    ) -> ReasoningTrace:
        """Run the full TAOV loop.

        Args:
            task: Natural-language description of the problem.
            initial_code: Optional seed code from the model.
            hint_tests: Pre-written tests if the caller supplies them.
            on_step: Callback(ThinkStep) called after each iteration.

        Returns:
            ReasoningTrace with full history.
        """
        trace = ReasoningTrace(task=task, tdd_enabled=self.tdd_enabled)
        current_code = initial_code
        current_tests = hint_tests

        for step_idx in range(1, self.max_steps + 1):
            t0 = time.time()
            step = ThinkStep(step_index=step_idx)

            # THINK --------------------------------------------------------
            context = self._build_context(trace.steps)
            step.plan = self._think(task, context, step_idx)

            # TDD: generate tests before solution on first step
            if self.tdd_enabled and step_idx == 1 and not current_tests:
                step.test_code = self.cot.build_tdd_tests(task)
            elif current_tests:
                step.test_code = current_tests

            # ACT ----------------------------------------------------------
            if not current_code:
                current_code = self._generate_placeholder_solution(task)
            step.code = current_code

            # OBSERVE -------------------------------------------------------
            exec_result = self.sandbox.run(current_code)
            step.observation = self._format_observation(exec_result)

            # VERIFY --------------------------------------------------------
            if step.test_code:
                test_result = self.sandbox.run_tests(current_code, step.test_code)
                step.test_result = self._format_observation(test_result)
                step.verified = test_result.get("returncode", -1) == 0
            else:
                # No tests: verified if code ran without error
                step.verified = exec_result.get("returncode", -1) == 0

            step.elapsed_ms = int((time.time() - t0) * 1000)
            trace.steps.append(step)

            if on_step:
                try:
                    on_step(step)
                except Exception:
                    pass

            if step.verified:
                trace.success = True
                trace.final_answer = self._build_answer(task, step)
                break
            else:
                # Attempt a fix on the next round
                current_code = self._attempt_fix(current_code, step.observation)

        trace.total_steps = len(trace.steps)
        if not trace.final_answer:
            last = trace.steps[-1] if trace.steps else None
            trace.final_answer = (
                self._build_answer(task, last)
                if last
                else f"Could not solve '{task}' within {self.max_steps} steps."
            )
        return trace

    # ------------------------------------------------------------------
    # Helpers (these are rule-based stubs; in production they call the LLM)
    # ------------------------------------------------------------------

    @staticmethod
    def _think(task: str, context: str, step_idx: int) -> str:
        return (
            f"Step {step_idx}: Analyse the task '{task}'. "
            f"Context from previous steps: {context[:200] or 'None'}. "
            "Plan: understand requirements → write tests → implement → verify."
        )

    @staticmethod
    def _build_context(steps: List[ThinkStep]) -> str:
        if not steps:
            return ""
        last = steps[-1]
        return f"Last observation: {last.observation[:300]}. Verified: {last.verified}."

    @staticmethod
    def _generate_placeholder_solution(task: str) -> str:
        """Generate a minimal stub so the loop can execute."""
        return textwrap.dedent(f"""\
            # Auto-generated stub for: {task}
            # Replace with real implementation

            def solution(*args, **kwargs):
                \"\"\"Placeholder – implement the solution here.\"\"\"
                return None

            if __name__ == '__main__':
                print(solution())
        """)

    @staticmethod
    def _attempt_fix(code: str, observation: str) -> str:
        """Minimal fix: append error context as comment for next attempt."""
        fix_note = f"\n# FIX NEEDED based on: {observation[:200]}\n"
        return code.rstrip() + fix_note

    @staticmethod
    def _format_observation(result: Dict[str, Any]) -> str:
        if result.get("timed_out"):
            return "⏰ TIMED OUT – code ran for too long"
        stdout = result.get("stdout", "").strip()
        stderr = result.get("stderr", "").strip()
        rc = result.get("returncode", -1)
        parts = []
        if stdout:
            parts.append(f"STDOUT:\n{stdout}")
        if stderr:
            parts.append(f"STDERR:\n{stderr}")
        parts.append(f"Exit code: {rc}")
        return "\n".join(parts)

    @staticmethod
    def _build_answer(task: str, step: ThinkStep) -> str:
        if step.verified:
            return (
                f"✅ Task solved: **{task}**\n\n"
                f"**Solution:**\n```python\n{step.code}\n```\n\n"
                f"**Verification:** {step.test_result or step.observation}"
            )
        return (
            f"⚠️ Best attempt for: **{task}**\n\n"
            f"```python\n{step.code}\n```\n\n"
            f"Last observation:\n{step.observation}"
        )
