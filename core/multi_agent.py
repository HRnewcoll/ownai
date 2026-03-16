"""Multi-agent framework – Architect → Coder → Reviewer pipeline.

Implements a 3-agent swarm that collaborates via shared state (AgentState).
Each agent specialises:
  - ArchitectAgent  → decomposes task into a detailed plan
  - CoderAgent      → writes code based on the plan
  - ReviewerAgent   → critiques, runs tests, requests re-plan if needed

Communication:
  Agents pass control signals:
    <PLAN_READY>   – Architect signals plan is done
    <CODE_READY>   – Coder signals code is done
    <RE_PLAN>      – Reviewer requests a better plan
    <RE_CODE>      – Reviewer requests code fix
    <APPROVED>     – Reviewer approves the solution
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .reasoning_engine import CodeSandbox, ChainOfThought, ReasoningTrace, ThinkStep

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Agent state – shared across all agents
# ---------------------------------------------------------------------------

@dataclass
class AgentState:
    """Shared mutable state passed between agents."""
    task: str
    session_id: str = "default"
    plan: str = ""
    code: str = ""
    test_code: str = ""
    review: str = ""
    signal: str = ""             # Latest control signal
    iterations: int = 0
    max_iterations: int = 5
    tdd_enabled: bool = True
    history: List[Dict[str, Any]] = field(default_factory=list)
    final_output: str = ""
    success: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def log_step(self, agent: str, content: str, signal: str = ""):
        self.history.append({
            "agent": agent,
            "content": content[:500],
            "signal": signal,
            "ts": time.time(),
            "iteration": self.iterations,
        })

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "plan": self.plan,
            "code": self.code,
            "review": self.review,
            "signal": self.signal,
            "iterations": self.iterations,
            "success": self.success,
            "final_output": self.final_output,
            "history": self.history,
        }


# ---------------------------------------------------------------------------
# Base agent
# ---------------------------------------------------------------------------

class BaseAgent:
    """Abstract base for all swarm agents."""

    name: str = "BaseAgent"
    role_description: str = "Generic agent"

    def __init__(self, sandbox: Optional[CodeSandbox] = None):
        self.sandbox = sandbox or CodeSandbox()
        self.cot = ChainOfThought()

    def run(self, state: AgentState) -> AgentState:
        raise NotImplementedError

    def _emit(self, state: AgentState, content: str, signal: str) -> AgentState:
        state.signal = signal
        state.log_step(self.name, content, signal)
        logger.debug("[%s] signal=%s content=%s", self.name, signal, content[:120])
        return state


# ---------------------------------------------------------------------------
# Architect agent
# ---------------------------------------------------------------------------

class ArchitectAgent(BaseAgent):
    """Decomposes a task into a structured plan.

    Output format (stored in state.plan):
        ## Plan
        1. ...
        2. ...
        ## Acceptance criteria
        - ...
    """
    name = "Architect"
    role_description = "Breaks tasks into clear, testable subtasks"

    def run(self, state: AgentState) -> AgentState:
        task = state.task
        review_feedback = state.review  # May be non-empty on re-plan

        if review_feedback and "<RE_PLAN>" in state.signal:
            plan = self._replan(task, state.code, review_feedback)
        else:
            plan = self._initial_plan(task)

        state.plan = plan
        return self._emit(state, plan, "<PLAN_READY>")

    # ------------------------------------------------------------------
    def _initial_plan(self, task: str) -> str:
        return (
            f"## Architect Plan for: {task}\n\n"
            "### Strategy\n"
            "1. **Understand** – Parse the exact requirements.\n"
            "2. **TDD First** – Write failing tests that define what success looks like.\n"
            "3. **Implement** – Write the minimal code that makes all tests pass.\n"
            "4. **Refine** – Add edge-case handling and documentation.\n"
            "5. **Verify** – Run full test suite; ensure zero failures.\n\n"
            "### Acceptance Criteria\n"
            "- All tests pass (exit code 0)\n"
            "- No hardcoded data that cheats tests\n"
            "- Code handles None/empty input gracefully\n\n"
            "### Suggested Function Signature\n"
            "```python\n"
            "def solution(*args, **kwargs):\n"
            "    ...\n"
            "```\n"
        )

    def _replan(self, task: str, failing_code: str, review: str) -> str:
        return (
            f"## Revised Plan (iteration)\n\n"
            f"**Review feedback:** {review[:400]}\n\n"
            f"**Original task:** {task}\n\n"
            "### Corrective Actions\n"
            "1. Address the specific failures identified by the Reviewer.\n"
            "2. Re-examine edge cases that were missed.\n"
            "3. Ensure tests are not over-fitted to specific inputs.\n"
            "4. Rewrite or patch the failing section.\n"
        )


# ---------------------------------------------------------------------------
# Coder agent (with TDD)
# ---------------------------------------------------------------------------

class CoderAgent(BaseAgent):
    """Writes code based on the Architect's plan.

    When TDD is enabled it generates failing tests first, then the
    implementation, and finally runs both inside the sandbox.
    """
    name = "Coder"
    role_description = "Translates plans into executable, tested code"

    def run(self, state: AgentState) -> AgentState:
        task = state.task
        plan = state.plan

        # Generate tests first (TDD)
        if state.tdd_enabled and not state.test_code:
            state.test_code = ChainOfThought.build_tdd_tests(task)

        # Generate solution code
        code = self._write_solution(task, plan, state.code)
        state.code = code

        # Verify in sandbox
        if state.test_code:
            result = self.sandbox.run_tests(code, state.test_code)
        else:
            result = self.sandbox.run(code)

        obs = self._fmt(result)
        passed = result.get("returncode", -1) == 0
        content = f"Code written.\nSandbox: {obs}"
        signal = "<CODE_READY>" if passed else "<CODE_NEEDS_FIX>"
        return self._emit(state, content, signal)

    # ------------------------------------------------------------------
    def _write_solution(self, task: str, plan: str, prev_code: str) -> str:
        """Generate solution (rule-based stub; LLM fills this in production)."""
        if prev_code and "<CODE_NEEDS_FIX>" not in prev_code:
            # Attempt a trivial improvement
            return prev_code.rstrip() + "\n# Improved by Coder agent\n"

        return (
            f"# OwnAI – Coder agent\n"
            f"# Task: {task}\n"
            f"# Plan steps followed: see architect plan\n\n"
            "def solution(*args, **kwargs):\n"
            "    \"\"\"Auto-generated solution stub.\"\"\"\n"
            "    # TODO: Implement based on architect plan\n"
            "    return None\n\n"
            "if __name__ == '__main__':\n"
            "    print(solution())\n"
        )

    @staticmethod
    def _fmt(result: Dict[str, Any]) -> str:
        if result.get("timed_out"):
            return "TIMED OUT"
        rc = result.get("returncode", -1)
        out = result.get("stdout", "").strip()[:300]
        err = result.get("stderr", "").strip()[:200]
        parts = [f"exit={rc}"]
        if out:
            parts.append(f"stdout={out}")
        if err:
            parts.append(f"stderr={err}")
        return " | ".join(parts)


# ---------------------------------------------------------------------------
# Reviewer agent
# ---------------------------------------------------------------------------

class ReviewerAgent(BaseAgent):
    """Critiques code, runs tests, and signals approval or re-work.

    Signal meanings:
        <APPROVED>  – code is correct and tests pass
        <RE_CODE>   – minor fix needed (same plan)
        <RE_PLAN>   – fundamental issue; architect must re-plan
    """
    name = "Reviewer"
    role_description = "Finds bugs, ensures tests pass, approves or requests rework"

    def run(self, state: AgentState) -> AgentState:
        code = state.code
        test_code = state.test_code

        if not code:
            return self._emit(state, "No code to review.", "<RE_CODE>")

        # Run tests
        if test_code:
            result = self.sandbox.run_tests(code, test_code)
        else:
            result = self.sandbox.run(code)

        passed = result.get("returncode", -1) == 0
        stdout = result.get("stdout", "")
        stderr = result.get("stderr", "")

        review = self._build_review(code, stdout, stderr, passed)
        state.review = review

        if passed:
            state.final_output = code
            state.success = True
            signal = "<APPROVED>"
        elif state.iterations >= state.max_iterations - 1:
            signal = "<APPROVED>"  # Force exit to avoid infinite loop
            state.final_output = code
        elif "SyntaxError" in stderr or "IndentationError" in stderr:
            signal = "<RE_CODE>"
        elif state.iterations > 2:
            signal = "<RE_PLAN>"
        else:
            signal = "<RE_CODE>"

        return self._emit(state, review, signal)

    # ------------------------------------------------------------------
    def _build_review(self, code: str, stdout: str, stderr: str, passed: bool) -> str:
        lines = ["## Reviewer Report"]
        if passed:
            lines.append("✅ **All tests passed.** Code is approved.")
        else:
            lines.append("❌ **Tests failed.** Issues found:")
            if stderr:
                lines.append(f"- Error output: `{stderr[:300]}`")
            if "FAILED" in stdout:
                failed = [l for l in stdout.splitlines() if "❌" in l or "FAILED" in l]
                for f in failed[:5]:
                    lines.append(f"- {f}")
        lines.append(f"\n**Code size:** {len(code)} chars, {code.count(chr(10))} lines")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Multi-agent orchestrator
# ---------------------------------------------------------------------------

class MultiAgentOrchestrator:
    """Coordinates Architect → Coder → Reviewer with loop control.

    Flow:
        Architect (THINK) → Coder (ACT+OBSERVE) → Reviewer (VERIFY)
           ↑                                              |
           └── <RE_PLAN> ─────────────────────────────────┘
           ↑                                              |
           └── <RE_CODE> ──────── back to Coder ─────────┘
    """

    def __init__(
        self,
        tdd_enabled: bool = True,
        max_iterations: int = 5,
        sandbox: Optional[CodeSandbox] = None,
    ):
        self.tdd_enabled = tdd_enabled
        self.max_iterations = max_iterations
        _sb = sandbox or CodeSandbox()
        self.architect = ArchitectAgent(sandbox=_sb)
        self.coder = CoderAgent(sandbox=_sb)
        self.reviewer = ReviewerAgent(sandbox=_sb)

    # ------------------------------------------------------------------
    def solve(
        self,
        task: str,
        session_id: str = "default",
        on_event: Optional[callable] = None,
    ) -> AgentState:
        """Run the multi-agent loop and return final AgentState."""
        state = AgentState(
            task=task,
            session_id=session_id,
            tdd_enabled=self.tdd_enabled,
            max_iterations=self.max_iterations,
        )

        def _notify(agent_name: str, state: AgentState):
            if on_event:
                try:
                    on_event({"agent": agent_name, "signal": state.signal, "state": state.to_dict()})
                except Exception:
                    pass

        for i in range(self.max_iterations):
            state.iterations = i + 1

            # ── ARCHITECT ──────────────────────────────────────────────
            if i == 0 or state.signal == "<RE_PLAN>":
                state = self.architect.run(state)
                _notify("Architect", state)

            # ── CODER ──────────────────────────────────────────────────
            if state.signal in ("<PLAN_READY>", "<RE_CODE>", "<CODE_NEEDS_FIX>"):
                state = self.coder.run(state)
                _notify("Coder", state)

            # ── REVIEWER ───────────────────────────────────────────────
            state = self.reviewer.run(state)
            _notify("Reviewer", state)

            if state.signal == "<APPROVED>":
                logger.info("[MultiAgent] Task approved after %d iterations.", i + 1)
                break

        return state

    # ------------------------------------------------------------------
    def solve_streaming(self, task: str, session_id: str = "default"):
        """Generator version – yields event dicts for real-time UI updates."""
        events: List[Dict[str, Any]] = []

        def _collect(event: Dict):
            events.append(event)

        import threading
        result: List[AgentState] = []

        def _run():
            state = self.solve(task, session_id=session_id, on_event=_collect)
            result.append(state)

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

        seen = 0
        while thread.is_alive() or seen < len(events):
            while seen < len(events):
                yield events[seen]
                seen += 1
            time.sleep(0.05)

        if result:
            yield {"agent": "Orchestrator", "signal": "<DONE>", "state": result[0].to_dict()}

    # ------------------------------------------------------------------
    def run(self, task: str, session_id: str = "default",
            on_event: Optional[callable] = None) -> AgentState:
        """Alias for solve() for API compatibility."""
        return self.solve(task, session_id=session_id, on_event=on_event)
