#!/usr/bin/env python3
"""OwnAI – Proof-of-Concept: Autonomous Agentic AI from Scratch.

Run this standalone script to see the full Think-Act-Observe-Verify loop in
action.  It works in two modes:

  1. **Ollama mode** (recommended): If Ollama is installed and a model is
     pulled, it will use it for real LLM responses.
     Install: https://ollama.com  → `ollama pull qwen2.5:7b`

  2. **Mock mode** (zero-dependency fallback): Uses smart pattern-matching to
     simulate model responses so you can explore the architecture without a GPU.

Usage::

    python proof_of_concept_ai.py                  # interactive
    python proof_of_concept_ai.py --demo           # run built-in demo problem
    python proof_of_concept_ai.py --model qwen2.5:7b --problem "Write a merge sort"

Architecture demonstrated:
  ┌─────────────────────────────────────────────────────┐
  │              Think-Act-Observe-Verify Loop           │
  │  THINK  →  ACT  →  OBSERVE  →  VERIFY  → (repeat)  │
  └─────────────────────────────────────────────────────┘
     Architect → Coder → Reviewer  (3-agent swarm)
     TDD: write failing tests BEFORE writing the solution
     Safe sandbox: code runs in an isolated subprocess
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_MODEL = "qwen2.5:7b"   # change to any model installed in Ollama
MAX_ROUNDS    = 6               # max Think-Act-Observe-Verify iterations
TIMEOUT_SECS  = 30              # max seconds per sandbox execution
SANDBOX_DIR   = os.path.join(tempfile.gettempdir(), "ownai_poc_sandbox")

# ─────────────────────────────────────────────────────────────────────────────
# Colour helpers (graceful fallback on Windows/no-TTY)
# ─────────────────────────────────────────────────────────────────────────────

_USE_COLOUR = sys.stdout.isatty() and os.name != "nt"


def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOUR else text


def cyan(t):    return _c(t, "96")
def green(t):   return _c(t, "92")
def yellow(t):  return _c(t, "93")
def red(t):     return _c(t, "91")
def bold(t):    return _c(t, "1")
def dim(t):     return _c(t, "2")


def banner(title: str, char: str = "─", width: int = 60):
    line = char * width
    print(f"\n{cyan(line)}")
    print(bold(f"  {title}"))
    print(cyan(line))


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LoopStep:
    """Records one full Think-Act-Observe-Verify cycle."""
    index: int
    agent: str = ""          # "architect" | "coder" | "reviewer"
    plan: str = ""
    test_code: str = ""
    solution_code: str = ""
    observation: str = ""
    test_result: str = ""
    verified: bool = False
    elapsed_ms: int = 0
    signal: str = ""         # PLAN_READY | CODE_READY | APPROVED | RE_CODE | RE_PLAN


@dataclass
class AgentSession:
    """Full session state shared between all three agents."""
    problem: str
    model: str = DEFAULT_MODEL
    use_ollama: bool = False
    steps: List[LoopStep] = field(default_factory=list)
    success: bool = False
    final_answer: str = ""
    round: int = 0

    # Accumulated context between rounds
    plan: str = ""
    test_code: str = ""
    solution: str = ""
    error_log: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Model bridge – Ollama or smart mock
# ─────────────────────────────────────────────────────────────────────────────

def _ollama_available() -> bool:
    try:
        import urllib.request
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
        return True
    except Exception:
        return False


def call_model(prompt: str, model: str = DEFAULT_MODEL, use_ollama: bool = False) -> str:
    """Call the LLM. Uses Ollama if available, else intelligent mock."""
    if use_ollama:
        return _call_ollama(prompt, model)
    return _mock_model(prompt)


def _call_ollama(prompt: str, model: str) -> str:
    """HTTP call to local Ollama server."""
    import urllib.request
    payload = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode()
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode())
            return data.get("response", "")
    except Exception as exc:
        return f"[Ollama error: {exc}]"


def _mock_model(prompt: str) -> str:
    """Smart mock that produces realistic responses without a real model."""
    p = prompt.lower()

    # ── Architect: planning ──────────────────────────────────────────────────
    if "architect" in p or "high-level plan" in p or "decompose" in p:
        return textwrap.dedent("""\
            PLAN:
            1. Understand the problem requirements completely.
            2. Identify edge cases (None, empty input, negative numbers).
            3. Write failing TDD tests that pin down expected behaviour.
            4. Implement the minimal function to pass the tests.
            5. Refactor for clarity and efficiency.
            6. Re-run tests to confirm everything passes.
            <PLAN_READY>
        """)

    # ── TDD test generation ──────────────────────────────────────────────────
    if "write failing tests" in p or "tdd" in p or "test first" in p:
        fn = "solution"
        for kw in ["factorial", "fibonacci", "sort", "reverse", "palindrome",
                   "square_root", "merge", "binary_search"]:
            if kw in p:
                fn = kw
                break
        return textwrap.dedent(f"""\
            Here are the failing TDD tests:

            ```python
            # tests.py  –  written BEFORE the implementation
            def test_basic():
                result = {fn}(5)
                assert result is not None, "Must return a value"

            def test_zero():
                result = {fn}(0)
                assert result is not None

            def test_type():
                result = {fn}(3)
                assert isinstance(result, (int, float, list, str, bool))

            def test_negative():
                try:
                    {fn}(-1)
                except (ValueError, TypeError):
                    pass  # graceful error is fine
            ```
            <CODE_READY>
        """)

    # ── Coder: factorial ─────────────────────────────────────────────────────
    if "factorial" in p:
        return textwrap.dedent("""\
            Here is the implementation:

            ```python
            def factorial(n: int) -> int:
                \"\"\"Return n! for non-negative integers.\"\"\"
                if not isinstance(n, int) or n < 0:
                    raise ValueError(f"factorial requires a non-negative integer, got {n!r}")
                result = 1
                for i in range(2, n + 1):
                    result *= i
                return result

            if __name__ == "__main__":
                for i in [0, 1, 5, 10]:
                    print(f"factorial({i}) = {factorial(i)}")
            ```
            <CODE_READY>
        """)

    # ── Coder: fibonacci ─────────────────────────────────────────────────────
    if "fibonacci" in p:
        return textwrap.dedent("""\
            ```python
            def fibonacci(n: int) -> int:
                \"\"\"Return the nth Fibonacci number (0-indexed).\"\"\"
                if n < 0:
                    raise ValueError("fibonacci requires n >= 0")
                a, b = 0, 1
                for _ in range(n):
                    a, b = b, a + b
                return a

            if __name__ == "__main__":
                print([fibonacci(i) for i in range(10)])
            ```
            <CODE_READY>
        """)

    # ── Coder: merge sort ────────────────────────────────────────────────────
    if "merge sort" in p or "mergesort" in p:
        return textwrap.dedent("""\
            ```python
            def merge_sort(arr: list) -> list:
                if len(arr) <= 1:
                    return arr
                mid = len(arr) // 2
                left  = merge_sort(arr[:mid])
                right = merge_sort(arr[mid:])
                return _merge(left, right)

            def _merge(left: list, right: list) -> list:
                result, i, j = [], 0, 0
                while i < len(left) and j < len(right):
                    if left[i] <= right[j]:
                        result.append(left[i]); i += 1
                    else:
                        result.append(right[j]); j += 1
                return result + left[i:] + right[j:]

            if __name__ == "__main__":
                data = [64, 34, 25, 12, 22, 11, 90]
                print("Sorted:", merge_sort(data))
            ```
            <CODE_READY>
        """)

    # ── Coder: bug-fix (sqrt of negative) ───────────────────────────────────
    if "square root" in p or "sqrt" in p or "negative" in p:
        return textwrap.dedent("""\
            The bug is that `math.sqrt(-x)` is passed a negative value.

            ```python
            import math

            def safe_sqrt(x: float) -> float:
                \"\"\"Return sqrt of the absolute value for negative inputs.\"\"\"
                return math.sqrt(abs(x))

            if __name__ == "__main__":
                for v in [-4, 0, 9, 16]:
                    print(f"safe_sqrt({v}) = {safe_sqrt(v)}")
            ```
            <CODE_READY>
        """)

    # ── Reviewer ─────────────────────────────────────────────────────────────
    if "reviewer" in p or "review" in p or "critique" in p:
        if "error" in p or "failed" in p or "stderr" in p:
            return textwrap.dedent("""\
                REVIEW: The code has issues.
                - The function raised an error during execution.
                - Needs type-checking for the input parameter.
                - Re-implement with proper guards.
                <RE_CODE>
            """)
        return textwrap.dedent("""\
            REVIEW: Code looks correct.
            - All test cases passed.
            - Edge cases handled.
            - Code is readable and efficient.
            <APPROVED>
        """)

    # ── Generic fallback ─────────────────────────────────────────────────────
    return (
        "I understand the task. I will plan carefully, write tests first, "
        "then implement the solution step by step.\n<PLAN_READY>"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Safe code sandbox
# ─────────────────────────────────────────────────────────────────────────────

_BLOCKED = [
    "os.system", "subprocess.call", "subprocess.Popen",
    "shutil.rmtree", "os.remove", "os.unlink",
    "__import__('os').system", "open('/etc", 'open("/etc',
    "socket.connect", "urllib.request.urlopen",
]


def _static_check(code: str) -> Optional[str]:
    """Return an error string if dangerous patterns are detected."""
    for pat in _BLOCKED:
        if pat in code:
            return f"🔒 Blocked pattern: {pat!r}"
    try:
        ast.parse(code)
    except SyntaxError as exc:
        return f"SyntaxError: {exc}"
    return None


def run_in_sandbox(code: str, timeout: int = TIMEOUT_SECS) -> Tuple[str, str, int]:
    """Execute code in a temp directory and return (stdout, stderr, returncode)."""
    err = _static_check(code)
    if err:
        return "", err, -1

    os.makedirs(SANDBOX_DIR, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", dir=SANDBOX_DIR, delete=False, encoding="utf-8"
    ) as fh:
        fh.write(code)
        path = fh.name

    timed_out = False
    try:
        proc = subprocess.Popen(
            [sys.executable, path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            cwd=SANDBOX_DIR, text=True,
        )

        def _kill():
            nonlocal timed_out
            if proc.poll() is None:
                timed_out = True
                proc.kill()

        timer = threading.Timer(timeout, _kill)
        timer.start()
        try:
            stdout, stderr = proc.communicate()
        finally:
            timer.cancel()

        if timed_out:
            return "", "⏰ Execution timed out.", 1
        return stdout[:8000], stderr[:2000], proc.returncode
    except Exception as exc:
        return "", str(exc), -1
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def run_tests_in_sandbox(solution: str, tests: str, timeout: int = TIMEOUT_SECS) -> Tuple[str, str, int]:
    """Combine solution + tests + auto-runner and execute."""
    runner = textwrap.dedent("""\

        # ── auto-runner ──────────────────────────────────────────────────
        import sys as _sys
        _passed = _failed = 0
        _fns = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
        if not _fns:
            print("⚠️  No test functions found (expected names starting with test_)")
            _sys.exit(0)
        for _name, _fn in _fns:
            try:
                _fn()
                _passed += 1
                print(f"  ✅ {_name}")
            except Exception as _e:
                _failed += 1
                print(f"  ❌ {_name}: {_e}")
        print(f"\\n{'─'*40}")
        print(f"PASSED: {_passed}  |  FAILED: {_failed}")
        _sys.exit(0 if _failed == 0 else 1)
    """)
    combined = solution.rstrip() + "\n\n# ── TESTS ──\n" + tests + runner
    return run_in_sandbox(combined, timeout)


# ─────────────────────────────────────────────────────────────────────────────
# Code extractor
# ─────────────────────────────────────────────────────────────────────────────

def extract_code(text: str) -> str:
    """Pull the first ```python ... ``` block from model output."""
    lines = text.splitlines()
    in_block = False
    collected: List[str] = []
    for line in lines:
        stripped = line.strip()
        if not in_block and (stripped.startswith("```python") or stripped == "```python"):
            in_block = True
            collected = []
        elif in_block and stripped.startswith("```"):
            break
        elif in_block:
            collected.append(line)
    if collected:
        return "\n".join(collected)
    # fallback: return raw text if no fenced block found
    return text


def extract_signal(text: str) -> str:
    """Extract control signal like <PLAN_READY>, <APPROVED>, <RE_CODE>."""
    for sig in ("<PLAN_READY>", "<CODE_READY>", "<APPROVED>", "<RE_CODE>", "<RE_PLAN>"):
        if sig in text:
            return sig.strip("<>")
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# Three-agent swarm
# ─────────────────────────────────────────────────────────────────────────────

class ArchitectAgent:
    """Decomposes the problem into a numbered plan."""

    name = "Architect"

    def run(self, session: AgentSession) -> str:
        history = self._format_history(session)
        prompt = f"""\
You are the ARCHITECT in a 3-agent software engineering team.

PROBLEM: {session.problem}

PREVIOUS ATTEMPTS:
{history or "  (none yet – this is round 1)"}

Your job:
- Provide a clear numbered PLAN that the Coder can follow.
- Identify edge cases, input types, and expected outputs.
- If previous code failed, explain what must change.

End your plan with the tag <PLAN_READY>.
"""
        response = call_model(prompt, session.model, session.use_ollama)
        session.plan = response
        return response

    @staticmethod
    def _format_history(session: AgentSession) -> str:
        if not session.steps:
            return ""
        parts = []
        for s in session.steps[-3:]:
            parts.append(
                f"Round {s.index}: "
                f"verified={s.verified}, "
                f"observation={s.observation[:150]!r}"
            )
        return "\n".join(parts)


class CoderAgent:
    """Writes Python code and TDD tests based on the Architect's plan."""

    name = "Coder"

    def run(self, session: AgentSession) -> Tuple[str, str]:
        # ── TDD: generate tests first ──────────────────────────────────────
        test_prompt = f"""\
You are the CODER.  **Write failing tests FIRST** (TDD) before any implementation.

PROBLEM: {session.problem}
PLAN:    {session.plan}

Write 3-5 pytest-style test functions (names starting with test_) that will
FAIL until the correct solution is implemented.  Use assertions.
Wrap the code in a ```python block.
"""
        test_response = call_model(test_prompt, session.model, session.use_ollama)
        session.test_code = extract_code(test_response) or session.test_code

        # ── Implement solution ─────────────────────────────────────────────
        sol_prompt = f"""\
You are the CODER.  Now implement the solution so the tests pass.

PROBLEM:        {session.problem}
PLAN:           {session.plan}
TESTS TO PASS:
{session.test_code}
PREVIOUS ERROR: {session.error_log or "None"}

Provide ONLY the implementation (no test code), wrapped in a ```python block.
Include `if __name__ == '__main__': ...` to demonstrate the output.
End with <CODE_READY>.
"""
        sol_response = call_model(sol_prompt, session.model, session.use_ollama)
        session.solution = extract_code(sol_response) or session.solution
        return session.test_code, session.solution


class ReviewerAgent:
    """Executes code, runs tests, and emits approval or re-work signal."""

    name = "Reviewer"

    def run(self, session: AgentSession) -> Tuple[bool, str, str]:
        """Returns (approved, observation, signal)."""
        if not session.solution.strip():
            return False, "No solution code provided.", "RE_CODE"

        # Run solution standalone first
        stdout, stderr, rc = run_in_sandbox(session.solution)
        run_obs = self._fmt(stdout, stderr, rc)

        # Run tests against solution
        test_out = test_err = ""
        test_rc = -1
        if session.test_code.strip():
            test_out, test_err, test_rc = run_tests_in_sandbox(
                session.solution, session.test_code
            )

        test_obs = self._fmt(test_out, test_err, test_rc) if session.test_code else ""

        # Review prompt
        review_prompt = f"""\
You are the REVIEWER.  Evaluate the solution.

PROBLEM:          {session.problem}
SOLUTION CODE:
{session.solution[:600]}

RUN OUTPUT:
{run_obs[:400]}

TEST RESULTS:
{test_obs[:400] or "(no tests run)"}

If all tests pass and the solution is correct, end with <APPROVED>.
If the code needs fixing, describe exactly what is wrong and end with <RE_CODE>.
If the plan itself is flawed, end with <RE_PLAN>.
"""
        review_response = call_model(review_prompt, session.model, session.use_ollama)
        signal = extract_signal(review_response) or (
            "APPROVED" if (rc == 0 and (not session.test_code or test_rc == 0))
            else "RE_CODE"
        )
        approved = signal == "APPROVED"
        combined_obs = run_obs + ("\n\n" + test_obs if test_obs else "")
        session.error_log = "" if approved else (stderr or test_err)[:500]
        return approved, combined_obs, signal

    @staticmethod
    def _fmt(stdout: str, stderr: str, rc: int) -> str:
        parts = []
        if stdout.strip():
            parts.append(f"STDOUT:\n{stdout.strip()}")
        if stderr.strip():
            parts.append(f"STDERR:\n{stderr.strip()}")
        parts.append(f"Exit: {rc}")
        return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator – runs the full loop
# ─────────────────────────────────────────────────────────────────────────────

def run_agentic_loop(problem: str, model: str = DEFAULT_MODEL, verbose: bool = True,
                     max_rounds: int = MAX_ROUNDS) -> AgentSession:
    """Run the complete Think-Act-Observe-Verify loop.

    Returns the final AgentSession with all history.
    """
    use_ollama = _ollama_available()
    session = AgentSession(problem=problem, model=model, use_ollama=use_ollama)

    architect = ArchitectAgent()
    coder     = CoderAgent()
    reviewer  = ReviewerAgent()

    banner(f"OwnAI – Agentic Loop  |  model: {model}  |  {'Ollama' if use_ollama else 'Mock'}")
    print(f"{bold('Problem:')} {problem}\n")
    if not use_ollama:
        print(dim("  [Ollama not found – using intelligent mock for demonstration]\n"))

    for round_num in range(1, max_rounds + 1):
        session.round = round_num
        t_start = time.time()
        step = LoopStep(index=round_num)

        print(f"\n{yellow(f'━━━  Round {round_num}/{max_rounds}  ━━━')}")

        # ── THINK: Architect plans ────────────────────────────────────────
        print(f"  {cyan('🧠 THINK')}  [{architect.name}] planning…")
        plan_response = architect.run(session)
        step.plan = session.plan
        step.agent = "architect"
        step.signal = extract_signal(plan_response)
        if verbose:
            print(dim(textwrap.indent(session.plan[:300], "    ")))

        # ── ACT: Coder writes tests + solution ────────────────────────────
        print(f"  {cyan('⚡ ACT')}   [{coder.name}] writing tests then code…")
        tests, solution = coder.run(session)
        step.test_code = tests
        step.solution_code = solution
        if verbose and solution:
            print(dim(textwrap.indent(solution[:400], "    ")))

        # ── OBSERVE + VERIFY: Reviewer executes and judges ────────────────
        print(f"  {cyan('👁  OBSERVE')} [{reviewer.name}] running sandbox…")
        approved, observation, signal = reviewer.run(session)
        step.observation = observation
        step.verified = approved
        step.signal = signal
        step.elapsed_ms = int((time.time() - t_start) * 1000)
        session.steps.append(step)

        status_icon = green("✅ APPROVED") if approved else red("❌ " + signal)
        print(f"  {cyan('✔  VERIFY')}  {status_icon}  ({step.elapsed_ms} ms)")
        if verbose:
            print(dim(textwrap.indent(observation[:300], "    ")))

        if approved:
            session.success = True
            session.final_answer = (
                f"✅ Solved in {round_num} round(s)\n\n"
                f"**Solution:**\n```python\n{solution}\n```\n\n"
                f"**Verification:**\n{observation}"
            )
            break

        # Signal routing
        if signal == "RE_PLAN":
            print(dim("    → Reviewer requested a new plan; looping to Architect."))
        elif signal == "RE_CODE":
            print(dim("    → Reviewer requested a code fix; Coder will retry."))
        # Continue loop

    if not session.success:
        session.final_answer = (
            f"⚠️  Reached max rounds ({max_rounds}) without verified solution.\n\n"
            + (f"Best attempt:\n```python\n{session.solution}\n```" if session.solution else "No solution generated.")
        )

    # ── Summary ───────────────────────────────────────────────────────────────
    banner("Results", char="═")
    print(f"{bold('Success:')} {'Yes ✅' if session.success else 'No ❌'}")
    print(f"{bold('Rounds:')}  {session.round}/{max_rounds}")
    print(f"\n{bold('Final Answer:')}")
    print(session.final_answer)
    return session


# ─────────────────────────────────────────────────────────────────────────────
# Built-in demo problems
# ─────────────────────────────────────────────────────────────────────────────

DEMO_PROBLEMS = [
    "Write a Python function factorial(n) that returns n! for non-negative integers.",
    "Write a Python function fibonacci(n) that returns the nth Fibonacci number.",
    "Implement merge sort in Python.",
    "Fix a Python function that incorrectly calculates the square root of a negative number using math.sqrt(-x).",
    "Write a binary search function that returns the index of a target in a sorted list, or -1 if not found.",
]


def interactive_menu() -> str:
    """Show a menu and return the selected problem string."""
    print(bold("\n🤖 OwnAI – Proof of Concept  (Autonomous Agentic AI)\n"))
    print("Choose a demo problem or type your own:\n")
    for i, p in enumerate(DEMO_PROBLEMS, 1):
        print(f"  {cyan(str(i))}. {p}")
    print(f"\n  {cyan('c')}. Enter a custom problem")
    print(f"  {cyan('q')}. Quit\n")

    while True:
        choice = input("Your choice: ").strip().lower()
        if choice == "q":
            print("Goodbye!")
            sys.exit(0)
        if choice == "c":
            problem = input("Enter your problem: ").strip()
            if problem:
                return problem
            print("Please enter a non-empty problem.")
            continue
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(DEMO_PROBLEMS):
                return DEMO_PROBLEMS[idx]
        except ValueError:
            pass
        print("Invalid choice. Try again.")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="OwnAI Proof-of-Concept – autonomous agentic AI loop"
    )
    parser.add_argument("--demo", action="store_true",
                        help="Run the first built-in demo problem automatically")
    parser.add_argument("--problem", type=str, default="",
                        help="Problem to solve (skip interactive menu)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"Ollama model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--rounds", type=int, default=MAX_ROUNDS,
                        help=f"Max reasoning rounds (default: {MAX_ROUNDS})")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress verbose step output")
    args = parser.parse_args()

    if args.demo:
        problem = DEMO_PROBLEMS[0]
    elif args.problem:
        problem = args.problem
    else:
        problem = interactive_menu()

    session = run_agentic_loop(
        problem=problem,
        model=args.model,
        verbose=not args.quiet,
        max_rounds=args.rounds,
    )

    # Save trace to JSON for inspection
    trace_path = os.path.join(tempfile.gettempdir(), "ownai_poc_trace.json")
    with open(trace_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "problem": session.problem,
                "model": session.model,
                "use_ollama": session.use_ollama,
                "success": session.success,
                "rounds": session.round,
                "final_answer": session.final_answer,
                "steps": [
                    {
                        "index": s.index,
                        "agent": s.agent,
                        "plan": s.plan[:300],
                        "solution_code": s.solution_code[:300],
                        "observation": s.observation[:300],
                        "verified": s.verified,
                        "signal": s.signal,
                        "elapsed_ms": s.elapsed_ms,
                    }
                    for s in session.steps
                ],
            },
            fh,
            indent=2,
        )
    print(f"\n{dim(f'Trace saved to: {trace_path}')}")


if __name__ == "__main__":
    main()
