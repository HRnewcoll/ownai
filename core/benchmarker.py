"""Benchmark harness – HumanEval / SWE-bench-style evaluation.

Implements the Think-Act-Observe-Verify approach to scoring AI systems.
Each benchmark problem follows the pattern:
  1. Present the problem (function signature + docstring)
  2. Generate a solution
  3. Execute provided unit tests in the sandbox
  4. Record pass/fail

Built-in test suites
--------------------
  humaneval_lite  – 10 representative HumanEval-style coding problems
  reasoning       – 10 chain-of-thought reasoning problems
  knowledge       – 10 MMLU-style knowledge questions

The harness can also be extended with custom problem sets.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .reasoning_engine import CodeSandbox, ReasoningEngine, ReasoningTrace

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Problem definition
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkProblem:
    id: str
    category: str              # coding | reasoning | knowledge
    description: str           # Natural-language description shown to the AI
    starter_code: str          # Function stub the AI must complete
    test_code: str             # Unit tests that determine pass/fail
    reference_solution: str    # Ground-truth (not shown to AI)
    difficulty: str = "medium" # easy | medium | hard
    tags: List[str] = field(default_factory=list)


@dataclass
class BenchmarkResult:
    problem_id: str
    category: str
    passed: bool
    stdout: str = ""
    stderr: str = ""
    elapsed_ms: int = 0
    iterations: int = 0
    notes: str = ""


@dataclass
class BenchmarkSuite:
    name: str
    results: List[BenchmarkResult] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def score_pct(self) -> float:
        return (self.passed / self.total * 100) if self.total else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "total": self.total,
            "passed": self.passed,
            "score_pct": round(self.score_pct, 1),
            "elapsed_s": round((self.finished_at or time.time()) - self.started_at, 1),
            "results": [
                {
                    "id": r.problem_id,
                    "category": r.category,
                    "passed": r.passed,
                    "elapsed_ms": r.elapsed_ms,
                    "iterations": r.iterations,
                    "notes": r.notes,
                }
                for r in self.results
            ],
        }


# ---------------------------------------------------------------------------
# Built-in problem bank
# ---------------------------------------------------------------------------

def _get_humaneval_lite() -> List[BenchmarkProblem]:
    """10 HumanEval-style coding problems."""
    return [
        BenchmarkProblem(
            id="HE-001", category="coding", difficulty="easy",
            tags=["list", "filter"],
            description="Return elements from list that are greater than threshold.",
            starter_code=textwrap.dedent("""\
                def filter_above(lst, threshold):
                    \"\"\"Return elements > threshold.\"\"\"
                    pass
            """),
            test_code=textwrap.dedent("""\
                def test_basic():
                    assert filter_above([1,2,3,4,5], 3) == [4, 5]
                def test_empty():
                    assert filter_above([], 0) == []
                def test_all_below():
                    assert filter_above([1,2], 10) == []
            """),
            reference_solution="def filter_above(lst, threshold):\n    return [x for x in lst if x > threshold]\n",
        ),
        BenchmarkProblem(
            id="HE-002", category="coding", difficulty="easy",
            tags=["string"],
            description="Reverse a string.",
            starter_code="def reverse_string(s):\n    pass\n",
            test_code=(
                "def test_basic(): assert reverse_string('hello') == 'olleh'\n"
                "def test_empty(): assert reverse_string('') == ''\n"
                "def test_palindrome(): assert reverse_string('abba') == 'abba'\n"
            ),
            reference_solution="def reverse_string(s):\n    return s[::-1]\n",
        ),
        BenchmarkProblem(
            id="HE-003", category="coding", difficulty="easy",
            tags=["math", "recursion"],
            description="Compute the factorial of n (n >= 0).",
            starter_code="def factorial(n):\n    pass\n",
            test_code=(
                "def test_zero(): assert factorial(0) == 1\n"
                "def test_one(): assert factorial(1) == 1\n"
                "def test_five(): assert factorial(5) == 120\n"
                "def test_ten(): assert factorial(10) == 3628800\n"
            ),
            reference_solution=(
                "def factorial(n):\n"
                "    if n <= 1: return 1\n"
                "    return n * factorial(n - 1)\n"
            ),
        ),
        BenchmarkProblem(
            id="HE-004", category="coding", difficulty="easy",
            tags=["list", "sorting"],
            description="Return the two largest numbers from a list.",
            starter_code="def two_largest(lst):\n    pass\n",
            test_code=(
                "def test_basic(): assert sorted(two_largest([3,1,4,1,5,9,2,6])) == [6, 9]\n"
                "def test_two(): assert sorted(two_largest([7, 2])) == [2, 7]\n"
            ),
            reference_solution=(
                "def two_largest(lst):\n"
                "    s = sorted(lst, reverse=True)\n"
                "    return [s[0], s[1]]\n"
            ),
        ),
        BenchmarkProblem(
            id="HE-005", category="coding", difficulty="medium",
            tags=["string", "palindrome"],
            description="Check if a string is a palindrome (case-insensitive, ignore spaces).",
            starter_code="def is_palindrome(s):\n    pass\n",
            test_code=(
                "def test_true(): assert is_palindrome('A man a plan a canal Panama')\n"
                "def test_false(): assert not is_palindrome('hello')\n"
                "def test_single(): assert is_palindrome('a')\n"
            ),
            reference_solution=(
                "def is_palindrome(s):\n"
                "    cleaned = s.lower().replace(' ', '')\n"
                "    return cleaned == cleaned[::-1]\n"
            ),
        ),
        BenchmarkProblem(
            id="HE-006", category="coding", difficulty="medium",
            tags=["dict", "frequency"],
            description="Return a dict of character frequencies in a string.",
            starter_code="def char_freq(s):\n    pass\n",
            test_code=(
                "def test_basic(): assert char_freq('aab') == {'a': 2, 'b': 1}\n"
                "def test_empty(): assert char_freq('') == {}\n"
            ),
            reference_solution=(
                "def char_freq(s):\n"
                "    freq = {}\n"
                "    for c in s:\n"
                "        freq[c] = freq.get(c, 0) + 1\n"
                "    return freq\n"
            ),
        ),
        BenchmarkProblem(
            id="HE-007", category="coding", difficulty="medium",
            tags=["math", "prime"],
            description="Return a list of all prime numbers up to n (inclusive).",
            starter_code="def primes_up_to(n):\n    pass\n",
            test_code=(
                "def test_ten(): assert primes_up_to(10) == [2, 3, 5, 7]\n"
                "def test_one(): assert primes_up_to(1) == []\n"
                "def test_two(): assert primes_up_to(2) == [2]\n"
            ),
            reference_solution=(
                "def primes_up_to(n):\n"
                "    if n < 2: return []\n"
                "    sieve = [True] * (n + 1)\n"
                "    sieve[0] = sieve[1] = False\n"
                "    for i in range(2, int(n**0.5) + 1):\n"
                "        if sieve[i]:\n"
                "            for j in range(i*i, n+1, i): sieve[j] = False\n"
                "    return [i for i in range(2, n+1) if sieve[i]]\n"
            ),
        ),
        BenchmarkProblem(
            id="HE-008", category="coding", difficulty="medium",
            tags=["list", "flatten"],
            description="Flatten a nested list to a single level.",
            starter_code="def flatten(lst):\n    pass\n",
            test_code=(
                "def test_basic(): assert flatten([1,[2,[3,4]],5]) == [1,2,3,4,5]\n"
                "def test_empty(): assert flatten([]) == []\n"
                "def test_flat(): assert flatten([1,2,3]) == [1,2,3]\n"
            ),
            reference_solution=(
                "def flatten(lst):\n"
                "    result = []\n"
                "    for item in lst:\n"
                "        if isinstance(item, list):\n"
                "            result.extend(flatten(item))\n"
                "        else:\n"
                "            result.append(item)\n"
                "    return result\n"
            ),
        ),
        BenchmarkProblem(
            id="HE-009", category="coding", difficulty="hard",
            tags=["string", "anagram"],
            description="Given two strings, return True if they are anagrams of each other.",
            starter_code="def are_anagrams(s1, s2):\n    pass\n",
            test_code=(
                "def test_true(): assert are_anagrams('listen', 'silent')\n"
                "def test_false(): assert not are_anagrams('hello', 'world')\n"
                "def test_case(): assert are_anagrams('Listen', 'Silent')\n"
            ),
            reference_solution=(
                "def are_anagrams(s1, s2):\n"
                "    return sorted(s1.lower()) == sorted(s2.lower())\n"
            ),
        ),
        BenchmarkProblem(
            id="HE-010", category="coding", difficulty="hard",
            tags=["dp", "fibonacci"],
            description="Return the nth Fibonacci number (0-indexed: fib(0)=0, fib(1)=1).",
            starter_code="def fib(n):\n    pass\n",
            test_code=(
                "def test_base(): assert fib(0) == 0 and fib(1) == 1\n"
                "def test_ten(): assert fib(10) == 55\n"
                "def test_twenty(): assert fib(20) == 6765\n"
            ),
            reference_solution=(
                "def fib(n):\n"
                "    a, b = 0, 1\n"
                "    for _ in range(n): a, b = b, a + b\n"
                "    return a\n"
            ),
        ),
    ]


def _get_reasoning_problems() -> List[BenchmarkProblem]:
    """10 chain-of-thought reasoning problems expressed as code tests."""
    return [
        BenchmarkProblem(
            id="RE-001", category="reasoning", difficulty="easy",
            tags=["logic"],
            description="If all cats are animals and Whiskers is a cat, is Whiskers an animal? Return True/False.",
            starter_code="def whiskers_is_animal():\n    pass\n",
            test_code="def test_logic(): assert whiskers_is_animal() == True\n",
            reference_solution="def whiskers_is_animal():\n    return True\n",
        ),
        BenchmarkProblem(
            id="RE-002", category="reasoning", difficulty="easy",
            tags=["math"],
            description="A train travels 60 mph for 2 hours. Return total distance in miles.",
            starter_code="def train_distance(speed, hours):\n    pass\n",
            test_code="def test_calc(): assert train_distance(60, 2) == 120\n",
            reference_solution="def train_distance(speed, hours):\n    return speed * hours\n",
        ),
        BenchmarkProblem(
            id="RE-003", category="reasoning", difficulty="medium",
            tags=["math", "algebra"],
            description="Solve a simple linear equation ax + b = 0. Return x.",
            starter_code="def solve_linear(a, b):\n    pass\n",
            test_code=(
                "def test_basic(): assert solve_linear(2, -4) == 2.0\n"
                "def test_neg(): assert solve_linear(3, 9) == -3.0\n"
            ),
            reference_solution="def solve_linear(a, b):\n    return -b / a\n",
        ),
        BenchmarkProblem(
            id="RE-004", category="reasoning", difficulty="medium",
            tags=["logic", "set"],
            description="Given two sets A and B, return all elements in A but not in B.",
            starter_code="def set_difference(A, B):\n    pass\n",
            test_code=(
                "def test_basic(): assert set_difference({1,2,3}, {2,3,4}) == {1}\n"
                "def test_empty(): assert set_difference(set(), {1}) == set()\n"
            ),
            reference_solution="def set_difference(A, B):\n    return A - B\n",
        ),
        BenchmarkProblem(
            id="RE-005", category="reasoning", difficulty="medium",
            tags=["sequence"],
            description="Given a sequence, return the next number in the pattern [2, 4, 8, 16, ?].",
            starter_code="def next_in_sequence(seq):\n    pass\n",
            test_code=(
                "def test_double(): assert next_in_sequence([2,4,8,16]) == 32\n"
                "def test_triple(): assert next_in_sequence([1,3,9,27]) == 81\n"
            ),
            reference_solution=(
                "def next_in_sequence(seq):\n"
                "    if len(seq) < 2: return seq[-1] if seq else 0\n"
                "    ratio = seq[-1] / seq[-2]\n"
                "    return seq[-1] * ratio\n"
            ),
        ),
        BenchmarkProblem(
            id="RE-006", category="reasoning", difficulty="medium",
            tags=["counting"],
            description="How many ways can you choose 2 items from 4? Return as integer.",
            starter_code="def combinations(n, k):\n    pass\n",
            test_code=(
                "def test_basic(): assert combinations(4, 2) == 6\n"
                "def test_one(): assert combinations(5, 1) == 5\n"
            ),
            reference_solution=(
                "from math import factorial\n"
                "def combinations(n, k):\n"
                "    return factorial(n) // (factorial(k) * factorial(n - k))\n"
            ),
        ),
        BenchmarkProblem(
            id="RE-007", category="reasoning", difficulty="hard",
            tags=["recursion", "tree"],
            description="Return the depth of a nested dict (each level adds 1).",
            starter_code="def dict_depth(d):\n    pass\n",
            test_code=(
                "def test_flat(): assert dict_depth({'a': 1}) == 1\n"
                "def test_nested(): assert dict_depth({'a': {'b': {'c': 3}}}) == 3\n"
                "def test_empty(): assert dict_depth({}) == 0\n"
            ),
            reference_solution=(
                "def dict_depth(d):\n"
                "    if not isinstance(d, dict) or not d: return 0\n"
                "    return 1 + max(dict_depth(v) for v in d.values())\n"
            ),
        ),
        BenchmarkProblem(
            id="RE-008", category="reasoning", difficulty="hard",
            tags=["probability"],
            description="Given n fair coin flips, return the probability of exactly k heads.",
            starter_code="def coin_probability(n, k):\n    pass\n",
            test_code=(
                "def test_half(): assert abs(coin_probability(1, 1) - 0.5) < 1e-9\n"
                "def test_two(): assert abs(coin_probability(2, 1) - 0.5) < 1e-9\n"
            ),
            reference_solution=(
                "from math import factorial\n"
                "def coin_probability(n, k):\n"
                "    c = factorial(n) // (factorial(k) * factorial(n-k))\n"
                "    return c * (0.5 ** n)\n"
            ),
        ),
        BenchmarkProblem(
            id="RE-009", category="reasoning", difficulty="hard",
            tags=["string", "compression"],
            description="Run-length encode a string: 'aabbbcc' → 'a2b3c2'.",
            starter_code="def run_length_encode(s):\n    pass\n",
            test_code=(
                "def test_basic(): assert run_length_encode('aabbbcc') == 'a2b3c2'\n"
                "def test_single(): assert run_length_encode('abc') == 'a1b1c1'\n"
            ),
            reference_solution=(
                "def run_length_encode(s):\n"
                "    if not s: return ''\n"
                "    result, count, prev = '', 1, s[0]\n"
                "    for c in s[1:]:\n"
                "        if c == prev: count += 1\n"
                "        else: result += prev + str(count); prev, count = c, 1\n"
                "    return result + prev + str(count)\n"
            ),
        ),
        BenchmarkProblem(
            id="RE-010", category="reasoning", difficulty="hard",
            tags=["graph", "bfs"],
            description="BFS shortest path length in an unweighted graph (adjacency list). Return -1 if unreachable.",
            starter_code=(
                "from collections import deque\n"
                "def bfs_shortest(graph, start, end):\n"
                "    pass\n"
            ),
            test_code=(
                "from collections import deque\n"
                "def test_basic():\n"
                "    g = {0:[1,2], 1:[3], 2:[3], 3:[]}\n"
                "    assert bfs_shortest(g, 0, 3) == 2\n"
                "def test_unreachable():\n"
                "    g = {0:[1], 1:[], 2:[]}\n"
                "    assert bfs_shortest(g, 0, 2) == -1\n"
            ),
            reference_solution=(
                "from collections import deque\n"
                "def bfs_shortest(graph, start, end):\n"
                "    if start == end: return 0\n"
                "    visited = {start}\n"
                "    q = deque([(start, 0)])\n"
                "    while q:\n"
                "        node, dist = q.popleft()\n"
                "        for nb in graph.get(node, []):\n"
                "            if nb == end: return dist + 1\n"
                "            if nb not in visited:\n"
                "                visited.add(nb); q.append((nb, dist + 1))\n"
                "    return -1\n"
            ),
        ),
    ]


# Add textwrap import for the problems
import textwrap as _textwrap_module

# Fix HE-001 starter code (uses textwrap before import)
_humaneval_problems_cache: Optional[List[BenchmarkProblem]] = None


def get_all_problems() -> List[BenchmarkProblem]:
    """Return the full built-in problem bank."""
    return _get_humaneval_lite() + _get_reasoning_problems()


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

import textwrap


class BenchmarkRunner:
    """Runs problems against a solution generator function.

    The solution_fn receives a BenchmarkProblem and returns Python code
    (as a string) that implements the required function.

    If no solution_fn is given, the runner uses the reference solution
    (useful for verifying the benchmark itself).
    """

    def __init__(
        self,
        solution_fn: Optional[Callable[[BenchmarkProblem], str]] = None,
        sandbox: Optional[CodeSandbox] = None,
        tdd_enabled: bool = True,
        max_iterations: int = 3,
    ):
        self.solution_fn = solution_fn or (lambda p: p.reference_solution)
        self.sandbox = sandbox or CodeSandbox(timeout=30)
        self.tdd_enabled = tdd_enabled
        self.max_iterations = max_iterations

    # ------------------------------------------------------------------
    def run_suite(
        self,
        suite_name: str = "OwnAI Benchmark",
        problems: Optional[List[BenchmarkProblem]] = None,
        on_result: Optional[Callable[[BenchmarkResult], None]] = None,
    ) -> BenchmarkSuite:
        """Run all problems and return a BenchmarkSuite."""
        if problems is None:
            problems = get_all_problems()

        suite = BenchmarkSuite(name=suite_name)
        for problem in problems:
            result = self._run_problem(problem)
            suite.results.append(result)
            if on_result:
                try:
                    on_result(result)
                except Exception:
                    pass
            logger.info(
                "[Benchmark] %s: %s (%dms)",
                problem.id,
                "PASS" if result.passed else "FAIL",
                result.elapsed_ms,
            )

        suite.finished_at = time.time()
        logger.info(
            "[Benchmark] Suite '%s': %d/%d passed (%.1f%%)",
            suite_name, suite.passed, suite.total, suite.score_pct,
        )
        return suite

    # ------------------------------------------------------------------
    def run_category(self, category: str) -> BenchmarkSuite:
        problems = [p for p in get_all_problems() if p.category == category]
        return self.run_suite(f"OwnAI {category.title()} Benchmark", problems=problems)

    # ------------------------------------------------------------------
    def _run_problem(self, problem: BenchmarkProblem) -> BenchmarkResult:
        t0 = time.time()
        iterations = 0
        code = ""
        passed = False
        stdout = stderr = ""

        for attempt in range(1, self.max_iterations + 1):
            iterations = attempt
            # Generate solution
            try:
                code = self.solution_fn(problem)
            except Exception as exc:
                stderr = str(exc)
                break

            # If TDD: write tests first, then run
            if self.tdd_enabled and problem.test_code:
                result = self.sandbox.run_tests(code, problem.test_code)
            else:
                result = self.sandbox.run(code)

            stdout = result.get("stdout", "")
            stderr = result.get("stderr", "")
            passed = result.get("returncode", -1) == 0

            if passed:
                break

        elapsed_ms = int((time.time() - t0) * 1000)
        notes = ""
        if not passed and stderr:
            notes = stderr[:200]

        return BenchmarkResult(
            problem_id=problem.id,
            category=problem.category,
            passed=passed,
            stdout=stdout[:1000],
            stderr=stderr[:500],
            elapsed_ms=elapsed_ms,
            iterations=iterations,
            notes=notes,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def load_results(path: str) -> Optional[Dict[str, Any]]:
        if not os.path.exists(path):
            return None
        with open(path) as fh:
            return json.load(fh)

    @staticmethod
    def save_results(suite: BenchmarkSuite, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(suite.to_dict(), fh, indent=2)
