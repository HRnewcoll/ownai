"""Memory manager – short-term conversation buffer + long-term semantic memory.

Architecture
------------
  Short-term (working memory)
    - Fixed-size in-memory deque of recent messages per session.
    - Evicted oldest messages when capacity is exceeded.

  Long-term (episodic memory)
    - SQLite store for persisted memory entries.
    - Optional vector similarity search using pure-numpy cosine similarity
      (no external vector DB required) or sentence-transformers when available.
    - Stores successful trajectories, learned facts, and conversation summaries.

  Graph-RAG for code
    - Parses Python source files using the `ast` module (no tree-sitter dep).
    - Builds a dependency map: function/class → files that define/call it.
    - Supports "what calls function X" and "where is class Y defined" queries.
"""
from __future__ import annotations

import ast
import json
import logging
import math
import os
import re
import sqlite3
import tempfile
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MemoryEntry:
    id: Optional[int]
    session_id: str
    role: str           # "user" | "assistant" | "system" | "fact" | "trajectory"
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    created_at: float = field(default_factory=time.time)
    importance: float = 1.0   # 0.0–1.0; higher = more important to keep


# ---------------------------------------------------------------------------
# Embedding helper (no external dep required)
# ---------------------------------------------------------------------------

class SimpleEmbedder:
    """Lightweight TF-IDF-style bag-of-words embedder.

    Falls back gracefully when sentence-transformers is unavailable.
    Not as good as dense embeddings but zero-dependency.
    """
    _VOCAB_SIZE = 1024

    def __init__(self):
        self._use_st = False
        self._st_model = None
        try:
            from sentence_transformers import SentenceTransformer
            self._st_model = SentenceTransformer("all-MiniLM-L6-v2")
            self._use_st = True
            logger.info("[Memory] Using sentence-transformers for embeddings.")
        except Exception:
            logger.info("[Memory] sentence-transformers unavailable – using bag-of-words.")

    def embed(self, text: str) -> List[float]:
        if self._use_st and self._st_model is not None:
            vec = self._st_model.encode(text, show_progress_bar=False)
            return vec.tolist()
        return self._bow_embed(text)

    def _bow_embed(self, text: str) -> List[float]:
        tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
        vec = [0.0] * self._VOCAB_SIZE
        for tok in tokens:
            h = hash(tok) % self._VOCAB_SIZE
            vec[h] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    @staticmethod
    def cosine_similarity(a: List[float], b: List[float]) -> float:
        if len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a)) or 1.0
        nb = math.sqrt(sum(y * y for y in b)) or 1.0
        return dot / (na * nb)


# ---------------------------------------------------------------------------
# Short-term (working) memory
# ---------------------------------------------------------------------------

class WorkingMemory:
    """Fixed-capacity ring buffer per session for recent messages."""

    def __init__(self, capacity: int = 20):
        self._capacity = capacity
        self._buffers: Dict[str, deque] = {}

    def add(self, session_id: str, role: str, content: str, metadata: Optional[Dict] = None):
        if session_id not in self._buffers:
            self._buffers[session_id] = deque(maxlen=self._capacity)
        self._buffers[session_id].append({
            "role": role,
            "content": content,
            "metadata": metadata or {},
            "ts": time.time(),
        })

    def get(self, session_id: str) -> List[Dict[str, Any]]:
        return list(self._buffers.get(session_id, []))

    def clear(self, session_id: str):
        if session_id in self._buffers:
            del self._buffers[session_id]

    def to_prompt_context(self, session_id: str, max_chars: int = 4000) -> str:
        messages = self.get(session_id)
        parts: List[str] = []
        total = 0
        for msg in reversed(messages):
            line = f"{msg['role'].upper()}: {msg['content']}"
            if total + len(line) > max_chars:
                break
            parts.insert(0, line)
            total += len(line)
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Long-term (episodic) memory with SQLite + vector similarity
# ---------------------------------------------------------------------------

class LongTermMemory:
    """Persistent memory store backed by SQLite."""

    def __init__(self, db_path: str, embedder: Optional[SimpleEmbedder] = None):
        self.db_path = db_path
        self.embedder = embedder or SimpleEmbedder()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_db()

    # ------------------------------------------------------------------
    def _init_db(self):
        cur = self._conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT DEFAULT '{}',
                embedding TEXT,
                importance REAL DEFAULT 1.0,
                created_at REAL NOT NULL
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_session ON memory(session_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_role ON memory(role)")
        self._conn.commit()

    # ------------------------------------------------------------------
    def store(self, entry: MemoryEntry) -> int:
        if entry.embedding is None and entry.content:
            try:
                entry.embedding = self.embedder.embed(entry.content)
            except Exception:
                entry.embedding = None

        cur = self._conn.cursor()
        cur.execute(
            """INSERT INTO memory (session_id, role, content, metadata, embedding, importance, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                entry.session_id,
                entry.role,
                entry.content,
                json.dumps(entry.metadata),
                json.dumps(entry.embedding) if entry.embedding else None,
                entry.importance,
                entry.created_at,
            ),
        )
        self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    # ------------------------------------------------------------------
    def retrieve_similar(
        self,
        query: str,
        session_id: Optional[str] = None,
        top_k: int = 5,
        min_similarity: float = 0.15,
        role_filter: Optional[str] = None,
    ) -> List[MemoryEntry]:
        """Return top-k entries most similar to query using cosine similarity."""
        query_emb = self.embedder.embed(query)

        where = []
        params: List[Any] = []
        if session_id:
            where.append("session_id = ?")
            params.append(session_id)
        if role_filter:
            where.append("role = ?")
            params.append(role_filter)
        where_clause = ("WHERE " + " AND ".join(where)) if where else ""

        cur = self._conn.cursor()
        cur.execute(
            f"SELECT id, session_id, role, content, metadata, embedding, importance, created_at "
            f"FROM memory {where_clause} ORDER BY created_at DESC LIMIT 500",
            params,
        )
        rows = cur.fetchall()

        scored: List[Tuple[float, MemoryEntry]] = []
        for row in rows:
            rid, sid, role, content, meta_str, emb_str, importance, created_at = row
            try:
                emb = json.loads(emb_str) if emb_str else None
            except Exception:
                emb = None
            sim = 0.0
            if emb:
                sim = SimpleEmbedder.cosine_similarity(query_emb, emb)
            entry = MemoryEntry(
                id=rid, session_id=sid, role=role, content=content,
                metadata=json.loads(meta_str or "{}"),
                embedding=emb, importance=importance, created_at=created_at,
            )
            scored.append((sim, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for sim, e in scored if sim >= min_similarity][:top_k]

    # ------------------------------------------------------------------
    def retrieve_recent(
        self,
        session_id: str,
        limit: int = 10,
        role_filter: Optional[str] = None,
    ) -> List[MemoryEntry]:
        where = ["session_id = ?"]
        params: List[Any] = [session_id]
        if role_filter:
            where.append("role = ?")
            params.append(role_filter)
        where_clause = " AND ".join(where)
        cur = self._conn.cursor()
        cur.execute(
            f"SELECT id, session_id, role, content, metadata, embedding, importance, created_at "
            f"FROM memory WHERE {where_clause} ORDER BY created_at DESC LIMIT ?",
            params + [limit],
        )
        rows = cur.fetchall()
        return [
            MemoryEntry(
                id=r[0], session_id=r[1], role=r[2], content=r[3],
                metadata=json.loads(r[4] or "{}"), embedding=None,
                importance=r[6], created_at=r[7],
            )
            for r in rows
        ]

    # ------------------------------------------------------------------
    def store_trajectory(self, session_id: str, task: str, solution: str, success: bool):
        """Persist a successful problem-solving trajectory for future retrieval."""
        entry = MemoryEntry(
            id=None,
            session_id=session_id,
            role="trajectory",
            content=f"TASK: {task}\nSOLUTION:\n{solution}",
            metadata={"task": task, "success": success},
            importance=1.0 if success else 0.3,
        )
        self.store(entry)

    # ------------------------------------------------------------------
    def forget_old(self, max_entries: int = 10_000):
        """Evict least-important old entries if store grows too large."""
        cur = self._conn.cursor()
        cur.execute("SELECT COUNT(*) FROM memory")
        count = cur.fetchone()[0]
        if count > max_entries:
            to_delete = count - max_entries
            cur.execute(
                "DELETE FROM memory WHERE id IN "
                "(SELECT id FROM memory ORDER BY importance ASC, created_at ASC LIMIT ?)",
                (to_delete,),
            )
            self._conn.commit()


# ---------------------------------------------------------------------------
# Graph-RAG: Python codebase dependency map
# ---------------------------------------------------------------------------

class CodeGraphRAG:
    """Parses Python source files and builds a function/class dependency graph.

    Answers queries like:
      - "which files define function calculate_loss?"
      - "what does module trainer.py export?"
      - "what calls get_model?"
    """

    def __init__(self):
        # symbol → {"defined_in": [...], "called_in": [...]}
        self._graph: Dict[str, Dict[str, List[str]]] = {}
        self._indexed_files: List[str] = []

    # ------------------------------------------------------------------
    def index_directory(self, root_dir: str, recursive: bool = True) -> int:
        """Walk directory, parse .py files, and build graph. Returns file count."""
        count = 0
        for dirpath, _, filenames in os.walk(root_dir):
            for fname in filenames:
                if fname.endswith(".py"):
                    fpath = os.path.join(dirpath, fname)
                    self._index_file(fpath)
                    count += 1
            if not recursive:
                break
        return count

    # ------------------------------------------------------------------
    def _index_file(self, filepath: str):
        try:
            with open(filepath, encoding="utf-8", errors="replace") as fh:
                source = fh.read()
            tree = ast.parse(source, filename=filepath)
        except Exception:
            return

        rel = os.path.relpath(filepath)
        if filepath not in self._indexed_files:
            self._indexed_files.append(filepath)

        for node in ast.walk(tree):
            # Definitions
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = node.name
                self._graph.setdefault(name, {"defined_in": [], "called_in": []})
                if rel not in self._graph[name]["defined_in"]:
                    self._graph[name]["defined_in"].append(rel)
            # Call sites
            elif isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name):
                    name = func.id
                elif isinstance(func, ast.Attribute):
                    name = func.attr
                else:
                    continue
                self._graph.setdefault(name, {"defined_in": [], "called_in": []})
                if rel not in self._graph[name]["called_in"]:
                    self._graph[name]["called_in"].append(rel)

    # ------------------------------------------------------------------
    def query(self, symbol: str) -> Dict[str, Any]:
        """Look up a symbol and return its graph node."""
        return self._graph.get(symbol, {"defined_in": [], "called_in": []})

    # ------------------------------------------------------------------
    def find_related(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Find symbols matching a text query (substring search)."""
        query_lower = query_text.lower()
        results = []
        for symbol, info in self._graph.items():
            if query_lower in symbol.lower():
                results.append({
                    "symbol": symbol,
                    "defined_in": info["defined_in"][:3],
                    "called_in": info["called_in"][:3],
                })
        return results[:top_k]

    # ------------------------------------------------------------------
    def to_summary(self) -> Dict[str, Any]:
        return {
            "indexed_files": len(self._indexed_files),
            "unique_symbols": len(self._graph),
            "top_symbols": sorted(
                self._graph.keys(),
                key=lambda s: len(self._graph[s]["called_in"]),
                reverse=True,
            )[:20],
        }


# ---------------------------------------------------------------------------
# Unified memory manager (facade)
# ---------------------------------------------------------------------------

class MemoryManager:
    """Top-level interface that combines working memory, long-term memory, and code RAG."""

    def __init__(self, db_path: Optional[str] = None, working_capacity: int = 20):
        if db_path is None:
            data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
            os.makedirs(data_dir, exist_ok=True)
            db_path = os.path.join(data_dir, "memory.db")
        self.db_path = db_path
        self.working = WorkingMemory(capacity=working_capacity)
        self.long_term = LongTermMemory(db_path)
        self.code_rag = CodeGraphRAG()
        self._embedder = self.long_term.embedder

    # ------------------------------------------------------------------
    def add_message(self, session_id: str, role: str, content: str, persist: bool = True):
        """Add a message to working memory and optionally long-term memory."""
        self.working.add(session_id, role, content)
        if persist:
            entry = MemoryEntry(
                id=None, session_id=session_id, role=role, content=content,
                importance=0.5,
            )
            self.long_term.store(entry)

    # ------------------------------------------------------------------
    def add_fact(self, session_id: str, fact: str, importance: float = 0.9):
        """Store a factual memory (higher importance)."""
        entry = MemoryEntry(
            id=None, session_id=session_id, role="fact", content=fact,
            importance=importance,
        )
        self.long_term.store(entry)

    # ------------------------------------------------------------------
    def get_context(self, session_id: str, query: str = "", max_chars: int = 3000) -> str:
        """Return a combined context string for injecting into prompts."""
        # Short-term
        short = self.working.to_prompt_context(session_id, max_chars=max_chars // 2)

        # Long-term semantic search
        long_parts: List[str] = []
        if query:
            similar = self.long_term.retrieve_similar(
                query, session_id=session_id, top_k=3
            )
            for entry in similar:
                long_parts.append(f"[Memory] {entry.content[:300]}")

        long = "\n".join(long_parts)
        if short and long:
            return short + "\n\n--- Relevant memories ---\n" + long
        return short or long

    # ------------------------------------------------------------------
    def recall_trajectories(self, task: str, top_k: int = 3) -> List[str]:
        """Retrieve similar past problem-solving trajectories."""
        entries = self.long_term.retrieve_similar(
            task, top_k=top_k, role_filter="trajectory"
        )
        return [e.content for e in entries]

    # ------------------------------------------------------------------
    def save_trajectory(self, session_id: str, task: str, solution: str, success: bool):
        self.long_term.store_trajectory(session_id, task, solution, success)

    # ------------------------------------------------------------------
    def index_codebase(self, root_dir: str) -> Dict[str, Any]:
        """Index a codebase directory for Graph-RAG queries."""
        count = self.code_rag.index_directory(root_dir)
        summary = self.code_rag.to_summary()
        summary["files_indexed"] = count
        return summary

    # ------------------------------------------------------------------
    def code_query(self, symbol_or_query: str) -> Dict[str, Any]:
        """Query the code graph."""
        direct = self.code_rag.query(symbol_or_query)
        related = self.code_rag.find_related(symbol_or_query)
        return {"direct": direct, "related": related}

    # ------------------------------------------------------------------
    def get_stats(self) -> Dict[str, Any]:
        cur = self.long_term._conn.cursor()
        cur.execute("SELECT COUNT(*) FROM memory")
        total = cur.fetchone()[0]
        cur.execute("SELECT role, COUNT(*) FROM memory GROUP BY role")
        by_role = dict(cur.fetchall())
        return {
            "total_memories": total,
            "by_role": by_role,
            "code_symbols": len(self.code_rag._graph),
            "code_files": len(self.code_rag._indexed_files),
        }
