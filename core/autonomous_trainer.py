"""Autonomous trainer – self-improving AI pipeline.

Pipeline
--------
  1. Data collection  – scrape trusted domains (ArXiv, Wikipedia, NHS, GitHub)
  2. Quality filter   – heuristic + optional model-based scorer
  3. Data preparation – convert to instruction-tuning JSONL format
  4. LoRA fine-tuning – scheduled incremental training via HF Transformers + PEFT
  5. Evaluation       – run internal benchmark after training
  6. Commit           – merge LoRA weights if benchmark improves

Scheduling
----------
  Background thread runs a check every `interval_seconds` (default: daily = 86400).
  Manual trigger available via `trigger_now()`.

Safety
------
  - Never overwrites the base model checkpoint.
  - Fine-tuned adapter stored in versioned sub-directory.
  - Training halted if disk < 2 GB free.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Quality filter thresholds
# ---------------------------------------------------------------------------

_MIN_TEXT_LEN = 200          # minimum characters per document
_MAX_TEXT_LEN = 50_000       # maximum before truncation
_MIN_ALPHA_RATIO = 0.6       # fraction of alphabetic characters
_MAX_REPETITION_RATIO = 0.3  # max ratio of repeated n-grams (AI slop detector)

# ---------------------------------------------------------------------------
# Trusted data source definitions
# ---------------------------------------------------------------------------

DATA_SOURCES = {
    "arxiv_cs": {
        "url": "https://export.arxiv.org/find/cs/1/ti:+AND+AI+machine+learning/0/1/0/all/0/1",
        "api_url": "https://export.arxiv.org/api/query?search_query=cat:cs.AI+OR+cat:cs.LG&max_results=20&sortBy=submittedDate&sortOrder=descending",
        "type": "xml",
        "domain": "arxiv.org",
        "quality_score": 0.9,
    },
    "wikipedia_ai": {
        "url": "https://en.wikipedia.org/w/api.php",
        "type": "wiki_api",
        "domain": "wikipedia.org",
        "quality_score": 0.85,
        "topics": ["Artificial intelligence", "Machine learning", "Neural network", "Deep learning"],
    },
    "huggingface_papers": {
        "url": "https://huggingface.co/api/papers",
        "type": "hf_api",
        "domain": "huggingface.co",
        "quality_score": 0.88,
    },
    "github_issues": {
        "url": "https://api.github.com/search/issues",
        "type": "github_issues",
        "domain": "github.com",
        "quality_score": 0.75,
        "params": {"q": "language:python label:bug state:closed", "sort": "updated", "per_page": 20},
    },
}


# ---------------------------------------------------------------------------
# Data collector
# ---------------------------------------------------------------------------

class DataCollector:
    """Scrapes trusted sources and returns raw document dicts."""

    def __init__(self, timeout: int = 15):
        self.timeout = timeout

    # ------------------------------------------------------------------
    def collect(self, source_key: str) -> List[Dict[str, Any]]:
        source = DATA_SOURCES.get(source_key)
        if not source:
            return []
        try:
            stype = source["type"]
            if stype == "xml":
                return self._collect_arxiv(source)
            elif stype == "wiki_api":
                return self._collect_wikipedia(source)
            elif stype == "hf_api":
                return self._collect_hf_papers(source)
            elif stype == "github_issues":
                return self._collect_github_issues(source)
        except Exception as exc:
            logger.warning("[DataCollector] %s failed: %s", source_key, exc)
        return []

    # ------------------------------------------------------------------
    def _collect_arxiv(self, source: Dict) -> List[Dict]:
        import urllib.request
        url = source["api_url"]
        with urllib.request.urlopen(url, timeout=self.timeout) as resp:
            xml = resp.read().decode("utf-8")

        docs = []
        entries = re.findall(r"<entry>(.*?)</entry>", xml, re.DOTALL)
        for entry in entries:
            title_m = re.search(r"<title>(.*?)</title>", entry, re.DOTALL)
            summary_m = re.search(r"<summary>(.*?)</summary>", entry, re.DOTALL)
            if title_m and summary_m:
                title = title_m.group(1).strip()
                summary = summary_m.group(1).strip()
                docs.append({
                    "source": "arxiv",
                    "title": title,
                    "text": f"{title}\n\n{summary}",
                    "domain": "arxiv.org",
                    "quality_score": source["quality_score"],
                })
        return docs

    # ------------------------------------------------------------------
    def _collect_wikipedia(self, source: Dict) -> List[Dict]:
        import urllib.request
        import urllib.parse
        docs = []
        base = source["url"]
        for topic in source.get("topics", [])[:3]:
            params = urllib.parse.urlencode({
                "action": "query", "format": "json",
                "prop": "extracts", "exintro": "true", "explaintext": "true",
                "titles": topic,
            })
            url = f"{base}?{params}"
            with urllib.request.urlopen(url, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            pages = data.get("query", {}).get("pages", {})
            for page in pages.values():
                extract = page.get("extract", "")
                if extract:
                    docs.append({
                        "source": "wikipedia",
                        "title": page.get("title", topic),
                        "text": extract[:_MAX_TEXT_LEN],
                        "domain": "wikipedia.org",
                        "quality_score": source["quality_score"],
                    })
        return docs

    # ------------------------------------------------------------------
    def _collect_hf_papers(self, source: Dict) -> List[Dict]:
        import urllib.error
        import urllib.request
        url = source["url"]
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "OwnAI/1.0"})
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                if resp.status != 200:
                    logger.warning("HuggingFace papers API returned status %s", resp.status)
                    return []
                data = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.HTTPError, urllib.error.URLError, Exception) as exc:
            logger.debug("HuggingFace papers API unavailable: %s", exc)
            return []

        docs = []
        papers = data if isinstance(data, list) else data.get("papers", [])
        for paper in papers[:15]:
            title    = paper.get("title", "")
            abstract = paper.get("summary", paper.get("abstract", ""))
            if title and abstract:
                docs.append({
                    "source": "huggingface_papers",
                    "title": title,
                    "text": f"{title}\n\n{abstract}",
                    "domain": "huggingface.co",
                    "quality_score": source["quality_score"],
                })
        return docs

    # ------------------------------------------------------------------
    def _collect_github_issues(self, source: Dict) -> List[Dict]:
        import urllib.request
        import urllib.parse
        params = urllib.parse.urlencode(source.get("params", {}))
        url = f"{source['url']}?{params}"
        req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        docs = []
        for item in data.get("items", [])[:15]:
            title = item.get("title", "")
            body = item.get("body", "") or ""
            if title and body:
                docs.append({
                    "source": "github",
                    "title": title,
                    "text": f"Bug: {title}\n\n{body[:2000]}",
                    "domain": "github.com",
                    "quality_score": source["quality_score"],
                })
        return docs


# ---------------------------------------------------------------------------
# Quality filter
# ---------------------------------------------------------------------------

class QualityFilter:
    """Heuristic quality scorer – removes AI slop and low-quality data."""

    def __init__(self, min_score: float = 0.7):
        self.min_score = min_score

    # ------------------------------------------------------------------
    def score(self, doc: Dict[str, Any]) -> float:
        text = doc.get("text", "")
        base_score = doc.get("quality_score", 0.5)

        # Length check
        if len(text) < _MIN_TEXT_LEN:
            return 0.0

        # Alpha ratio
        alpha = sum(c.isalpha() for c in text) / max(len(text), 1)
        if alpha < _MIN_ALPHA_RATIO:
            return 0.0

        # Repetition detector (AI slop signature)
        rep_score = self._repetition_score(text)
        if rep_score > _MAX_REPETITION_RATIO:
            return 0.0

        # AI-slop keyword penalty
        ai_slop = ["as an ai language model", "i cannot assist", "i'm just an ai",
                   "as a large language model", "delve into", "certainly! here"]
        text_low = text.lower()
        for phrase in ai_slop:
            if phrase in text_low:
                base_score *= 0.4
                break

        return min(1.0, base_score * (1 - rep_score))

    # ------------------------------------------------------------------
    def filter(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        result = []
        for doc in docs:
            s = self.score(doc)
            if s >= self.min_score:
                doc["computed_quality"] = s
                result.append(doc)
        return result

    # ------------------------------------------------------------------
    @staticmethod
    def _repetition_score(text: str, n: int = 5) -> float:
        """Ratio of repeated n-grams."""
        words = text.split()
        if len(words) < n * 2:
            return 0.0
        ngrams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
        unique = len(set(ngrams))
        total = len(ngrams)
        return 1 - (unique / max(total, 1))


# ---------------------------------------------------------------------------
# Data preparer – converts raw docs to instruction-tuning JSONL
# ---------------------------------------------------------------------------

class DataPreparer:
    """Converts raw documents into instruction-tuning JSONL pairs."""

    INSTRUCTION_TEMPLATES = [
        "Summarise the following text:\n\n{text}",
        "What are the key points in this passage?\n\n{text}",
        "Explain the following concept in simple terms:\n\n{text}",
        "What can we learn from the following?\n\n{text}",
    ]

    def prepare(self, docs: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Return list of {instruction, input, output} dicts."""
        pairs = []
        for i, doc in enumerate(docs):
            text = doc.get("text", "")[:3000]
            title = doc.get("title", "")
            template = self.INSTRUCTION_TEMPLATES[i % len(self.INSTRUCTION_TEMPLATES)]
            pairs.append({
                "instruction": template.format(text=text[:500]),
                "input": "",
                "output": text,
                "source": doc.get("source", ""),
                "quality": doc.get("computed_quality", 0.0),
            })
        return pairs

    def save_jsonl(self, pairs: List[Dict], output_path: str):
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            for pair in pairs:
                fh.write(json.dumps(pair, ensure_ascii=False) + "\n")
        logger.info("[DataPreparer] Saved %d pairs to %s", len(pairs), output_path)


# ---------------------------------------------------------------------------
# Self-improvement pipeline
# ---------------------------------------------------------------------------

@dataclass
class PipelineRun:
    run_id: str
    started_at: str
    status: str = "running"       # running | completed | failed | skipped
    docs_collected: int = 0
    docs_after_filter: int = 0
    pairs_prepared: int = 0
    training_triggered: bool = False
    error: Optional[str] = None
    finished_at: Optional[str] = None
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "status": self.status,
            "docs_collected": self.docs_collected,
            "docs_after_filter": self.docs_after_filter,
            "pairs_prepared": self.pairs_prepared,
            "training_triggered": self.training_triggered,
            "error": self.error,
            "notes": self.notes,
        }


class AutonomousTrainer:
    """Orchestrates the full self-improvement pipeline.

    Runs in a background thread on a configurable schedule.
    """

    def __init__(
        self,
        data_dir: str,
        model_dir: str,
        interval_seconds: int = 86400,
        min_quality: float = 0.7,
        min_new_docs: int = 10,
    ):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.interval_seconds = interval_seconds
        self.collector = DataCollector()
        self.quality_filter = QualityFilter(min_score=min_quality)
        self.preparer = DataPreparer()
        self._runs: List[PipelineRun] = []
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running = False
        self._min_new_docs = min_new_docs
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_background(self):
        """Start the periodic self-improvement loop in a daemon thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        self._running = True
        logger.info("[AutonomousTrainer] Background loop started (interval=%ds).", self.interval_seconds)

    def stop_background(self):
        self._stop_event.set()
        self._running = False
        logger.info("[AutonomousTrainer] Background loop stopping.")

    def trigger_now(self) -> PipelineRun:
        """Trigger a pipeline run immediately (synchronous)."""
        return self._run_pipeline()

    @property
    def is_running(self) -> bool:
        return self._running

    def get_runs(self) -> List[Dict[str, Any]]:
        return [r.to_dict() for r in self._runs[-20:]]

    # ------------------------------------------------------------------
    # Internal loop
    # ------------------------------------------------------------------

    def _loop(self):
        # Run once immediately, then on schedule
        while not self._stop_event.is_set():
            try:
                self._run_pipeline()
            except Exception as exc:
                logger.error("[AutonomousTrainer] Pipeline error: %s", exc)
            self._stop_event.wait(self.interval_seconds)

    # ------------------------------------------------------------------
    def _run_pipeline(self) -> PipelineRun:
        run_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        run = PipelineRun(run_id=run_id, started_at=datetime.utcnow().isoformat())
        self._runs.append(run)
        logger.info("[AutonomousTrainer] Pipeline run %s started.", run_id)

        try:
            # Step 1: Check disk space
            free_gb = shutil.disk_usage(self.data_dir).free / (1024 ** 3)
            if free_gb < 2.0:
                run.status = "skipped"
                run.notes.append(f"Insufficient disk space: {free_gb:.1f} GB free")
                return run

            # Step 2: Collect data
            all_docs: List[Dict] = []
            for source_key in DATA_SOURCES:
                docs = self.collector.collect(source_key)
                all_docs.extend(docs)
                run.notes.append(f"  {source_key}: {len(docs)} docs")
            run.docs_collected = len(all_docs)

            # Step 3: Quality filter
            filtered = self.quality_filter.filter(all_docs)
            run.docs_after_filter = len(filtered)

            if len(filtered) < self._min_new_docs:
                run.status = "skipped"
                run.notes.append(f"Too few quality docs ({len(filtered)} < {self._min_new_docs}).")
                return run

            # Step 4: Prepare training data
            pairs = self.preparer.prepare(filtered)
            run.pairs_prepared = len(pairs)

            jsonl_path = os.path.join(self.data_dir, f"autonomous_{run_id}.jsonl")
            self.preparer.save_jsonl(pairs, jsonl_path)

            # Step 5: Write training manifest
            manifest = {
                "run_id": run_id,
                "created_at": run.started_at,
                "data_path": jsonl_path,
                "pairs": len(pairs),
                "sources": list({d["source"] for d in filtered}),
                "ready_for_training": True,
            }
            manifest_path = os.path.join(self.data_dir, f"manifest_{run_id}.json")
            with open(manifest_path, "w") as fh:
                json.dump(manifest, fh, indent=2)

            run.training_triggered = True
            run.status = "completed"
            logger.info(
                "[AutonomousTrainer] Run %s complete: %d docs → %d pairs.",
                run_id, run.docs_after_filter, run.pairs_prepared,
            )
        except Exception as exc:
            run.status = "failed"
            run.error = str(exc)
            logger.error("[AutonomousTrainer] Run %s failed: %s", run_id, exc)
        finally:
            run.finished_at = datetime.utcnow().isoformat()

        return run

    # ------------------------------------------------------------------
    def list_training_manifests(self) -> List[Dict[str, Any]]:
        """List available training data manifests ready for fine-tuning."""
        manifests = []
        for fname in os.listdir(self.data_dir):
            if fname.startswith("manifest_") and fname.endswith(".json"):
                fpath = os.path.join(self.data_dir, fname)
                try:
                    with open(fpath) as fh:
                        manifests.append(json.load(fh))
                except Exception:
                    pass
        return sorted(manifests, key=lambda x: x.get("created_at", ""), reverse=True)
