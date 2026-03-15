"""Ollama bridge – local LLM inference without external APIs.

Provides a unified interface for calling locally-running Ollama models.
Falls back gracefully when Ollama is not installed / not running.

Usage::

    from core.ollama_bridge import OllamaBridge, LocalInferenceRouter

    bridge = OllamaBridge(model="qwen2.5:7b")
    response = bridge.chat("What is 2+2?")

    # Multi-turn conversation
    router = LocalInferenceRouter(model="qwen2.5:7b")
    reply = router.ask(messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": "Explain recursion briefly."},
    ])
"""
from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL  = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL    = os.environ.get("OWNAI_DEFAULT_MODEL", "qwen2.5:7b")
DEFAULT_TIMEOUT  = 120        # seconds per request
MAX_RETRIES      = 3
RETRY_DELAY      = 2.0        # seconds between retries


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------

def is_ollama_running() -> bool:
    """Return True if the local Ollama server is reachable."""
    try:
        req = urllib.request.Request(f"{OLLAMA_BASE_URL}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=3):
            return True
    except Exception:
        return False


def list_local_models() -> List[str]:
    """Return list of model names currently installed in Ollama."""
    try:
        req = urllib.request.Request(f"{OLLAMA_BASE_URL}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
        return [m["name"] for m in data.get("models", [])]
    except Exception as exc:
        logger.debug("Could not list Ollama models: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ChatMessage:
    role: str     # "system" | "user" | "assistant"
    content: str

    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class OllamaResponse:
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    elapsed_ms: int = 0
    success: bool = True
    error: Optional[str] = None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


# ---------------------------------------------------------------------------
# Core bridge
# ---------------------------------------------------------------------------

class OllamaBridge:
    """Direct HTTP client for the Ollama REST API.

    Supports:
      - Single-turn completion (``generate``)
      - Multi-turn chat (``chat``)
      - Streaming responses
      - Model pull / management
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = OLLAMA_BASE_URL,
        timeout: int = DEFAULT_TIMEOUT,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        system_prompt: str = "",
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(
        self,
        user_message: str,
        history: Optional[List[Dict[str, str]]] = None,
        system: Optional[str] = None,
    ) -> OllamaResponse:
        """Send a chat message and return the response.

        Args:
            user_message: The user's message.
            history: Optional list of prior messages [{"role":..,"content":..}].
            system: Override the instance system prompt for this call.

        Returns:
            OllamaResponse with .text, .success, and token counts.
        """
        messages = []
        sys_msg = system or self.system_prompt
        if sys_msg:
            messages.append({"role": "system", "content": sys_msg})
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_message})
        return self._chat_request(messages)

    def complete(self, prompt: str) -> OllamaResponse:
        """Single-shot text completion (no conversation history)."""
        return self._generate_request(prompt)

    def stream_chat(
        self,
        user_message: str,
        history: Optional[List[Dict[str, str]]] = None,
        system: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """Stream response tokens one chunk at a time.

        Yields text fragments as they arrive from the model.
        """
        messages = []
        if system or self.system_prompt:
            messages.append({"role": "system", "content": system or self.system_prompt})
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_message})

        payload = json.dumps({
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }).encode()

        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8").strip()
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                        token = chunk.get("message", {}).get("content", "")
                        if token:
                            yield token
                    except json.JSONDecodeError:
                        continue
        except Exception as exc:
            yield f"\n[Stream error: {exc}]"

    def pull_model(self, model: Optional[str] = None) -> bool:
        """Pull (download) a model from Ollama library. Returns True on success."""
        target = model or self.model
        payload = json.dumps({"name": target, "stream": False}).encode()
        req = urllib.request.Request(
            f"{self.base_url}/api/pull",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                data = json.loads(resp.read().decode())
            return data.get("status") == "success"
        except Exception as exc:
            logger.error("Failed to pull model %s: %s", target, exc)
            return False

    # ------------------------------------------------------------------
    # Private request helpers
    # ------------------------------------------------------------------

    def _chat_request(self, messages: List[Dict[str, str]]) -> OllamaResponse:
        payload = json.dumps({
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }).encode()
        return self._post("/api/chat", payload, key="message.content")

    def _generate_request(self, prompt: str) -> OllamaResponse:
        payload = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }).encode()
        return self._post("/api/generate", payload, key="response")

    def _post(self, path: str, payload: bytes, key: str = "response") -> OllamaResponse:
        url = self.base_url + path
        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        t0 = time.time()
        last_exc: Optional[Exception] = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read().decode())
                elapsed = int((time.time() - t0) * 1000)

                # Navigate dotted key (e.g. "message.content")
                text = data
                for part in key.split("."):
                    if isinstance(text, dict):
                        text = text.get(part, "")
                    else:
                        text = ""
                        break

                return OllamaResponse(
                    text=str(text),
                    model=self.model,
                    prompt_tokens=data.get("prompt_eval_count", 0),
                    completion_tokens=data.get("eval_count", 0),
                    elapsed_ms=elapsed,
                    success=True,
                )
            except urllib.error.URLError as exc:
                last_exc = exc
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY * attempt)
            except Exception as exc:
                last_exc = exc
                break

        elapsed = int((time.time() - t0) * 1000)
        return OllamaResponse(
            text="",
            model=self.model,
            elapsed_ms=elapsed,
            success=False,
            error=str(last_exc),
        )


# ---------------------------------------------------------------------------
# High-level router – picks the best available local inference method
# ---------------------------------------------------------------------------

class LocalInferenceRouter:
    """Smart router that tries Ollama first, then HuggingFace transformers pipeline.

    This means the system works even without Ollama, as long as
    `transformers` and a small model are available.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.7,
        max_tokens: int = 512,
        system_prompt: str = "You are a helpful AI assistant.",
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self._ollama: Optional[OllamaBridge] = None
        self._hf_pipe = None
        self._backend: Optional[str] = None
        self._init()

    # ------------------------------------------------------------------

    def _init(self):
        if is_ollama_running():
            self._ollama = OllamaBridge(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                system_prompt=self.system_prompt,
            )
            self._backend = "ollama"
            logger.info("LocalInferenceRouter: using Ollama (model=%s)", self.model)
            return
        # Try HuggingFace transformers for a lightweight fallback
        try:
            from transformers import pipeline as hf_pipeline
            import torch
            small_model = "microsoft/phi-2"
            self._hf_pipe = hf_pipeline(
                "text-generation",
                model=small_model,
                device_map="auto",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                max_new_tokens=self.max_tokens,
            )
            self._backend = "transformers"
            logger.info("LocalInferenceRouter: using HuggingFace transformers (%s)", small_model)
        except Exception as exc:
            logger.warning("No local inference backend available: %s", exc)
            self._backend = "none"

    # ------------------------------------------------------------------

    def ask(
        self,
        messages: List[Dict[str, str]],
        stream: bool = False,
    ) -> str:
        """Send a message list and return the reply text.

        Args:
            messages: List of {"role": .., "content": ..} dicts.
            stream: If True and backend is Ollama, stream tokens to stdout.

        Returns:
            The model's reply as a string.
        """
        if self._backend == "ollama" and self._ollama:
            user_msg = next(
                (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
            )
            history = [m for m in messages if m["role"] != "user" or m is not messages[-1]]
            if stream:
                result = ""
                for token in self._ollama.stream_chat(user_msg, history):
                    print(token, end="", flush=True)
                    result += token
                print()
                return result
            resp = self._ollama.chat(user_msg, history)
            if resp.success:
                return resp.text
            logger.warning("Ollama error: %s", resp.error)
            return f"[Ollama error: {resp.error}]"

        if self._backend == "transformers" and self._hf_pipe:
            prompt = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
            out = self._hf_pipe(prompt)
            generated = out[0]["generated_text"]
            if generated.startswith(prompt):
                generated = generated[len(prompt):].strip()
            return generated or "(no response)"

        # Absolute fallback – return a placeholder
        return (
            "⚠️ No local inference backend found. "
            "Install Ollama (https://ollama.com) and run `ollama pull qwen2.5:7b`."
        )

    def is_available(self) -> bool:
        return self._backend not in (None, "none")

    def backend_name(self) -> str:
        return self._backend or "none"

    def switch_model(self, model: str):
        """Hot-swap the model (Ollama only)."""
        self.model = model
        if self._ollama:
            self._ollama.model = model


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def quick_chat(message: str, model: str = DEFAULT_MODEL, system: str = "") -> str:
    """One-shot helper for simple questions."""
    bridge = OllamaBridge(model=model, system_prompt=system)
    resp = bridge.chat(message)
    return resp.text if resp.success else f"[Error: {resp.error}]"


def get_status() -> Dict[str, Any]:
    """Return current Ollama status and available models."""
    running = is_ollama_running()
    return {
        "ollama_running": running,
        "models": list_local_models() if running else [],
        "base_url": OLLAMA_BASE_URL,
        "default_model": DEFAULT_MODEL,
    }
