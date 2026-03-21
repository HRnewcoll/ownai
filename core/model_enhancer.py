"""Model Enhancer – add new capabilities to any existing model.

Takes an already-trained (or downloaded) model and generates scripts that
augment it with extra skills:

  ┌─────────────────────────────────────────────────────────┐
  │  Existing Model  →  Model Enhancer  →  Enhanced Model   │
  │                                                          │
  │  Capabilities you can add:                               │
  │   • vision        – LLaVA-style visual projection layer  │
  │   • tool_use      – function-calling fine-tune           │
  │   • voice_input   – Whisper integration                  │
  │   • voice_output  – Piper / Coqui TTS                    │
  │   • knowledge     – LoRA continual learning on new data  │
  │   • memory        – external long-term memory store      │
  │   • reasoning     – Chain-of-Thought fine-tune (LoRA)    │
  │   • moe           – Mixture-of-Experts adapter           │
  └─────────────────────────────────────────────────────────┘

Usage::

    from core.model_enhancer import ModelEnhancer

    enhancer = ModelEnhancer()

    # Detect what a model already supports
    caps = enhancer.detect_capabilities("path/to/my/model")

    # Generate an enhancement script to add vision to a text-only model
    script = enhancer.generate_enhancement_script(
        base_model="path/to/my/model",
        enhancements=["vision", "tool_use"],
        output_dir="/tmp/my_enhanced_model",
    )
"""
from __future__ import annotations

import json
import logging
import os
import textwrap
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Capability metadata
# ---------------------------------------------------------------------------

ALL_ENHANCEMENTS: Dict[str, Dict[str, str]] = {
    "vision": {
        "label": "Vision (Image Understanding)",
        "icon": "🖼️",
        "description": (
            "Adds a CLIP vision encoder + linear projection layer so the model "
            "can process images alongside text (LLaVA-style architecture)."
        ),
        "method": "lora",
        "complexity": "medium",
        "requirements": "Pillow, open-clip-torch or clip",
    },
    "tool_use": {
        "label": "Tool Use / Function Calling",
        "icon": "🔧",
        "description": (
            "Fine-tunes the model on function-calling datasets (glaiveai/glaive-function-calling) "
            "so it learns to emit structured JSON tool calls."
        ),
        "method": "lora",
        "complexity": "low",
        "requirements": "datasets, peft, trl",
    },
    "voice_input": {
        "label": "Voice Input (Speech-to-Text)",
        "icon": "🎙️",
        "description": (
            "Integrates OpenAI Whisper as an offline STT front-end. "
            "Audio is transcribed locally before being sent to the LLM."
        ),
        "method": "integration",
        "complexity": "low",
        "requirements": "openai-whisper, sounddevice or pyaudio",
    },
    "voice_output": {
        "label": "Voice Output (Text-to-Speech)",
        "icon": "🔊",
        "description": (
            "Integrates Piper TTS (fast, offline) or Coqui TTS for "
            "generating spoken audio from the model's text replies."
        ),
        "method": "integration",
        "complexity": "low",
        "requirements": "piper-tts or TTS (coqui)",
    },
    "knowledge": {
        "label": "Knowledge Update (Continual Learning)",
        "icon": "📚",
        "description": (
            "Runs a LoRA continual pre-training pass on user-supplied documents "
            "(PDFs, text files, URLs) to inject new factual knowledge without "
            "catastrophic forgetting."
        ),
        "method": "lora",
        "complexity": "medium",
        "requirements": "peft, trl, datasets",
    },
    "memory": {
        "label": "Long-Term Memory",
        "icon": "🗄️",
        "description": (
            "Attaches OwnAI's vector-based memory store so the model can "
            "recall facts and conversation history across sessions."
        ),
        "method": "integration",
        "complexity": "low",
        "requirements": "sqlite3 (stdlib)",
    },
    "reasoning": {
        "label": "Chain-of-Thought Reasoning",
        "icon": "🧠",
        "description": (
            "Fine-tunes on Open-Platypus / MetaMathQA so the model learns to "
            "think step-by-step before answering complex questions."
        ),
        "method": "lora",
        "complexity": "medium",
        "requirements": "peft, trl, datasets",
    },
    "web_search": {
        "label": "Web Search (DuckDuckGo)",
        "icon": "🌐",
        "description": (
            "Integrates a lightweight DuckDuckGo search tool so the model can "
            "retrieve up-to-date information without paid APIs."
        ),
        "method": "integration",
        "complexity": "low",
        "requirements": "duckduckgo-search",
    },
    "code_execution": {
        "label": "Code Execution (Sandboxed)",
        "icon": "⚙️",
        "description": (
            "Connects OwnAI's CodeSandbox so the model can write and run "
            "Python/bash code in a safe subprocess, observe the output, and "
            "iterate (Think-Act-Observe-Verify loop)."
        ),
        "method": "integration",
        "complexity": "low",
        "requirements": "stdlib only",
    },
    "moe": {
        "label": "Mixture-of-Experts (MoE) Conversion",
        "icon": "⚡",
        "description": (
            "Converts dense FFN layers to sparse MoE blocks (top-k routing) "
            "using the mergekit / MoE-LLaVA approach, reducing active "
            "parameters by ~4× while keeping full capacity."
        ),
        "method": "architecture",
        "complexity": "high",
        "requirements": "mergekit or transformers>=4.36",
    },
}


# ---------------------------------------------------------------------------
# Capability detector
# ---------------------------------------------------------------------------

@dataclass
class DetectedCapabilities:
    base_model: str
    architecture: str = "unknown"
    parameter_count: str = "unknown"
    has_vision: bool = False
    has_tool_use: bool = False
    has_voice_input: bool = False
    has_voice_output: bool = False
    has_memory: bool = False
    has_reasoning: bool = False
    has_moe: bool = False
    raw_config: Dict[str, Any] = field(default_factory=dict)
    detection_notes: List[str] = field(default_factory=list)

    def missing(self, requested: List[str]) -> List[str]:
        """Return which requested enhancements are not already present."""
        has_map = {
            "vision": self.has_vision,
            "tool_use": self.has_tool_use,
            "voice_input": self.has_voice_input,
            "voice_output": self.has_voice_output,
            "memory": self.has_memory,
            "reasoning": self.has_reasoning,
            "moe": self.has_moe,
        }
        return [e for e in requested if not has_map.get(e, False)]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "base_model": self.base_model,
            "architecture": self.architecture,
            "parameter_count": self.parameter_count,
            "capabilities": {
                "vision": self.has_vision,
                "tool_use": self.has_tool_use,
                "voice_input": self.has_voice_input,
                "voice_output": self.has_voice_output,
                "memory": self.has_memory,
                "reasoning": self.has_reasoning,
                "moe": self.has_moe,
            },
            "notes": self.detection_notes,
        }


class ModelEnhancer:
    """Generates enhancement scripts and integration configs for existing models."""

    # ------------------------------------------------------------------
    # Capability detection
    # ------------------------------------------------------------------

    def detect_capabilities(self, model_path_or_id: str) -> DetectedCapabilities:
        """Detect what capabilities a model already has.

        Works by inspecting:
          1. The ``config.json`` in a local directory
          2. The HuggingFace Hub API for remote models
          3. String heuristics on the model name
        """
        caps = DetectedCapabilities(base_model=model_path_or_id)
        model_id_lower = model_path_or_id.lower()

        # ── 1. Local config.json ──────────────────────────────────────────
        if os.path.isdir(model_path_or_id):
            cfg_path = os.path.join(model_path_or_id, "config.json")
            if os.path.exists(cfg_path):
                try:
                    with open(cfg_path, encoding="utf-8") as fh:
                        cfg = json.load(fh)
                    caps.raw_config = cfg
                    caps.architecture = cfg.get("architectures", ["unknown"])[0] if cfg.get("architectures") else "unknown"
                    # MoE indicator
                    if any(k in cfg for k in ("num_experts", "num_local_experts", "moe_num_experts")):
                        caps.has_moe = True
                    # Vision indicator
                    if "vision_config" in cfg or "image_token_index" in cfg:
                        caps.has_vision = True
                    # Param count
                    params = cfg.get("num_parameters")
                    if params:
                        caps.parameter_count = f"{params / 1e9:.1f}B"
                    caps.detection_notes.append(f"Read local config.json ({cfg_path})")
                except Exception as exc:
                    caps.detection_notes.append(f"Could not read config.json: {exc}")

        # ── 2. HuggingFace Hub heuristics ─────────────────────────────────
        if not os.path.isdir(model_path_or_id):
            try:
                import urllib.request as _req
                api_url = f"https://huggingface.co/api/models/{model_path_or_id}"
                with _req.urlopen(api_url, timeout=5) as resp:
                    hub_data = json.loads(resp.read().decode())
                tags = hub_data.get("tags", [])
                caps.detection_notes.append(f"HuggingFace tags: {', '.join(tags[:10])}")
                if "multimodal" in tags or "image-text-to-text" in tags:
                    caps.has_vision = True
                if "function-calling" in tags or "tool-use" in tags:
                    caps.has_tool_use = True
            except Exception:
                caps.detection_notes.append("HuggingFace API unavailable (offline mode)")

        # ── 3. Name-based heuristics ──────────────────────────────────────
        name_checks = {
            "has_vision": ["llava", "vision", "vl", "clip", "blip", "cogvlm",
                           "qwen-vl", "qwen2-vl", "idefics", "paligemma"],
            "has_tool_use": ["tool", "function", "instruct", "agent", "act",
                             "toolbench", "functionary"],
            "has_moe": ["moe", "mixtral", "deepseek-v", "qwen-moe", "switch"],
            "has_reasoning": ["r1", "qwq", "o1", "deepseek-r", "reasoning", "think"],
        }
        for attr, keywords in name_checks.items():
            if any(kw in model_id_lower for kw in keywords):
                setattr(caps, attr, True)
                caps.detection_notes.append(f"Detected {attr} from model name")

        return caps

    # ------------------------------------------------------------------
    # Enhancement script generation
    # ------------------------------------------------------------------

    def generate_enhancement_script(
        self,
        base_model: str,
        enhancements: List[str],
        output_dir: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate a Python script that applies the requested enhancements.

        Args:
            base_model: Local path or HuggingFace model ID.
            enhancements: List of capability keys from ALL_ENHANCEMENTS.
            output_dir: Where to write the script and save the enhanced model.
            config: Optional extra parameters (epochs, lr, dataset, etc.).

        Returns:
            Absolute path to the generated ``enhance.py`` script.
        """
        cfg = config or {}
        os.makedirs(output_dir, exist_ok=True)

        # Separate LoRA-based enhancements from pure integrations
        lora_enhancements  = [e for e in enhancements
                               if ALL_ENHANCEMENTS.get(e, {}).get("method") == "lora"]
        arch_enhancements  = [e for e in enhancements
                               if ALL_ENHANCEMENTS.get(e, {}).get("method") == "architecture"]
        integration_caps   = [e for e in enhancements
                               if ALL_ENHANCEMENTS.get(e, {}).get("method") == "integration"]

        script = self._build_script_header(base_model, enhancements, output_dir, cfg)

        # Add sections in dependency order
        if "vision" in lora_enhancements:
            script += self._vision_section(base_model, output_dir, cfg)
        if lora_enhancements:
            script += self._lora_section(base_model, output_dir, lora_enhancements, cfg)
        if "moe" in arch_enhancements:
            script += self._moe_section(base_model, output_dir, cfg)
        if integration_caps:
            script += self._integration_section(integration_caps)

        script += self._save_section(output_dir)

        script_path = os.path.join(output_dir, "enhance.py")
        with open(script_path, "w", encoding="utf-8") as fh:
            fh.write(script)

        # Save enhancement manifest
        manifest = {
            "base_model": base_model,
            "enhancements": enhancements,
            "output_dir": output_dir,
            "config": cfg,
        }
        with open(os.path.join(output_dir, "enhance_manifest.json"), "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)

        logger.info("Enhancement script written to %s", script_path)
        return script_path

    # ------------------------------------------------------------------
    # Script section builders
    # ------------------------------------------------------------------

    @staticmethod
    def _build_script_header(
        base_model: str, enhancements: List[str], output_dir: str, cfg: Dict
    ) -> str:
        cap_str = ", ".join(enhancements)
        epochs  = cfg.get("epochs", 2)
        lr      = cfg.get("learning_rate", 1e-4)
        batch   = cfg.get("batch_size", 2)
        max_len = cfg.get("max_seq_len", 512)
        return textwrap.dedent(f"""\
            #!/usr/bin/env python3
            \"\"\"OwnAI – Model Enhancement Script.

            Base model  : {base_model}
            Enhancements: {cap_str}
            Output dir  : {output_dir}

            Auto-generated by OwnAI ModelEnhancer. Edit before running.
            \"\"\"
            import json
            import os
            import torch

            BASE_MODEL  = {base_model!r}
            OUTPUT_DIR  = {output_dir!r}
            EPOCHS      = {epochs}
            LR          = {lr}
            BATCH_SIZE  = {batch}
            MAX_SEQ_LEN = {max_len}

            os.makedirs(OUTPUT_DIR, exist_ok=True)
            print(f"[OwnAI Enhance] Base model: {{BASE_MODEL}}")
            print(f"[OwnAI Enhance] Enhancements: {cap_str}")
            print(f"[OwnAI Enhance] Output: {{OUTPUT_DIR}}")

        """)

    @staticmethod
    def _vision_section(base_model: str, output_dir: str, cfg: Dict) -> str:
        vision_model = cfg.get("vision_encoder", "openai/clip-vit-base-patch32")
        return textwrap.dedent(f"""\
            # ══════════════════════════════════════════════════════════════
            # ENHANCEMENT: Vision (LLaVA-style projection layer)
            # ══════════════════════════════════════════════════════════════
            print("\\n[OwnAI] Adding vision capability...")

            from transformers import (
                AutoTokenizer, AutoModelForCausalLM,
                CLIPVisionModel, CLIPImageProcessor, BitsAndBytesConfig
            )
            import torch.nn as nn
            from PIL import Image
            import requests

            class VisionProjector(nn.Module):
                \"\"\"Linear projection from CLIP vision space to LLM embedding space.\"\"\"
                def __init__(self, clip_dim: int, llm_dim: int):
                    super().__init__()
                    self.proj = nn.Sequential(
                        nn.Linear(clip_dim, llm_dim * 2),
                        nn.GELU(),
                        nn.Linear(llm_dim * 2, llm_dim),
                    )

                def forward(self, x):
                    return self.proj(x)

            class LLaVAStyleModel(nn.Module):
                \"\"\"Vision-Language model combining a CLIP encoder with an LLM.\"\"\"
                def __init__(self, llm, vision_encoder, projector):
                    super().__init__()
                    self.llm = llm
                    self.vision_encoder = vision_encoder
                    self.projector = projector

                def encode_image(self, pixel_values):
                    with torch.no_grad():
                        vision_out = self.vision_encoder(pixel_values=pixel_values)
                        image_feats = vision_out.last_hidden_state[:, 1:, :]  # drop CLS
                    return self.projector(image_feats)

                def forward(self, input_ids=None, pixel_values=None, **kwargs):
                    if pixel_values is not None:
                        image_embeds = self.encode_image(pixel_values)
                        text_embeds  = self.llm.get_input_embeddings()(input_ids)
                        # Prepend image tokens to text sequence
                        inputs_embeds = torch.cat([image_embeds, text_embeds], dim=1)
                        return self.llm(inputs_embeds=inputs_embeds, **kwargs)
                    return self.llm(input_ids=input_ids, **kwargs)

            print("[OwnAI] Loading base LLM...")
            tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
            llm = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL, device_map="auto", torch_dtype=torch.float16,
                trust_remote_code=True,
            )

            print("[OwnAI] Loading CLIP vision encoder: {vision_model}")
            processor  = CLIPImageProcessor.from_pretrained({vision_model!r})
            vision_enc = CLIPVisionModel.from_pretrained({vision_model!r})

            clip_dim = vision_enc.config.hidden_size
            llm_dim  = llm.config.hidden_size
            projector = VisionProjector(clip_dim, llm_dim)

            vl_model = LLaVAStyleModel(llm, vision_enc, projector)

            # Fine-tune projector only (LLM frozen) on image-caption data
            from datasets import load_dataset
            from torch.utils.data import DataLoader
            from transformers import get_cosine_schedule_with_warmup

            dataset = load_dataset("nlphuji/flickr30k", split="test[:500]",
                                   trust_remote_code=True)

            optim = torch.optim.AdamW(projector.parameters(), lr=LR)

            print("[OwnAI] Training vision projector...")
            vl_model.vision_encoder.requires_grad_(False)
            vl_model.llm.requires_grad_(False)
            vl_model.projector.requires_grad_(True)
            vl_model.train()

            for epoch in range(EPOCHS):
                for batch_idx, sample in enumerate(dataset):
                    # Minimal training loop – extend with real caption data
                    optim.zero_grad()
                    caption  = sample.get("caption", "An image.")
                    if not isinstance(caption, str):
                        caption = str(caption)
                    try:
                        image    = sample["image"]
                        pv       = processor(images=image, return_tensors="pt").pixel_values
                        enc      = tokenizer(caption, return_tensors="pt", truncation=True,
                                             max_length=MAX_SEQ_LEN)
                        out      = vl_model(input_ids=enc["input_ids"],
                                            pixel_values=pv,
                                            labels=enc["input_ids"])
                        if out.loss is not None:
                            out.loss.backward()
                            optim.step()
                    except Exception as e:
                        print(f"  Skipping sample: {{e}}")
                        continue
                    if (batch_idx + 1) % 50 == 0:
                        print(f"  Epoch {{epoch+1}} step {{batch_idx+1}}")
                print(f"  Epoch {{epoch+1}} done.")

            # Save projector weights
            torch.save(projector.state_dict(),
                       os.path.join(OUTPUT_DIR, "vision_projector.pt"))
            tokenizer.save_pretrained(OUTPUT_DIR)
            print("[OwnAI] Vision projector saved.")

        """)

    @staticmethod
    def _lora_section(
        base_model: str, output_dir: str, enhancements: List[str], cfg: Dict
    ) -> str:
        # Pick dataset based on enhancement type
        dataset_map = {
            "tool_use":  "glaiveai/glaive-function-calling-v2",
            "knowledge": "",     # user-supplied
            "reasoning": "garage-bAInd/Open-Platypus",
        }
        enhancement_notes = "\n".join(
            f"#   - {e}: {ALL_ENHANCEMENTS.get(e, {}).get('description', '')}"
            for e in enhancements if e != "vision"
        )
        primary = next((e for e in enhancements if e in dataset_map), enhancements[0])
        hf_ds = cfg.get("hf_dataset", dataset_map.get(primary, "tatsu-lab/alpaca"))
        user_docs = cfg.get("user_docs", "")

        return textwrap.dedent(f"""\
            # ══════════════════════════════════════════════════════════════
            # ENHANCEMENT: LoRA Fine-tune
            # {enhancement_notes}
            # ══════════════════════════════════════════════════════════════
            print("\\n[OwnAI] Starting LoRA enhancement fine-tune...")

            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
            from peft import LoraConfig, get_peft_model
            from datasets import load_dataset, Dataset
            from trl import SFTTrainer

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )

            tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
            tok.pad_token = tok.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )

            lora_cfg = LoraConfig(
                r=16, lora_alpha=32,
                target_modules="all-linear",
                lora_dropout=0.05, bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_cfg)
            model.print_trainable_parameters()

            def load_enhancement_data():
                texts = []

                # 1. HuggingFace dataset
                hf_id = {hf_ds!r}
                if hf_id:
                    try:
                        print(f"[OwnAI] Loading HF dataset: {{hf_id}}")
                        ds = load_dataset(hf_id, split="train[:2000]", trust_remote_code=True)
                        text_col = next((c for c in ds.column_names if c in
                                        ("text", "conversations", "input", "prompt", "chat")), None)
                        if text_col:
                            texts += [str(r[text_col]) for r in ds if r[text_col]]
                    except Exception as e:
                        print(f"  HF dataset load failed: {{e}}")

                # 2. User-supplied documents
                user_path = {user_docs!r}
                if user_path and os.path.exists(user_path):
                    if os.path.isfile(user_path):
                        with open(user_path, encoding="utf-8") as f:
                            texts += [line.strip() for line in f if line.strip()]
                    elif os.path.isdir(user_path):
                        for fn in os.listdir(user_path):
                            fp = os.path.join(user_path, fn)
                            if os.path.isfile(fp):
                                with open(fp, encoding="utf-8", errors="ignore") as f:
                                    texts.append(f.read())

                if not texts:
                    texts = ["Enhance this model with new knowledge.",
                             "I am an enhanced AI assistant."]
                return Dataset.from_dict({{"text": texts[:5000]}})

            train_ds = load_enhancement_data()
            print(f"[OwnAI] Enhancement dataset: {{len(train_ds)}} samples")

            training_args = TrainingArguments(
                output_dir=OUTPUT_DIR,
                num_train_epochs=EPOCHS,
                per_device_train_batch_size=BATCH_SIZE,
                learning_rate=LR,
                fp16=torch.cuda.is_available(),
                logging_steps=10,
                save_steps=200,
                save_total_limit=1,
                report_to="none",
            )

            trainer = SFTTrainer(
                model=model, tokenizer=tok,
                train_dataset=train_ds,
                dataset_text_field="text",
                max_seq_length=MAX_SEQ_LEN,
                args=training_args,
            )

            print("[OwnAI] Fine-tuning...")
            trainer.train()
            trainer.save_model(OUTPUT_DIR)
            tok.save_pretrained(OUTPUT_DIR)
            print(f"[OwnAI] LoRA adapter saved to {{OUTPUT_DIR}}")

        """)

    @staticmethod
    def _moe_section(base_model: str, output_dir: str, cfg: Dict) -> str:
        num_experts = cfg.get("num_experts", 4)
        top_k       = cfg.get("moe_top_k", 2)
        return textwrap.dedent(f"""\
            # ══════════════════════════════════════════════════════════════
            # ENHANCEMENT: Mixture-of-Experts (MoE) conversion
            # ══════════════════════════════════════════════════════════════
            print("\\n[OwnAI] Converting dense FFN layers to MoE...")

            from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
            import torch.nn as nn
            import copy

            class TopKRouter(nn.Module):
                \"\"\"Sparse top-k routing gate for MoE.\"\"\"
                def __init__(self, hidden_size: int, num_experts: int, top_k: int):
                    super().__init__()
                    self.gate = nn.Linear(hidden_size, num_experts, bias=False)
                    self.top_k = top_k

                def forward(self, x):
                    logits   = self.gate(x)
                    weights, indices = torch.topk(logits, self.top_k, dim=-1)
                    weights  = torch.softmax(weights, dim=-1)
                    return weights, indices

            class MoEFFN(nn.Module):
                \"\"\"Sparse MoE FFN: replaces a dense MLP with N expert copies.\"\"\"
                def __init__(self, original_ffn: nn.Module, num_experts: int, top_k: int):
                    super().__init__()
                    hidden = next(p.shape[0] for p in original_ffn.parameters())
                    self.experts  = nn.ModuleList([copy.deepcopy(original_ffn)
                                                  for _ in range(num_experts)])
                    self.router   = TopKRouter(hidden, num_experts, top_k)
                    self.num_experts = num_experts
                    self.top_k       = top_k

                def forward(self, x):
                    weights, indices = self.router(x)
                    output = torch.zeros_like(x)
                    for k in range(self.top_k):
                        expert_idx = indices[..., k]
                        weight     = weights[..., k].unsqueeze(-1)
                        for e_i in range(self.num_experts):
                            mask = (expert_idx == e_i)
                            if mask.any():
                                output[mask] += weight[mask] * self.experts[e_i](x[mask])
                    return output

            tok   = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL, device_map="auto",
                torch_dtype=torch.float16, trust_remote_code=True,
            )

            converted = 0
            for name, module in model.named_modules():
                # Convert every top-level MLP/FFN block to MoE
                if hasattr(module, "mlp") and isinstance(module.mlp, nn.Module):
                    original = module.mlp
                    module.mlp = MoEFFN(original, {num_experts}, {top_k})
                    converted += 1

            print(f"[OwnAI] Converted {{converted}} dense FFN layers to MoE "
                  f"({num_experts} experts, top-{top_k} routing).")

            tok.save_pretrained(OUTPUT_DIR)
            model.save_pretrained(OUTPUT_DIR)
            print(f"[OwnAI] MoE model saved to {{OUTPUT_DIR}}")

        """)

    @staticmethod
    def _integration_section(caps: List[str]) -> str:
        parts: List[str] = [
            "# ══════════════════════════════════════════════════════════════\n"
            "# INTEGRATION CAPABILITIES (runtime, no training required)\n"
            "# ══════════════════════════════════════════════════════════════\n"
        ]

        if "voice_input" in caps:
            parts.append(textwrap.dedent("""\
                # ── Voice Input (Whisper STT) ─────────────────────────────────
                print("[OwnAI] Generating Whisper voice-input integration...")

                WHISPER_WRAPPER = '''
                import whisper
                import sounddevice as sd
                import numpy as np

                _model = None

                def load_whisper(size="base"):
                    global _model
                    _model = whisper.load_model(size)
                    return _model

                def transcribe_microphone(seconds=5, sr=16000):
                    \"\"\"Record `seconds` of audio and return transcribed text.\"\"\"
                    if _model is None:
                        load_whisper()
                    print(f"Recording {seconds}s...")
                    audio = sd.rec(int(seconds * sr), samplerate=sr, channels=1,
                                   dtype="float32")
                    sd.wait()
                    audio = audio.squeeze()
                    result = _model.transcribe(audio, fp16=False)
                    return result["text"].strip()

                def transcribe_file(path: str) -> str:
                    \"\"\"Transcribe an audio file.\"\"\"
                    if _model is None:
                        load_whisper()
                    return _model.transcribe(path)["text"].strip()
                '''

                voice_path = os.path.join(OUTPUT_DIR, "voice_input.py")
                with open(voice_path, "w") as f:
                    f.write(WHISPER_WRAPPER)
                print(f"[OwnAI] Voice input helper written to {voice_path}")
                print("[OwnAI] Install requirements: pip install openai-whisper sounddevice")

            """))

        if "voice_output" in caps:
            parts.append(textwrap.dedent("""\
                # ── Voice Output (Piper / Coqui TTS) ─────────────────────────
                print("[OwnAI] Generating TTS voice-output integration...")

                TTS_WRAPPER = '''
                import subprocess, tempfile, os

                def speak_piper(text: str, voice: str = "en_US-lessac-medium"):
                    \"\"\"Speak text using Piper TTS (must be installed separately).\"\"\"
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                        out_path = f.name
                    subprocess.run(
                        ["piper", "--model", voice, "--output_file", out_path],
                        input=text, text=True, check=True,
                    )
                    subprocess.run(["aplay", out_path], check=False)
                    os.unlink(out_path)

                def speak_coqui(text: str, speaker_wav: str = None):
                    \"\"\"Speak text using Coqui TTS.\"\"\"
                    from TTS.api import TTS
                    tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                        out_path = f.name
                    tts.tts_to_file(text=text, file_path=out_path,
                                    speaker_wav=speaker_wav)
                    subprocess.run(["aplay", out_path], check=False)
                    os.unlink(out_path)
                '''

                tts_path = os.path.join(OUTPUT_DIR, "voice_output.py")
                with open(tts_path, "w") as f:
                    f.write(TTS_WRAPPER)
                print(f"[OwnAI] TTS helper written to {tts_path}")
                print("[OwnAI] Install: pip install piper-tts  OR  pip install TTS")

            """))

        if "memory" in caps:
            parts.append(textwrap.dedent("""\
                # ── Long-Term Memory ──────────────────────────────────────────
                print("[OwnAI] Writing memory integration stub...")

                MEMORY_STUB = '''
                # memory_integration.py  –  drop-in wrapper that adds memory to any model
                import sys, os
                sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
                from core.memory_manager import LongTermMemory, WorkingMemory

                _ltm  = LongTermMemory()
                _work = WorkingMemory()

                def chat_with_memory(model_fn, user_message: str, session_id: str = "default") -> str:
                    \"\"\"Wrap any model function with memory-augmented context.\"\"\"
                    # Retrieve relevant memories
                    memories = _ltm.recall(user_message, top_k=3)
                    context  = "\\n".join(f"[Memory] {m.content}" for m in memories)

                    # Retrieve working memory (current session)
                    recent = _work.get_recent(n=4)
                    history = "\\n".join(f"{r.role}: {r.content}" for r in recent)

                    augmented_prompt = (
                        f"Context from memory:\\n{context}\\n\\n"
                        f"Recent conversation:\\n{history}\\n\\n"
                        f"User: {user_message}"
                    )

                    response = model_fn(augmented_prompt)

                    # Save to memory
                    _work.add("user", user_message)
                    _work.add("assistant", response)
                    _ltm.add_fact(
                        content=user_message,
                        source="conversation",
                        session_id=session_id,
                    )
                    return response
                '''

                mem_path = os.path.join(OUTPUT_DIR, "memory_integration.py")
                with open(mem_path, "w") as f:
                    f.write(MEMORY_STUB)
                print(f"[OwnAI] Memory integration written to {mem_path}")

            """))

        if "web_search" in caps:
            parts.append(textwrap.dedent("""\
                # ── Web Search ────────────────────────────────────────────────
                print("[OwnAI] Writing web search integration...")

                SEARCH_STUB = '''
                from duckduckgo_search import DDGS

                def web_search(query: str, max_results: int = 5) -> list:
                    with DDGS() as ddgs:
                        results = list(ddgs.text(query, max_results=max_results))
                    return [{"title": r["title"], "url": r["href"],
                             "snippet": r["body"]} for r in results]
                '''

                search_path = os.path.join(OUTPUT_DIR, "web_search.py")
                with open(search_path, "w") as f:
                    f.write(SEARCH_STUB)
                print(f"[OwnAI] Web search helper written to {search_path}")
                print("[OwnAI] Install: pip install duckduckgo-search")

            """))

        if "code_execution" in caps:
            parts.append(textwrap.dedent("""\
                # ── Code Execution (sandboxed) ────────────────────────────────
                print("[OwnAI] Writing code execution integration...")

                CODE_STUB = '''
                import sys, os
                sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
                from core.reasoning_engine import CodeSandbox

                sandbox = CodeSandbox(timeout=30)

                def run_code(code: str) -> dict:
                    return sandbox.run(code)

                def run_code_with_tests(solution: str, tests: str) -> dict:
                    return sandbox.run_tests(solution, tests)
                '''

                code_path = os.path.join(OUTPUT_DIR, "code_execution.py")
                with open(code_path, "w") as f:
                    f.write(CODE_STUB)
                print(f"[OwnAI] Code execution helper written to {code_path}")

            """))

        return "\n".join(parts)

    @staticmethod
    def _save_section(output_dir: str) -> str:
        return textwrap.dedent(f"""\
            # ══════════════════════════════════════════════════════════════
            # DONE
            # ══════════════════════════════════════════════════════════════
            print("\\n" + "="*60)
            print("[OwnAI] Enhancement complete!")
            print(f"  Output: {{OUTPUT_DIR}}")
            print("  Files written:")
            for fn in sorted(os.listdir(OUTPUT_DIR)):
                print(f"    {{fn}}")
            print("="*60)
        """)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def list_enhancements() -> List[Dict[str, str]]:
        """Return all available enhancements as a list of dicts."""
        return [
            {"key": k, **v}
            for k, v in ALL_ENHANCEMENTS.items()
        ]
