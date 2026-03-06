"""Model builder – generates training scripts and config based on user selections."""
import os
import json
import textwrap
import logging

logger = logging.getLogger(__name__)


class ModelBuilder:
    """Generates training configurations and starter scripts for different AI types."""

    def build_config(self, form_data: dict) -> dict:
        """Convert wizard form data into a normalised model config dict."""
        config = {
            "name": form_data.get("name", "my_ai"),
            "ai_type": form_data.get("ai_type", "chatbot"),
            "base_model": form_data.get("base_model", "microsoft/phi-2"),
            "capabilities": form_data.getlist("capabilities") if hasattr(form_data, "getlist") else form_data.get("capabilities", []),
            "content_policy": form_data.get("content_policy", "standard"),
            "agentic": form_data.get("agentic", "off") == "on",
            "fine_tune_method": form_data.get("fine_tune_method", "lora"),
            "learning_rate": float(form_data.get("learning_rate", 2e-4)),
            "epochs": int(form_data.get("epochs", 3)),
            "batch_size": int(form_data.get("batch_size", 4)),
            "max_seq_len": int(form_data.get("max_seq_len", 512)),
            "quantization": form_data.get("quantization", "none"),
            "datasets": form_data.getlist("datasets") if hasattr(form_data, "getlist") else form_data.get("datasets", []),
            "hf_dataset": form_data.get("hf_dataset", ""),
            "web_urls": form_data.get("web_urls", ""),
            "output_dir": "",  # filled later when model record is created
        }
        return config

    def generate_training_script(self, config: dict, output_dir: str) -> str:
        """Generate a Python training script tailored to the model config."""
        ai_type = config.get("ai_type", "chatbot")
        script = self._get_script_template(ai_type, config, output_dir)
        script_path = os.path.join(output_dir, "train.py")
        os.makedirs(output_dir, exist_ok=True)
        with open(script_path, "w") as f:
            f.write(script)
        # Also save the config JSON
        with open(os.path.join(output_dir, "ownai_config.json"), "w") as f:
            json.dump(config, f, indent=2)
        return script_path

    # ------------------------------------------------------------------
    # Script templates
    # ------------------------------------------------------------------

    def _get_script_template(self, ai_type: str, config: dict, output_dir: str) -> str:
        dispatch = {
            "chatbot": self._chatbot_script,
            "image_generator": self._diffusion_script,
            "image_recognition": self._vision_script,
            "trading_bot": self._trading_script,
            "news_sentiment": self._sentiment_script,
            "medical_ai": self._vision_script,
            "space_ai": self._vision_script,
            "custom": self._chatbot_script,
        }
        fn = dispatch.get(ai_type, self._chatbot_script)
        return fn(config, output_dir)

    def _chatbot_script(self, config: dict, output_dir: str) -> str:
        lora_r = 16
        lora_alpha = 32
        use_lora = config.get("fine_tune_method", "lora") in ("lora", "qlora")
        use_4bit = config.get("quantization", "none") in ("4bit",) or config.get("fine_tune_method") == "qlora"
        return textwrap.dedent(f"""
            #!/usr/bin/env python3
            \"\"\"OwnAI – Auto-generated chatbot training script.\"\"\"
            import json, os, sys
            import torch
            from transformers import (
                AutoTokenizer, AutoModelForCausalLM,
                TrainingArguments, BitsAndBytesConfig,
            )
            from datasets import load_dataset, Dataset
            from trl import SFTTrainer

            CONFIG_PATH = os.path.join(os.path.dirname(__file__), "ownai_config.json")
            with open(CONFIG_PATH) as f:
                cfg = json.load(f)

            OUTPUT_DIR = r"{output_dir}"
            BASE_MODEL = cfg.get("base_model", "microsoft/phi-2")
            EPOCHS = cfg.get("epochs", 3)
            LR = cfg.get("learning_rate", 2e-4)
            BATCH = cfg.get("batch_size", 4)
            MAX_LEN = cfg.get("max_seq_len", 512)

            print(f"[OwnAI] Loading base model: {{BASE_MODEL}}")

            bnb_config = None
            {"bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.float16)" if use_4bit else ""}

            tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
            tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )

            {"# Apply LoRA adapters" if use_lora else ""}
            {"from peft import LoraConfig, get_peft_model" if use_lora else ""}
            {"lora_cfg = LoraConfig(r=" + str(lora_r) + ", lora_alpha=" + str(lora_alpha) + ", target_modules='all-linear', lora_dropout=0.05, bias='none', task_type='CAUSAL_LM')" if use_lora else ""}
            {"model = get_peft_model(model, lora_cfg)" if use_lora else ""}

            # ---- Load dataset ----
            hf_ds = cfg.get("hf_dataset", "")
            local_datasets = cfg.get("datasets", [])

            def load_training_data():
                texts = []
                if hf_ds:
                    print(f"[OwnAI] Loading HuggingFace dataset: {{hf_ds}}")
                    ds = load_dataset(hf_ds, split="train")
                    col = [c for c in ds.column_names if "text" in c.lower()]
                    if col:
                        texts += ds[col[0]]
                for path in local_datasets:
                    if os.path.exists(path):
                        with open(path, encoding="utf-8") as f:
                            for line in f:
                                line = line.strip()
                                if line:
                                    try:
                                        obj = json.loads(line)
                                        texts.append(obj.get("text", str(obj)))
                                    except Exception:
                                        texts.append(line)
                if not texts:
                    texts = ["Hello, I am an AI assistant.", "How can I help you today?"]
                return Dataset.from_dict({{"text": texts}})

            train_dataset = load_training_data()
            print(f"[OwnAI] Dataset size: {{len(train_dataset)}} samples")

            training_args = TrainingArguments(
                output_dir=OUTPUT_DIR,
                num_train_epochs=EPOCHS,
                per_device_train_batch_size=BATCH,
                learning_rate=LR,
                fp16=torch.cuda.is_available(),
                logging_steps=10,
                save_steps=100,
                save_total_limit=2,
                report_to="none",
            )

            trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=train_dataset,
                dataset_text_field="text",
                max_seq_length=MAX_LEN,
                args=training_args,
            )

            print("[OwnAI] Starting training...")
            trainer.train()
            trainer.save_model(OUTPUT_DIR)
            print(f"[OwnAI] Model saved to {{OUTPUT_DIR}}")
        """).strip() + "\n"

    def _diffusion_script(self, config: dict, output_dir: str) -> str:
        return textwrap.dedent(f"""
            #!/usr/bin/env python3
            \"\"\"OwnAI – Auto-generated image generation (diffusion) training script.\"\"\"
            import json, os
            from diffusers import StableDiffusionPipeline
            import torch

            CONFIG_PATH = os.path.join(os.path.dirname(__file__), "ownai_config.json")
            with open(CONFIG_PATH) as f:
                cfg = json.load(f)

            OUTPUT_DIR = r"{output_dir}"
            BASE_MODEL = cfg.get("base_model", "runwayml/stable-diffusion-v1-5")

            print(f"[OwnAI] Loading diffusion model: {{BASE_MODEL}}")
            pipe = StableDiffusionPipeline.from_pretrained(
                BASE_MODEL, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            if torch.cuda.is_available():
                pipe = pipe.to("cuda")

            print("[OwnAI] Model loaded. Fine-tuning with DreamBooth/LoRA requires image datasets.")
            print("[OwnAI] Place your images in:", cfg.get("datasets", []))
            print("[OwnAI] Use 'diffusers' DreamBooth training scripts for full fine-tuning.")
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            print(f"[OwnAI] Config saved to {{OUTPUT_DIR}}")
        """).strip() + "\n"

    def _vision_script(self, config: dict, output_dir: str) -> str:
        return textwrap.dedent(f"""
            #!/usr/bin/env python3
            \"\"\"OwnAI – Auto-generated vision model training script.\"\"\"
            import json, os
            import torch
            from transformers import ViTForImageClassification, ViTFeatureExtractor, TrainingArguments, Trainer
            from datasets import load_dataset

            CONFIG_PATH = os.path.join(os.path.dirname(__file__), "ownai_config.json")
            with open(CONFIG_PATH) as f:
                cfg = json.load(f)

            OUTPUT_DIR = r"{output_dir}"
            BASE_MODEL = cfg.get("base_model", "google/vit-base-patch16-224")
            EPOCHS = cfg.get("epochs", 3)
            LR = cfg.get("learning_rate", 2e-4)
            BATCH = cfg.get("batch_size", 16)

            print(f"[OwnAI] Loading vision model: {{BASE_MODEL}}")
            feature_extractor = ViTFeatureExtractor.from_pretrained(BASE_MODEL)
            model = ViTForImageClassification.from_pretrained(BASE_MODEL, ignore_mismatched_sizes=True)

            hf_ds = cfg.get("hf_dataset", "")
            if hf_ds:
                dataset = load_dataset(hf_ds)
                print(f"[OwnAI] Dataset loaded: {{hf_ds}}")
            else:
                print("[OwnAI] No dataset configured – upload images in the OwnAI UI.")

            training_args = TrainingArguments(
                output_dir=OUTPUT_DIR,
                num_train_epochs=EPOCHS,
                per_device_train_batch_size=BATCH,
                learning_rate=LR,
                save_total_limit=2,
                report_to="none",
            )
            print("[OwnAI] Training args ready. Attach your image dataset to proceed.")
            os.makedirs(OUTPUT_DIR, exist_ok=True)
        """).strip() + "\n"

    def _trading_script(self, config: dict, output_dir: str) -> str:
        return textwrap.dedent(f"""
            #!/usr/bin/env python3
            \"\"\"OwnAI – Auto-generated trading bot training script.\"\"\"
            import json, os
            import numpy as np
            import torch
            import torch.nn as nn

            CONFIG_PATH = os.path.join(os.path.dirname(__file__), "ownai_config.json")
            with open(CONFIG_PATH) as f:
                cfg = json.load(f)

            OUTPUT_DIR = r"{output_dir}"
            EPOCHS = cfg.get("epochs", 50)
            LR = cfg.get("learning_rate", 1e-3)

            class TradingLSTM(nn.Module):
                def __init__(self, input_size=5, hidden=128, layers=2, output=3):
                    super().__init__()
                    self.lstm = nn.LSTM(input_size, hidden, layers, batch_first=True, dropout=0.2)
                    self.fc = nn.Linear(hidden, output)  # Buy / Hold / Sell

                def forward(self, x):
                    out, _ = self.lstm(x)
                    return self.fc(out[:, -1, :])

            model = TradingLSTM()
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)
            criterion = nn.CrossEntropyLoss()

            print("[OwnAI] Trading LSTM model created.")
            print("[OwnAI] Provide OHLCV CSV data via the OwnAI dataset uploader to train.")
            print(f"[OwnAI] Model params: {{sum(p.numel() for p in model.parameters()):,}}")
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "model_init.pt"))
            print("[OwnAI] Initial model saved.")
        """).strip() + "\n"

    def _sentiment_script(self, config: dict, output_dir: str) -> str:
        return textwrap.dedent(f"""
            #!/usr/bin/env python3
            \"\"\"OwnAI – Auto-generated news/sentiment analysis training script.\"\"\"
            import json, os
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
            from datasets import load_dataset

            CONFIG_PATH = os.path.join(os.path.dirname(__file__), "ownai_config.json")
            with open(CONFIG_PATH) as f:
                cfg = json.load(f)

            OUTPUT_DIR = r"{output_dir}"
            BASE_MODEL = cfg.get("base_model", "distilbert-base-uncased")
            EPOCHS = cfg.get("epochs", 3)
            LR = cfg.get("learning_rate", 2e-5)
            BATCH = cfg.get("batch_size", 16)

            print(f"[OwnAI] Loading model: {{BASE_MODEL}}")
            tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
            model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=3)

            hf_ds = cfg.get("hf_dataset", "")
            if hf_ds:
                dataset = load_dataset(hf_ds)
                print(f"[OwnAI] Loaded dataset: {{hf_ds}}")

            training_args = TrainingArguments(
                output_dir=OUTPUT_DIR,
                num_train_epochs=EPOCHS,
                per_device_train_batch_size=BATCH,
                learning_rate=LR,
                save_total_limit=2,
                report_to="none",
            )
            print("[OwnAI] Sentiment model ready for training.")
            os.makedirs(OUTPUT_DIR, exist_ok=True)
        """).strip() + "\n"
