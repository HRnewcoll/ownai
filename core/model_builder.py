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
            # Free-text niche description (used when ai_type == "custom")
            "custom_niche": form_data.get("custom_niche", "").strip(),
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
    # Script template dispatch
    # ------------------------------------------------------------------

    def _get_script_template(self, ai_type: str, config: dict, output_dir: str) -> str:
        dispatch = {
            # Text & Language
            "chatbot": self._chatbot_script,
            "code_ai": self._code_ai_script,
            "translator": self._translation_script,
            "summarizer": self._summarizer_script,
            "text_classifier": self._classifier_script,
            "creative_writer": self._chatbot_script,  # same LLM fine-tune path
            # Vision & Images
            "image_generator": self._diffusion_script,
            "image_recognition": self._vision_script,
            "object_detector": self._object_detection_script,
            "document_ai": self._vision_script,
            "face_ai": self._vision_script,
            # Audio & Speech
            "speech_to_text": self._audio_script,
            "audio_classifier": self._audio_script,
            "music_ai": self._audio_script,
            # Data & Analytics
            "trading_bot": self._trading_script,
            "news_sentiment": self._sentiment_script,
            "anomaly_detector": self._anomaly_script,
            "data_analyst": self._tabular_script,
            "recommendation": self._recommendation_script,
            # Domain Specific
            "medical_ai": self._vision_script,
            "space_ai": self._vision_script,
            "legal_ai": self._chatbot_script,
            "education_ai": self._chatbot_script,
            "customer_support": self._chatbot_script,
            "agriculture_ai": self._vision_script,
            "game_ai": self._rl_script,
            "cyber_ai": self._classifier_script,
            # Custom
            "custom": self._custom_script,
        }
        fn = dispatch.get(ai_type, self._chatbot_script)
        return fn(config, output_dir)

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _bnb_and_lora_lines(self, config: dict) -> tuple[str, str, str]:
        """Return (bnb_import, bnb_config_line, lora_lines) strings."""
        use_lora = config.get("fine_tune_method", "lora") in ("lora", "qlora")
        use_4bit = config.get("quantization", "none") == "4bit" or config.get("fine_tune_method") == "qlora"

        bnb_line = (
            "bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', "
            "bnb_4bit_compute_dtype=torch.float16)"
            if use_4bit else "bnb_config = None"
        )
        lora_lines = ""
        if use_lora:
            lora_lines = (
                "from peft import LoraConfig, get_peft_model\n"
                "lora_cfg = LoraConfig(r=16, lora_alpha=32, target_modules='all-linear', "
                "lora_dropout=0.05, bias='none', task_type='CAUSAL_LM')\n"
                "model = get_peft_model(model, lora_cfg)\n"
            )
        return bnb_line, lora_lines

    # ------------------------------------------------------------------
    # Text & Language scripts
    # ------------------------------------------------------------------

    def _chatbot_script(self, config: dict, output_dir: str) -> str:
        bnb_line, lora_lines = self._bnb_and_lora_lines(config)
        niche = config.get("custom_niche", "")
        niche_comment = f"# Niche: {niche}" if niche else ""
        return textwrap.dedent(f"""
            #!/usr/bin/env python3
            \"\"\"OwnAI – Auto-generated LLM fine-tuning script.\"\"\"
            {niche_comment}
            import json, os
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
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
            {bnb_line}

            tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
            tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL, quantization_config=bnb_config, device_map="auto", trust_remote_code=True
            )

            {lora_lines}

            def load_training_data():
                texts = []
                hf_ds = cfg.get("hf_dataset", "")
                if hf_ds:
                    print(f"[OwnAI] Loading HuggingFace dataset: {{hf_ds}}")
                    ds = load_dataset(hf_ds, split="train")
                    col = next((c for c in ds.column_names if "text" in c.lower()), None)
                    if col:
                        texts += [str(v) for v in ds[col] if v]
                for path in cfg.get("datasets", []):
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
                output_dir=OUTPUT_DIR, num_train_epochs=EPOCHS,
                per_device_train_batch_size=BATCH, learning_rate=LR,
                fp16=torch.cuda.is_available(), logging_steps=10,
                save_steps=100, save_total_limit=2, report_to="none",
            )

            trainer = SFTTrainer(
                model=model, tokenizer=tokenizer, train_dataset=train_dataset,
                dataset_text_field="text", max_seq_length=MAX_LEN, args=training_args,
            )

            print("[OwnAI] Starting training...")
            trainer.train()
            trainer.save_model(OUTPUT_DIR)
            print(f"[OwnAI] Model saved to {{OUTPUT_DIR}}")
        """).strip() + "\n"

    def _code_ai_script(self, config: dict, output_dir: str) -> str:
        bnb_line, lora_lines = self._bnb_and_lora_lines(config)
        return textwrap.dedent(f"""
            #!/usr/bin/env python3
            \"\"\"OwnAI – Auto-generated code AI training script.\"\"\"
            import json, os
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
            from datasets import load_dataset, Dataset
            from trl import SFTTrainer

            CONFIG_PATH = os.path.join(os.path.dirname(__file__), "ownai_config.json")
            with open(CONFIG_PATH) as f:
                cfg = json.load(f)

            OUTPUT_DIR = r"{output_dir}"
            BASE_MODEL = cfg.get("base_model", "bigcode/starcoder2-3b")
            EPOCHS = cfg.get("epochs", 3)
            LR = cfg.get("learning_rate", 2e-4)
            BATCH = cfg.get("batch_size", 4)
            MAX_LEN = cfg.get("max_seq_len", 1024)  # code needs longer context

            print(f"[OwnAI] Loading code model: {{BASE_MODEL}}")
            {bnb_line}

            tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
            tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL, quantization_config=bnb_config, device_map="auto", trust_remote_code=True
            )

            {lora_lines}

            def load_code_data():
                texts = []
                hf_ds = cfg.get("hf_dataset", "")
                if hf_ds:
                    ds = load_dataset(hf_ds, split="train")
                    # Try common code dataset columns
                    col = next((c for c in ds.column_names if c in ("content", "code", "text", "func_code_string")), None)
                    if col:
                        texts += [str(t) for t in ds[col] if isinstance(t, str) and len(t) > 20]
                for path in cfg.get("datasets", []):
                    if os.path.exists(path):
                        with open(path, encoding="utf-8") as f:
                            texts.append(f.read())
                if not texts:
                    texts = ["def hello_world():\\n    print('Hello, World!')"]
                return Dataset.from_dict({{"text": texts}})

            train_dataset = load_code_data()
            print(f"[OwnAI] Code dataset: {{len(train_dataset)}} samples")

            training_args = TrainingArguments(
                output_dir=OUTPUT_DIR, num_train_epochs=EPOCHS,
                per_device_train_batch_size=BATCH, learning_rate=LR,
                fp16=torch.cuda.is_available(), logging_steps=10,
                save_steps=100, save_total_limit=2, report_to="none",
            )

            trainer = SFTTrainer(
                model=model, tokenizer=tokenizer, train_dataset=train_dataset,
                dataset_text_field="text", max_seq_length=MAX_LEN, args=training_args,
            )

            print("[OwnAI] Training code AI...")
            trainer.train()
            trainer.save_model(OUTPUT_DIR)
            print(f"[OwnAI] Code AI saved to {{OUTPUT_DIR}}")
        """).strip() + "\n"

    def _translation_script(self, config: dict, output_dir: str) -> str:
        return textwrap.dedent(f"""
            #!/usr/bin/env python3
            \"\"\"OwnAI – Auto-generated translation model training script.\"\"\"
            import json, os
            import torch
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
            from datasets import load_dataset, Dataset

            CONFIG_PATH = os.path.join(os.path.dirname(__file__), "ownai_config.json")
            with open(CONFIG_PATH) as f:
                cfg = json.load(f)

            OUTPUT_DIR = r"{output_dir}"
            BASE_MODEL = cfg.get("base_model", "Helsinki-NLP/opus-mt-en-de")
            EPOCHS = cfg.get("epochs", 3)
            LR = cfg.get("learning_rate", 5e-5)
            BATCH = cfg.get("batch_size", 16)

            print(f"[OwnAI] Loading translation model: {{BASE_MODEL}}")
            tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
            model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)

            hf_ds = cfg.get("hf_dataset", "")
            if hf_ds:
                dataset = load_dataset(hf_ds)
                print(f"[OwnAI] Dataset loaded: {{hf_ds}}")
            else:
                print("[OwnAI] No dataset – provide parallel text pairs via OwnAI datasets.")

            def preprocess(examples):
                translation = examples.get("translation", {{}})
                if isinstance(translation, list):
                    translation = translation[0] if translation else {{}}
                src = examples.get("en", translation.get("en", ""))
                tgt = examples.get("de", translation.get("de", ""))
                model_inputs = tokenizer(src, max_length=512, truncation=True, padding="max_length")
                labels = tokenizer(tgt, max_length=512, truncation=True, padding="max_length")
                model_inputs["labels"] = labels["input_ids"]
                return model_inputs

            training_args = Seq2SeqTrainingArguments(
                output_dir=OUTPUT_DIR, num_train_epochs=EPOCHS,
                per_device_train_batch_size=BATCH, learning_rate=LR,
                fp16=torch.cuda.is_available(), save_total_limit=2,
                predict_with_generate=True, report_to="none",
            )
            print("[OwnAI] Translation model ready. Attach parallel corpus to proceed.")
            os.makedirs(OUTPUT_DIR, exist_ok=True)
        """).strip() + "\n"

    def _summarizer_script(self, config: dict, output_dir: str) -> str:
        return textwrap.dedent(f"""
            #!/usr/bin/env python3
            \"\"\"OwnAI – Auto-generated summarisation model training script.\"\"\"
            import json, os
            import torch
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
            from datasets import load_dataset

            CONFIG_PATH = os.path.join(os.path.dirname(__file__), "ownai_config.json")
            with open(CONFIG_PATH) as f:
                cfg = json.load(f)

            OUTPUT_DIR = r"{output_dir}"
            BASE_MODEL = cfg.get("base_model", "facebook/bart-large-cnn")
            EPOCHS = cfg.get("epochs", 3)
            LR = cfg.get("learning_rate", 5e-5)
            BATCH = cfg.get("batch_size", 8)

            print(f"[OwnAI] Loading summarisation model: {{BASE_MODEL}}")
            tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
            model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)

            hf_ds = cfg.get("hf_dataset", "")
            if hf_ds:
                dataset = load_dataset(hf_ds)
                print(f"[OwnAI] Dataset: {{hf_ds}}")
            else:
                print("[OwnAI] No dataset – provide article/summary pairs via OwnAI datasets.")

            training_args = Seq2SeqTrainingArguments(
                output_dir=OUTPUT_DIR, num_train_epochs=EPOCHS,
                per_device_train_batch_size=BATCH, learning_rate=LR,
                fp16=torch.cuda.is_available(), save_total_limit=2,
                predict_with_generate=True, report_to="none",
            )
            print("[OwnAI] Summariser ready. Provide article/summary dataset pairs.")
            os.makedirs(OUTPUT_DIR, exist_ok=True)
        """).strip() + "\n"

    def _classifier_script(self, config: dict, output_dir: str) -> str:
        """Sequence classification (text classifier, cyber_ai, sentiment, etc.)."""
        return textwrap.dedent(f"""
            #!/usr/bin/env python3
            \"\"\"OwnAI – Auto-generated text classification training script.\"\"\"
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
            NUM_LABELS = 2  # adjust for your classification task

            print(f"[OwnAI] Loading classification model: {{BASE_MODEL}}")
            tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
            model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=NUM_LABELS)

            hf_ds = cfg.get("hf_dataset", "")
            if hf_ds:
                dataset = load_dataset(hf_ds)
                print(f"[OwnAI] Dataset: {{hf_ds}}")
            else:
                print("[OwnAI] No dataset – upload labelled text samples via OwnAI datasets.")

            def tokenize(batch):
                return tokenizer(batch.get("text", batch.get("sentence", "")), truncation=True, padding="max_length", max_length=256)

            training_args = TrainingArguments(
                output_dir=OUTPUT_DIR, num_train_epochs=EPOCHS,
                per_device_train_batch_size=BATCH, learning_rate=LR,
                fp16=torch.cuda.is_available(), save_total_limit=2, report_to="none",
            )
            print("[OwnAI] Text classifier ready. Attach labelled dataset to proceed.")
            os.makedirs(OUTPUT_DIR, exist_ok=True)
        """).strip() + "\n"

    # ------------------------------------------------------------------
    # Vision scripts
    # ------------------------------------------------------------------

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

            print("[OwnAI] Model loaded. Fine-tune with DreamBooth/LoRA using your image dataset.")
            print("[OwnAI] Datasets:", cfg.get("datasets", []))
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            print(f"[OwnAI] Config saved to {{OUTPUT_DIR}}")
        """).strip() + "\n"

    def _vision_script(self, config: dict, output_dir: str) -> str:
        return textwrap.dedent(f"""
            #!/usr/bin/env python3
            \"\"\"OwnAI – Auto-generated vision model training script.\"\"\"
            import json, os
            import torch
            from transformers import AutoFeatureExtractor, AutoModelForImageClassification, TrainingArguments, Trainer
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
            feature_extractor = AutoFeatureExtractor.from_pretrained(BASE_MODEL)
            model = AutoModelForImageClassification.from_pretrained(BASE_MODEL, ignore_mismatched_sizes=True)

            hf_ds = cfg.get("hf_dataset", "")
            if hf_ds:
                dataset = load_dataset(hf_ds)
                print(f"[OwnAI] Dataset loaded: {{hf_ds}}")
            else:
                print("[OwnAI] No dataset – upload labelled images via OwnAI datasets.")

            training_args = TrainingArguments(
                output_dir=OUTPUT_DIR, num_train_epochs=EPOCHS,
                per_device_train_batch_size=BATCH, learning_rate=LR,
                save_total_limit=2, report_to="none",
            )
            print("[OwnAI] Vision model ready. Attach image dataset to proceed.")
            os.makedirs(OUTPUT_DIR, exist_ok=True)
        """).strip() + "\n"

    def _object_detection_script(self, config: dict, output_dir: str) -> str:
        return textwrap.dedent(f"""
            #!/usr/bin/env python3
            \"\"\"OwnAI – Auto-generated object detection training script.\"\"\"
            import json, os
            import torch
            from transformers import AutoImageProcessor, AutoModelForObjectDetection, TrainingArguments, Trainer
            from datasets import load_dataset

            CONFIG_PATH = os.path.join(os.path.dirname(__file__), "ownai_config.json")
            with open(CONFIG_PATH) as f:
                cfg = json.load(f)

            OUTPUT_DIR = r"{output_dir}"
            BASE_MODEL = cfg.get("base_model", "facebook/detr-resnet-50")
            EPOCHS = cfg.get("epochs", 10)
            LR = cfg.get("learning_rate", 1e-4)
            BATCH = cfg.get("batch_size", 4)

            print(f"[OwnAI] Loading object detection model: {{BASE_MODEL}}")
            processor = AutoImageProcessor.from_pretrained(BASE_MODEL)
            model = AutoModelForObjectDetection.from_pretrained(BASE_MODEL)

            hf_ds = cfg.get("hf_dataset", "")
            if hf_ds:
                dataset = load_dataset(hf_ds)
                print(f"[OwnAI] Dataset: {{hf_ds}}")
            else:
                print("[OwnAI] No dataset – upload COCO-format or Pascal VOC images/annotations.")

            training_args = TrainingArguments(
                output_dir=OUTPUT_DIR, num_train_epochs=EPOCHS,
                per_device_train_batch_size=BATCH, learning_rate=LR,
                save_total_limit=2, report_to="none",
            )
            print("[OwnAI] Object detection model ready. Attach annotated image dataset.")
            os.makedirs(OUTPUT_DIR, exist_ok=True)
        """).strip() + "\n"

    # ------------------------------------------------------------------
    # Audio / Speech scripts
    # ------------------------------------------------------------------

    def _audio_script(self, config: dict, output_dir: str) -> str:
        ai_type = config.get("ai_type", "speech_to_text")
        base_model_default = {
            "speech_to_text": "openai/whisper-small",
            "audio_classifier": "facebook/wav2vec2-base",
            "music_ai": "facebook/musicgen-small",
        }.get(ai_type, "openai/whisper-small")
        return textwrap.dedent(f"""
            #!/usr/bin/env python3
            \"\"\"OwnAI – Auto-generated audio/speech AI training script.\"\"\"
            import json, os
            import torch
            from transformers import AutoProcessor, AutoModelForAudioClassification, TrainingArguments, Trainer
            from datasets import load_dataset, Audio

            CONFIG_PATH = os.path.join(os.path.dirname(__file__), "ownai_config.json")
            with open(CONFIG_PATH) as f:
                cfg = json.load(f)

            OUTPUT_DIR = r"{output_dir}"
            BASE_MODEL = cfg.get("base_model", "{base_model_default}")
            EPOCHS = cfg.get("epochs", 5)
            LR = cfg.get("learning_rate", 3e-5)
            BATCH = cfg.get("batch_size", 8)
            AI_TYPE = cfg.get("ai_type", "audio_classifier")

            print(f"[OwnAI] Loading audio model: {{BASE_MODEL}}")

            if AI_TYPE == "speech_to_text":
                from transformers import WhisperProcessor, WhisperForConditionalGeneration
                processor = WhisperProcessor.from_pretrained(BASE_MODEL)
                model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL)
                print("[OwnAI] Whisper speech-to-text model loaded.")
            elif AI_TYPE == "music_ai":
                from transformers import MusicgenProcessor, MusicgenForConditionalGeneration
                processor = MusicgenProcessor.from_pretrained(BASE_MODEL)
                model = MusicgenForConditionalGeneration.from_pretrained(BASE_MODEL)
                print("[OwnAI] MusicGen model loaded.")
            else:
                processor = AutoProcessor.from_pretrained(BASE_MODEL)
                model = AutoModelForAudioClassification.from_pretrained(BASE_MODEL)
                print("[OwnAI] Audio classifier model loaded.")

            hf_ds = cfg.get("hf_dataset", "")
            if hf_ds:
                dataset = load_dataset(hf_ds)
                print(f"[OwnAI] Audio dataset: {{hf_ds}}")
            else:
                print("[OwnAI] No dataset – upload audio files (.wav/.mp3) via OwnAI datasets.")

            training_args = TrainingArguments(
                output_dir=OUTPUT_DIR, num_train_epochs=EPOCHS,
                per_device_train_batch_size=BATCH, learning_rate=LR,
                save_total_limit=2, report_to="none",
            )
            print("[OwnAI] Audio model ready. Attach audio dataset to proceed.")
            os.makedirs(OUTPUT_DIR, exist_ok=True)
        """).strip() + "\n"

    # ------------------------------------------------------------------
    # Data & Analytics scripts
    # ------------------------------------------------------------------

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
            BASE_MODEL = cfg.get("base_model", "scratch_lstm")

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
            print("[OwnAI] Provide OHLCV CSV data (open,high,low,close,volume) via OwnAI datasets.")
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
                output_dir=OUTPUT_DIR, num_train_epochs=EPOCHS,
                per_device_train_batch_size=BATCH, learning_rate=LR,
                save_total_limit=2, report_to="none",
            )
            print("[OwnAI] Sentiment model ready for training.")
            os.makedirs(OUTPUT_DIR, exist_ok=True)
        """).strip() + "\n"

    def _anomaly_script(self, config: dict, output_dir: str) -> str:
        return textwrap.dedent(f"""
            #!/usr/bin/env python3
            \"\"\"OwnAI – Auto-generated anomaly detection training script.\"\"\"
            import json, os
            import numpy as np
            import torch
            import torch.nn as nn

            CONFIG_PATH = os.path.join(os.path.dirname(__file__), "ownai_config.json")
            with open(CONFIG_PATH) as f:
                cfg = json.load(f)

            OUTPUT_DIR = r"{output_dir}"
            EPOCHS = cfg.get("epochs", 30)
            LR = cfg.get("learning_rate", 1e-3)
            BATCH = cfg.get("batch_size", 64)

            class Autoencoder(nn.Module):
                def __init__(self, input_dim=32, latent_dim=8):
                    super().__init__()
                    self.encoder = nn.Sequential(
                        nn.Linear(input_dim, 16), nn.ReLU(), nn.Linear(16, latent_dim)
                    )
                    self.decoder = nn.Sequential(
                        nn.Linear(latent_dim, 16), nn.ReLU(), nn.Linear(16, input_dim)
                    )

                def forward(self, x):
                    return self.decoder(self.encoder(x))

            model = Autoencoder()
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)
            criterion = nn.MSELoss()

            print("[OwnAI] Anomaly detection autoencoder created.")
            print("[OwnAI] Upload normal-state data (CSV/JSON) for training.")
            print("[OwnAI] At inference, high reconstruction error = anomaly detected.")
            print(f"[OwnAI] Model params: {{sum(p.numel() for p in model.parameters()):,}}")
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "model_init.pt"))
            print("[OwnAI] Anomaly detector initialised.")
        """).strip() + "\n"

    def _tabular_script(self, config: dict, output_dir: str) -> str:
        return textwrap.dedent(f"""
            #!/usr/bin/env python3
            \"\"\"OwnAI – Auto-generated tabular data / forecasting training script.\"\"\"
            import json, os
            import numpy as np

            CONFIG_PATH = os.path.join(os.path.dirname(__file__), "ownai_config.json")
            with open(CONFIG_PATH) as f:
                cfg = json.load(f)

            OUTPUT_DIR = r"{output_dir}"
            BASE_MODEL = cfg.get("base_model", "scratch_xgboost")
            EPOCHS = cfg.get("epochs", 100)

            print(f"[OwnAI] Data analyst / forecasting AI — model type: {{BASE_MODEL}}")

            if BASE_MODEL == "scratch_xgboost":
                try:
                    import xgboost as xgb
                    print("[OwnAI] XGBoost ready. Upload CSV with features + target column.")
                    params = dict(n_estimators=EPOCHS, learning_rate=cfg.get("learning_rate", 0.1),
                                  max_depth=6, tree_method="hist")
                    model = xgb.XGBClassifier(**params)
                    print("[OwnAI] XGBoost model configured:", params)
                except ImportError:
                    print("[OwnAI] XGBoost not installed – run: pip install xgboost")
            else:
                import torch, torch.nn as nn
                class ForecastLSTM(nn.Module):
                    def __init__(self, input_size=10, hidden=64, output=1):
                        super().__init__()
                        self.lstm = nn.LSTM(input_size, hidden, 2, batch_first=True)
                        self.fc = nn.Linear(hidden, output)
                    def forward(self, x):
                        out, _ = self.lstm(x)
                        return self.fc(out[:, -1, :])
                model = ForecastLSTM()
                torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "model_init.pt"))
                print("[OwnAI] LSTM forecaster initialised.")

            os.makedirs(OUTPUT_DIR, exist_ok=True)
            print("[OwnAI] Upload your CSV dataset via OwnAI datasets page.")
        """).strip() + "\n"

    def _recommendation_script(self, config: dict, output_dir: str) -> str:
        return textwrap.dedent(f"""
            #!/usr/bin/env python3
            \"\"\"OwnAI – Auto-generated recommendation system training script.\"\"\"
            import json, os
            import numpy as np
            import torch
            import torch.nn as nn

            CONFIG_PATH = os.path.join(os.path.dirname(__file__), "ownai_config.json")
            with open(CONFIG_PATH) as f:
                cfg = json.load(f)

            OUTPUT_DIR = r"{output_dir}"
            EPOCHS = cfg.get("epochs", 20)
            LR = cfg.get("learning_rate", 1e-3)
            EMBEDDING_DIM = 64

            class NeuralCF(nn.Module):
                def __init__(self, n_users=10000, n_items=10000, emb_dim=EMBEDDING_DIM):
                    super().__init__()
                    self.user_emb = nn.Embedding(n_users, emb_dim)
                    self.item_emb = nn.Embedding(n_items, emb_dim)
                    self.fc = nn.Sequential(
                        nn.Linear(emb_dim * 2, 128), nn.ReLU(),
                        nn.Linear(128, 64), nn.ReLU(),
                        nn.Linear(64, 1), nn.Sigmoid()
                    )

                def forward(self, user, item):
                    u = self.user_emb(user)
                    i = self.item_emb(item)
                    return self.fc(torch.cat([u, i], dim=-1)).squeeze()

            model = NeuralCF()
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)
            criterion = nn.BCELoss()

            print("[OwnAI] Neural Collaborative Filtering model created.")
            print("[OwnAI] Upload user-item interaction data (CSV: user_id, item_id, rating).")
            print(f"[OwnAI] Model params: {{sum(p.numel() for p in model.parameters()):,}}")
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "model_init.pt"))
            print("[OwnAI] Recommendation model initialised.")
        """).strip() + "\n"

    def _rl_script(self, config: dict, output_dir: str) -> str:
        return textwrap.dedent(f"""
            #!/usr/bin/env python3
            \"\"\"OwnAI – Auto-generated reinforcement learning (game AI) training script.\"\"\"
            import json, os
            import torch
            import torch.nn as nn
            import numpy as np

            CONFIG_PATH = os.path.join(os.path.dirname(__file__), "ownai_config.json")
            with open(CONFIG_PATH) as f:
                cfg = json.load(f)

            OUTPUT_DIR = r"{output_dir}"
            BASE_MODEL = cfg.get("base_model", "scratch_ppo")
            EPOCHS = cfg.get("epochs", 1000)
            LR = cfg.get("learning_rate", 3e-4)

            print(f"[OwnAI] Game AI / RL agent — algorithm: {{BASE_MODEL}}")

            class PolicyNetwork(nn.Module):
                def __init__(self, obs_dim=4, action_dim=2):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(obs_dim, 64), nn.Tanh(),
                        nn.Linear(64, 64), nn.Tanh(),
                        nn.Linear(64, action_dim),
                    )
                    self.value_head = nn.Linear(64, 1)

                def forward(self, x):
                    features = self.net[:-1](x)
                    actions = self.net[-1](features)
                    value = self.value_head(features)
                    return actions, value

            policy = PolicyNetwork()
            optimizer = torch.optim.Adam(policy.parameters(), lr=LR)

            print("[OwnAI] PPO policy network created.")
            print("[OwnAI] Connect to your game environment (OpenAI Gym, Unity ML-Agents, custom env).")
            print(f"[OwnAI] Policy params: {{sum(p.numel() for p in policy.parameters()):,}}")
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            torch.save(policy.state_dict(), os.path.join(OUTPUT_DIR, "policy_init.pt"))
            print("[OwnAI] RL agent initialised.")
        """).strip() + "\n"

    # ------------------------------------------------------------------
    # Custom / Any Niche script
    # ------------------------------------------------------------------

    def _custom_script(self, config: dict, output_dir: str) -> str:
        """Smart custom script – picks best template based on niche description keywords.

        More specific / overlap-prone checks are handled first to avoid false matches.
        """
        niche = config.get("custom_niche", "").lower()

        # Trading first (has "predict" overlap with tabular, needs to win)
        if any(k in niche for k in ("trade", "trading", "stock", "crypto", "forex",
                                     "market price", "price prediction", "ohlcv")):
            return self._trading_script(config, output_dir)
        # Anomaly (has "detect" overlap with vision)
        if any(k in niche for k in ("anomal", "fraud", "outlier", "unusual pattern", "intrusion")):
            return self._anomaly_script(config, output_dir)
        # Recommendation
        if any(k in niche for k in ("recommend", "suggest", "personali", "user-item", "collaborative")):
            return self._recommendation_script(config, output_dir)
        # Game / RL
        if any(k in niche for k in ("game agent", "rl agent", "reinforcement", "npc", "play chess", "play game")):
            return self._rl_script(config, output_dir)
        # Tabular / forecasting (generic predict/forecast, NOT trading)
        if any(k in niche for k in ("forecast", "time series", "time-series", "tabular", "csv data")):
            return self._tabular_script(config, output_dir)
        # Audio / speech
        if any(k in niche for k in ("audio", "sound", "music", "speech", "voice", "transcri", "whisper")):
            return self._audio_script(config, output_dir)
        # Translation
        if any(k in niche for k in ("translat", "language pair", "multilingual")):
            return self._translation_script(config, output_dir)
        # Summarisation
        if any(k in niche for k in ("summar", "abstract", "digest", "brief")):
            return self._summarizer_script(config, output_dir)
        # Vision (keep "detect" out to avoid conflict with anomaly)
        if any(k in niche for k in ("image", "photo", "picture", "visual", "vision",
                                     "recogni", "classif image", "classif photo", "object detect")):
            return self._vision_script(config, output_dir)
        # Text classification
        if any(k in niche for k in ("classif", "label", "categor", "spam", "intent", "topic")):
            return self._classifier_script(config, output_dir)
        # Code AI
        if any(k in niche for k in ("code", "programm", "developer", "software", "debug", "review")):
            return self._code_ai_script(config, output_dir)
        # Default: LLM chatbot / general assistant
        return self._chatbot_script(config, output_dir)

