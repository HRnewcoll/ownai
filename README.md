<div align="center">

# 🧠 OwnAI

**The AI Development Platform for Everyone**

> Build, train, and run your own AI with a few clicks. No PhD required.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-000000?logo=flask)](https://flask.palletsprojects.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-FFD21E)](https://huggingface.co)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)](docker-compose.yml)

OwnAI is a self-hosted, all-in-one platform that lets you design, train, and run custom AI models through a beautiful web UI — from chatbots to image generators, trading bots, medical detection AI, game agents, and anything in between.

</div>

---

## ✨ What you can build

**28 AI types across 6 categories:**

| 📝 Text & Language | 👁️ Vision & Images | 🎵 Audio & Speech |
|---|---|---|
| 💬 Chatbot / Conversational AI | 🎨 Image Generator | 🎙️ Speech-to-Text |
| 💻 Code Generation & Review | 🔍 Image Recognition | 🔊 Audio Classifier |
| 🌍 Language Translator | 📦 Object Detector | 🎵 Music Generation |
| 📋 Text Summariser | 📄 Document AI / OCR | |
| 🏷️ Text Classifier | 🧑 Face Recognition | |
| ✍️ Creative Writer | | |

| 📊 Data & Analytics | 🔬 Domain Specific | ⚙️ Custom |
|---|---|---|
| 📈 Trading Bot (Crypto/Forex) | 🏥 Medical / Cancer Detection | ⚙️ Any niche you describe |
| 📰 News Sentiment | 🔭 Space / Astronomy | |
| 🚨 Anomaly Detector | ⚖️ Legal AI | |
| 📊 Data Analyst / Forecasting | 🎓 Education / Tutoring | |
| ⭐ Recommendation AI | 🎧 Customer Support | |
| | 🌱 Agriculture / Crop AI | |
| | 🎮 Game AI / NPC Agent | |
| | 🛡️ Cybersecurity AI | |

---

## 🚀 Quick Start

### Option 1 – One-click setup (recommended)

```bash
git clone https://github.com/HRnewcoll/ownai.git && cd ownai

# Linux / macOS
bash setup.sh

# Windows — double-click setup.bat
```

Then open **http://localhost:5000** in your browser.

### Option 2 – Manual install

```bash
git clone https://github.com/HRnewcoll/ownai.git
cd ownai
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

### Option 3 – Docker

```bash
docker compose up -d
# App runs at http://localhost:5000
# Models saved to ./models_dir  ·  Datasets saved to ./uploads
```

---

## 🏗️ How it works

```
1. Build  → Choose AI type, base model, capabilities & training settings
2. Train  → OwnAI generates a training script and runs it with live log streaming
3. Run    → Chat, upload images/audio, or feed data to your trained AI
4. Export → Download your model weights as a zip for use anywhere
```

---

## 📸 Screenshots

| Dashboard | Builder Wizard | Training |
|---|---|---|
| 28 AI types grouped by category | Searchable grid with category filters | Live log streaming + epoch progress |

---

## ⚙️ Features

### 🔨 Builder Wizard
- Step-by-step wizard: type → base model → capabilities → training config → review
- **Live search** across all 28 types; **category filter tabs**
- **Custom niche field**: describe any AI in plain English — OwnAI auto-selects the right training template
- Pre-configured base models for every type (Phi-2, Mistral, Llama 3, StarCoder, ViT, Whisper, DETR…)

### ⚡ Training
- Auto-generated training scripts tailored to your AI type and config
- Fine-tune methods: **LoRA**, **QLoRA (4-bit)**, full fine-tuning
- Live log streaming via **WebSocket**
- Epoch/step progress bar parsed from training output
- View generated training script, stop training at any time
- Export trained model as a zip with one click

### 📂 Datasets
- **File upload**: `.txt`, `.csv`, `.jsonl`, `.json`, images, PDFs
- **HuggingFace Hub**: search and link from 100,000+ public datasets
- **Web scraping**: fetch and extract text from URLs (SSRF-protected)

### 💬 Run / Inference
- **Text/Language models**: full chat interface with markdown, voice input
- **Vision models**: image drag-and-drop + analysis panel
- **Audio models**: audio file upload + player + transcription
- **Data/Trading models**: structured input form with type-specific fields
- **Agentic models**: web search, file access, shell commands, browser control

### 🤖 Agentic Mode
- Web search (DuckDuckGo)
- File system access (read/write/list)
- Shell command execution (sandboxed blocklist)
- Browser control
- MCP Protocol support

---

## 📁 Project Structure

```
ownai/
├── app.py                  # Flask app + all API routes
├── requirements.txt        # Python dependencies
├── setup.sh / setup.bat    # One-click installers
├── Dockerfile              # Container build
├── docker-compose.yml      # Full stack deployment
├── .env.example            # Config template
├── core/
│   ├── config.py           # 28 AI types, capabilities, base models registry
│   ├── model_builder.py    # Training script generation (15 templates)
│   ├── trainer.py          # Background training + live log streaming
│   ├── dataset_manager.py  # Upload, HF Hub, web scraping (SSRF-safe)
│   └── agent.py            # Agent tools (web search, PC control, etc.)
├── templates/
│   ├── base.html           # Sidebar, topbar, theme, system status, onboarding
│   ├── index.html          # Dashboard
│   ├── builder.html        # AI Builder wizard
│   ├── train.html          # Training interface + progress
│   ├── models.html         # Model management
│   ├── run.html            # Type-aware run UI (chat / vision / audio / data)
│   ├── datasets.html       # Dataset management
│   └── help.html           # Full documentation + ecosystem resources
├── static/
│   ├── css/style.css       # Dark/light theme + all component styles
│   └── js/app.js           # Frontend JS (system status, onboarding, theme)
├── models_dir/             # Trained model checkpoints (auto-created)
└── uploads/                # Dataset files (auto-created)
```

---

## ⚙️ Configuration

Copy `.env.example` to `.env` and edit as needed:

| Variable | Default | Description |
|---|---|---|
| `SECRET_KEY` | random | Flask session key — set a fixed value in production |
| `PORT` | `5000` | HTTP port |
| `HUGGINGFACE_HUB_TOKEN` | — | Token for gated models (Llama, Gemma) |
| `CUDA_VISIBLE_DEVICES` | `0` | GPU index; `-1` for CPU only |

---

## 📋 Requirements

- **Python 3.10+**
- **8 GB RAM** minimum (16 GB+ for 7B models)
- **NVIDIA GPU + CUDA** optional but strongly recommended
- **~10 GB disk** for base model downloads

The sidebar shows a live **GPU status indicator** — green = CUDA GPU detected, amber = CPU only.

---

## 🌐 Ecosystem

OwnAI is built on and inspired by the open-source AI ecosystem:

**Foundations:** PyTorch · 🤗 Transformers · 🤗 Datasets · scikit-learn · XGBoost  
**Training:** LoRA/QLoRA via PEFT · TRL · Accelerate · DeepSpeed  
**Vision:** YOLO (Ultralytics) · DETR · CLIP · ViT  
**Audio:** Whisper · Wav2Vec2 · MusicGen  
**Agents:** DuckDuckGo Search · MCP Protocol  
**Deployment:** ONNX · Docker · Flask-SocketIO

See the **[Help → Ecosystem & Resources](http://localhost:5000/help#ecosystem)** page for 50+ curated community repos.

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch: `git checkout -b feat/my-feature`
3. Make your changes with tests
4. Open a pull request

---

## 📄 License

MIT — use freely, build boldly.
