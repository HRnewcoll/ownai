# 🧠 OwnAI – The AI Development Platform for Everyone

> Build, train, and run your own AI with a few clicks. No PhD required.

OwnAI is a self-hosted, all-in-one platform that lets you design, train, and deploy custom AI models through a beautiful web UI. From chatbots to image generators, trading bots, medical detection AI, and more — if it can be built with AI, you can build it here.

---

## ✨ Features

| Category | What you can do |
|---|---|
| **AI Types** | Chatbot, Image Generator, Image Recognition, Trading Bot, News Sentiment, Medical/Cancer Detection, Space/Astronomy AI, Custom |
| **Capabilities** | Multimodal (text + images), Voice input (Whisper), Chain-of-thought reasoning, Tool use/function calling, MCP protocol, Web search (DuckDuckGo), PC/computer control (files, commands, browser) |
| **Training** | LoRA, QLoRA (4-bit), full fine-tuning; custom epochs/batch/LR; real-time log streaming via WebSocket |
| **Datasets** | Upload files (.txt, .csv, .jsonl, images), link HuggingFace Hub datasets, scrape web pages |
| **Agentic Mode** | Multi-step autonomous task planning, tool calling, PC control (inspired by OpenClaw & CoPaw) |
| **Content Policy** | Standard (safe) or Uncensored/Abliterated |
| **Models** | Phi-2, Mistral 7B, Llama 2/3, Gemma, Falcon, ViT, CLIP, ResNet, FinBERT, DistilBERT, Stable Diffusion, SDXL, train from scratch |

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/HRnewcoll/ownai.git
cd ownai
pip install -r requirements.txt
```

### 2. Run

```bash
python app.py
```

Then open **http://localhost:5000** in your browser.

---

## 🏗️ How It Works

1. **Build** – Use the step-by-step wizard to choose your AI type, base model, capabilities, and training settings
2. **Train** – OwnAI generates a tailored training script and runs it with real-time log streaming
3. **Run** – Chat with your model, use agent tools (web search, file access, command execution), and more

---

## 📁 Project Structure

```
ownai/
├── app.py                  # Flask application + API routes
├── requirements.txt        # Python dependencies
├── core/
│   ├── config.py           # AI types, capabilities, base model registry
│   ├── model_builder.py    # Training script generation
│   ├── trainer.py          # Background training + log streaming
│   ├── dataset_manager.py  # File uploads, HF Hub, web scraping
│   └── agent.py            # Agent tools (web search, PC control, etc.)
├── templates/              # Jinja2 HTML templates
│   ├── base.html
│   ├── index.html          # Dashboard
│   ├── builder.html        # AI Builder wizard
│   ├── train.html          # Training interface
│   ├── models.html         # Model management
│   ├── run.html            # Chat / inference UI
│   └── datasets.html       # Dataset management
├── static/
│   ├── css/style.css       # Dark/light theme UI
│   └── js/app.js           # Frontend JS
├── models_dir/             # Saved model checkpoints (auto-created)
└── uploads/                # Uploaded dataset files (auto-created)
```

---

## 🤖 Supported AI Types

| Type | Use cases |
|---|---|
| 💬 Chatbot | Customer service, personal assistants, tutors |
| 🎨 Image Generator | Art, product images, concept art |
| 👁️ Image Recognition | Object detection, classification, medical imaging |
| 📈 Trading Bot | Crypto/forex strategy automation |
| 📰 News Sentiment | Market sentiment, brand monitoring |
| 🏥 Medical AI | Cancer detection, diagnostic assistance |
| 🔭 Space AI | Astronomical object detection, spectral analysis |
| ⚙️ Custom | Anything else |

---

## 🔧 Agent Capabilities

When **Agentic Mode** is enabled, your AI can:

- 🌐 **Search the web** (via DuckDuckGo)
- 📁 **Browse directories** on your PC
- 📄 **Read & write files**
- ⚡ **Execute shell commands** (sandboxed with safety blocklist)
- 🌍 **Open URLs** in the system browser

> Inspired by [OpenClaw](https://github.com/openclaw/) and [CoPaw](https://github.com/agentscope-ai/CoPaw)

---

## ⚙️ Configuration

| Environment Variable | Default | Description |
|---|---|---|
| `PORT` | `5000` | Port to run on |
| `SECRET_KEY` | random | Flask secret key |

---

## 📋 Requirements

- Python 3.10+
- NVIDIA GPU recommended for training (CPU works for inference)
- ~8 GB RAM minimum; 16 GB+ recommended for 7B models

---

## 📄 License

MIT – use freely, build boldly.