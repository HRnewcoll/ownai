"""Central configuration for OwnAI."""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths
MODELS_DIR = os.path.join(BASE_DIR, "models_dir")
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
DATABASE_PATH = os.path.join(BASE_DIR, "ownai.db")
DATA_DIR = os.path.join(BASE_DIR, "data")
BENCHMARK_RESULTS_DIR = os.path.join(BASE_DIR, "data", "benchmarks")
MEMORY_DB_PATH = os.path.join(BASE_DIR, "data", "memory.db")

# Ensure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(BENCHMARK_RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Category definitions – used for UI grouping/filtering
# ---------------------------------------------------------------------------

AI_CATEGORIES = {
    "text": {
        "label": "Text & Language",
        "icon": "📝",
        "description": "Natural language: chat, code, translation, summarisation, classification",
    },
    "vision": {
        "label": "Vision & Images",
        "icon": "👁️",
        "description": "Images and video: generation, recognition, detection, documents",
    },
    "audio": {
        "label": "Audio & Speech",
        "icon": "🎵",
        "description": "Sound: speech-to-text, audio classification, music generation",
    },
    "data": {
        "label": "Data & Analytics",
        "icon": "📊",
        "description": "Structured data: trading, forecasting, anomaly detection, recommendations",
    },
    "domain": {
        "label": "Domain Specific",
        "icon": "🔬",
        "description": "Specialist fields: medical, legal, agriculture, education, cybersecurity",
    },
    "custom": {
        "label": "Custom / Any Niche",
        "icon": "⚙️",
        "description": "Describe any AI you want to build – the platform adapts to your niche",
    },
}

# ---------------------------------------------------------------------------
# AI type registry
# Every type has: label, icon, description, color, capabilities, category,
#                example_use_cases (list of short strings shown in UI)
# ---------------------------------------------------------------------------

AI_TYPES = {
    # ── Text & Language ──────────────────────────────────────────────────────
    "chatbot": {
        "label": "Chatbot / Conversational AI",
        "icon": "💬",
        "description": "Build a conversational AI that can answer questions, hold discussions, and assist users.",
        "color": "#6366f1",
        "category": "text",
        "capabilities": ["text", "multimodal", "voice", "voice_out", "reasoning", "deep_thinking",
                         "tool_use", "mcp", "web_search", "pc_control", "memory", "agentic_loop",
                         "multi_agent", "sentiment", "graph_rag"],
        "example_use_cases": ["Customer service bot", "Personal assistant", "FAQ bot", "Support agent"],
    },
    "code_ai": {
        "label": "Code Generation / Review AI",
        "icon": "💻",
        "description": "AI that writes, completes, explains, and reviews code across any programming language.",
        "color": "#0ea5e9",
        "category": "text",
        "capabilities": ["text", "reasoning", "deep_thinking", "tool_use", "agentic_loop",
                         "tdd", "multi_agent", "memory", "graph_rag", "benchmark"],
        "example_use_cases": ["Copilot-style autocomplete", "Bug finder", "Code explainer", "Refactoring tool"],
    },
    "translator": {
        "label": "Language Translator",
        "icon": "🌍",
        "description": "AI that translates text between languages accurately, with domain-specific vocabulary.",
        "color": "#14b8a6",
        "category": "text",
        "capabilities": ["text"],
        "example_use_cases": ["Document translator", "Real-time chat translator", "Legal/medical translator"],
    },
    "summarizer": {
        "label": "Text Summariser",
        "icon": "📋",
        "description": "Condense long documents, articles, meetings, or videos into concise summaries.",
        "color": "#a78bfa",
        "category": "text",
        "capabilities": ["text", "multimodal"],
        "example_use_cases": ["News summariser", "Meeting notes", "Research paper digest", "Email briefer"],
    },
    "text_classifier": {
        "label": "Text Classifier",
        "icon": "🏷️",
        "description": "Categorise text into predefined classes: spam, topics, intent, emotion, and more.",
        "color": "#f59e0b",
        "category": "text",
        "capabilities": ["text"],
        "example_use_cases": ["Spam filter", "Topic tagger", "Intent detector", "Emotion analysis"],
    },
    "creative_writer": {
        "label": "Creative Writing AI",
        "icon": "✍️",
        "description": "AI for stories, poetry, scripts, marketing copy, and any creative text generation.",
        "color": "#f43f5e",
        "category": "text",
        "capabilities": ["text", "reasoning"],
        "example_use_cases": ["Story generator", "Blog writer", "Ad copywriter", "Screenplay assistant"],
    },

    # ── Vision & Images ──────────────────────────────────────────────────────
    "image_generator": {
        "label": "Image Generator",
        "icon": "🎨",
        "description": "Create AI that generates images from text prompts using diffusion models.",
        "color": "#ec4899",
        "category": "vision",
        "capabilities": ["text", "multimodal"],
        "example_use_cases": ["Product images", "Art creation", "Concept art", "Logo design"],
    },
    "image_recognition": {
        "label": "Image Recognition / Vision AI",
        "icon": "🔍",
        "description": "Build AI that identifies objects, scenes, faces and patterns in images.",
        "color": "#f59e0b",
        "category": "vision",
        "capabilities": ["multimodal", "tool_use"],
        "example_use_cases": ["Product classifier", "Scene tagger", "Quality control", "Wildlife ID"],
    },
    "object_detector": {
        "label": "Object Detection AI",
        "icon": "📦",
        "description": "Detect and locate multiple objects in images or video streams with bounding boxes.",
        "color": "#f97316",
        "category": "vision",
        "capabilities": ["multimodal", "tool_use"],
        "example_use_cases": ["Security camera AI", "Vehicle counter", "Defect detection", "Retail shelf AI"],
    },
    "document_ai": {
        "label": "Document AI / OCR",
        "icon": "📄",
        "description": "Extract, classify, and understand text from scanned documents, PDFs, and forms.",
        "color": "#64748b",
        "category": "vision",
        "capabilities": ["multimodal", "text", "tool_use"],
        "example_use_cases": ["Invoice extractor", "Form reader", "ID verification", "Contract parser"],
    },
    "face_ai": {
        "label": "Face Recognition AI",
        "icon": "🧑",
        "description": "Identify, verify, or analyse faces – for attendance, security, or emotion detection.",
        "color": "#dc2626",
        "category": "vision",
        "capabilities": ["multimodal"],
        "example_use_cases": ["Attendance system", "Age/emotion detection", "Access control"],
    },
    "medical_ai": {
        "label": "Medical / Cancer Detection AI",
        "icon": "🏥",
        "description": "Build AI for medical image analysis, anomaly detection, or diagnostic assistance.",
        "color": "#ef4444",
        "category": "domain",
        "capabilities": ["multimodal", "tool_use"],
        "example_use_cases": ["X-ray analyser", "Tumour detector", "Retinal scan AI", "Skin lesion classifier"],
    },
    "space_ai": {
        "label": "Space / Astronomy AI",
        "icon": "🔭",
        "description": "AI for analysing astronomical data, detecting celestial objects, or observing space events.",
        "color": "#8b5cf6",
        "category": "domain",
        "capabilities": ["multimodal", "tool_use", "web_search"],
        "example_use_cases": ["Exoplanet finder", "Galaxy classifier", "Meteor detection", "Telescope data analyser"],
    },

    # ── Audio & Speech ────────────────────────────────────────────────────────
    "speech_to_text": {
        "label": "Speech-to-Text / Transcription",
        "icon": "🎙️",
        "description": "Transcribe audio and speech to text with high accuracy, including custom vocabularies.",
        "color": "#0284c7",
        "category": "audio",
        "capabilities": ["voice", "text"],
        "example_use_cases": ["Meeting transcriber", "Subtitle generator", "Podcast notes", "Voice search"],
    },
    "audio_classifier": {
        "label": "Audio / Sound Classifier",
        "icon": "🔊",
        "description": "Classify audio clips: environmental sounds, music genres, animal calls, machinery noise.",
        "color": "#7c3aed",
        "category": "audio",
        "capabilities": ["voice"],
        "example_use_cases": ["Gunshot detector", "Machine fault listener", "Bird identifier", "Music genre tagger"],
    },
    "music_ai": {
        "label": "Music Generation AI",
        "icon": "🎵",
        "description": "Generate melodies, harmonies, or full music tracks conditioned on style or mood.",
        "color": "#c026d3",
        "category": "audio",
        "capabilities": ["voice", "text"],
        "example_use_cases": ["Background music creator", "Jingle generator", "Game soundtrack AI"],
    },

    # ── Data & Analytics ──────────────────────────────────────────────────────
    "trading_bot": {
        "label": "Trading Bot (Crypto / Forex)",
        "icon": "📈",
        "description": "Develop AI-powered trading bots for crypto or forex markets with strategy automation.",
        "color": "#10b981",
        "category": "data",
        "capabilities": ["text", "tool_use", "web_search", "pc_control"],
        "example_use_cases": ["BTC momentum trader", "Forex scalper", "Options strategy bot", "DeFi yield optimiser"],
    },
    "news_sentiment": {
        "label": "News Sentiment Analyser",
        "icon": "📰",
        "description": "Create AI that analyses news articles and social media for market or topic sentiment.",
        "color": "#3b82f6",
        "category": "data",
        "capabilities": ["text", "web_search", "tool_use"],
        "example_use_cases": ["Market mood tracker", "Brand monitor", "Political sentiment", "Product review analyser"],
    },
    "anomaly_detector": {
        "label": "Anomaly Detection AI",
        "icon": "🚨",
        "description": "Find unusual patterns in time-series, logs, transactions, or sensor data.",
        "color": "#f59e0b",
        "category": "data",
        "capabilities": ["text", "tool_use"],
        "example_use_cases": ["Fraud detector", "Server log monitor", "Manufacturing QA", "IoT sensor alert"],
    },
    "data_analyst": {
        "label": "Data Analyst / Forecasting AI",
        "icon": "📊",
        "description": "Analyse tabular data, predict future values, and surface insights from structured datasets.",
        "color": "#059669",
        "category": "data",
        "capabilities": ["text", "tool_use", "reasoning"],
        "example_use_cases": ["Sales forecaster", "Demand predictor", "Churn predictor", "Energy forecast"],
    },
    "recommendation": {
        "label": "Recommendation AI",
        "icon": "⭐",
        "description": "Personalised recommendations for products, content, music, or any item catalogue.",
        "color": "#f59e0b",
        "category": "data",
        "capabilities": ["text", "tool_use"],
        "example_use_cases": ["E-commerce recommender", "Content personaliser", "Movie suggester", "Playlist AI"],
    },

    # ── Domain Specific ───────────────────────────────────────────────────────
    "legal_ai": {
        "label": "Legal AI",
        "icon": "⚖️",
        "description": "AI for legal document analysis, contract review, case research, and compliance checking.",
        "color": "#7c3aed",
        "category": "domain",
        "capabilities": ["text", "reasoning", "tool_use"],
        "example_use_cases": ["Contract reviewer", "Case law researcher", "GDPR compliance checker", "NDA summariser"],
    },
    "education_ai": {
        "label": "Education / Tutoring AI",
        "icon": "🎓",
        "description": "Personalised tutors, quiz generators, course explainers, and student feedback systems.",
        "color": "#2563eb",
        "category": "domain",
        "capabilities": ["text", "voice", "multimodal", "reasoning"],
        "example_use_cases": ["Maths tutor", "Language coach", "Quiz creator", "Essay feedback bot"],
    },
    "customer_support": {
        "label": "Customer Support AI",
        "icon": "🎧",
        "description": "Handle customer queries, ticket routing, escalation, and knowledge base lookup.",
        "color": "#0891b2",
        "category": "domain",
        "capabilities": ["text", "voice", "tool_use", "web_search"],
        "example_use_cases": ["Tier-1 support agent", "FAQ responder", "Ticket classifier", "Returns handler"],
    },
    "agriculture_ai": {
        "label": "Agriculture / Crop AI",
        "icon": "🌱",
        "description": "Detect plant diseases, predict yields, optimise irrigation, and monitor crop health.",
        "color": "#65a30d",
        "category": "domain",
        "capabilities": ["multimodal", "tool_use", "reasoning"],
        "example_use_cases": ["Disease detector", "Yield predictor", "Weed identifier", "Drone imagery analyser"],
    },
    "game_ai": {
        "label": "Game AI / NPC Agent",
        "icon": "🎮",
        "description": "Intelligent game characters, RL-trained agents, procedural generation, or game testing.",
        "color": "#dc2626",
        "category": "domain",
        "capabilities": ["reasoning", "tool_use", "pc_control"],
        "example_use_cases": ["RL game agent", "NPC dialogue", "Procedural level design", "Game testing bot"],
    },
    "cyber_ai": {
        "label": "Cybersecurity AI",
        "icon": "🛡️",
        "description": "Detect threats, classify malware, analyse logs, and identify vulnerabilities.",
        "color": "#374151",
        "category": "domain",
        "capabilities": ["text", "tool_use", "reasoning", "web_search"],
        "example_use_cases": ["Malware classifier", "Intrusion detector", "Phishing filter", "SIEM log analyser"],
    },

    # ── Custom / Any Niche ────────────────────────────────────────────────────
    "custom": {
        "label": "Custom / My Own Niche",
        "icon": "⚙️",
        "description": "Describe any AI you want to build. The platform configures everything for your specific niche.",
        "color": "#6b7280",
        "category": "custom",
        "capabilities": ["text", "multimodal", "voice", "reasoning", "tool_use", "mcp", "web_search", "pc_control"],
        "example_use_cases": ["Anything you can imagine", "Describe it in your own words"],
    },
}

# ---------------------------------------------------------------------------
# Capability metadata
# ---------------------------------------------------------------------------

CAPABILITIES = {
    "text": {"label": "Text Processing", "icon": "📝", "description": "Core natural language understanding and generation"},
    "multimodal": {"label": "Multimodal (Images + Text)", "icon": "🖼️", "description": "Process and generate both text and images"},
    "voice": {"label": "Voice / Speech Input", "icon": "🎙️", "description": "Accept spoken input and convert to text via Whisper"},
    "voice_out": {"label": "Voice Output (TTS)", "icon": "🔊", "description": "Generate spoken audio responses via Piper/Coqui TTS"},
    "image_out": {"label": "Image Output", "icon": "🎨", "description": "Generate images as part of responses"},
    "reasoning": {"label": "Chain-of-Thought Reasoning", "icon": "🧠", "description": "Step-by-step reasoning for complex problems"},
    "deep_thinking": {"label": "Deep Thinking / Scratchpad", "icon": "💭", "description": "Extended hidden reasoning before generating answers"},
    "tool_use": {"label": "Tool Use / Function Calling", "icon": "🔧", "description": "Call external APIs and use tools dynamically"},
    "mcp": {"label": "MCP Protocol Support", "icon": "🔌", "description": "Model Context Protocol for extended tool ecosystems"},
    "web_search": {"label": "Web Search", "icon": "🌐", "description": "Search the internet in real-time for up-to-date info"},
    "pc_control": {"label": "PC / Computer Control", "icon": "🖥️", "description": "Agentic control of desktop: files, apps, browser (like OpenClaw/CoPaw)"},
    "memory": {"label": "Long-Term Memory", "icon": "🗄️", "description": "Persist facts and past conversations across sessions"},
    "agentic_loop": {"label": "Agentic Think-Act-Observe Loop", "icon": "🔄", "description": "Think → Act → Observe → Verify loop for robust problem solving"},
    "tdd": {"label": "Test-Driven Development (TDD)", "icon": "✅", "description": "Write failing tests first, then implement – boosts benchmark scores"},
    "multi_agent": {"label": "Multi-Agent Swarm", "icon": "🤝", "description": "Architect + Coder + Reviewer agents collaborate to solve tasks"},
    "moe": {"label": "Mixture of Experts (MoE)", "icon": "⚡", "description": "Sparse expert routing for fast, efficient inference"},
    "autonomous_training": {"label": "Autonomous Self-Training", "icon": "🚀", "description": "Scrapes quality data and fine-tunes on a schedule"},
    "graph_rag": {"label": "Graph-RAG (Code Intelligence)", "icon": "🕸️", "description": "Maps code dependencies for precise retrieval in large codebases"},
    "sentiment": {"label": "Emotional Intelligence / Empathy", "icon": "❤️", "description": "Detects tone and responds with appropriate empathy"},
    "benchmark": {"label": "Built-in Benchmarking", "icon": "📊", "description": "HumanEval + reasoning benchmarks to track improvement over time"},
}

# ---------------------------------------------------------------------------
# Base model options per AI type
# ---------------------------------------------------------------------------

_LLM_GENERAL = [
    # SOTA / Frontier open models (2024-2025)
    {"id": "Qwen/Qwen2.5-72B-Instruct", "label": "Qwen 2.5 72B Instruct – SOTA reasoning & coding", "size": "72B"},
    {"id": "Qwen/Qwen2.5-7B-Instruct", "label": "Qwen 2.5 7B Instruct – Fast & capable", "size": "7B"},
    {"id": "deepseek-ai/DeepSeek-V3", "label": "DeepSeek V3 – 671B MoE, top-tier", "size": "671B"},
    {"id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "label": "DeepSeek R1 Distill 7B – Reasoning model", "size": "7B"},
    {"id": "meta-llama/Meta-Llama-3.1-70B-Instruct", "label": "Llama 3.1 70B Instruct – Strong general purpose", "size": "70B"},
    {"id": "meta-llama/Meta-Llama-3.1-8B-Instruct", "label": "Llama 3.1 8B Instruct – Fast & efficient", "size": "8B"},
    {"id": "mistralai/Mixtral-8x7B-Instruct-v0.1", "label": "Mixtral 8x7B MoE – Sparse expert routing", "size": "47B"},
    {"id": "mistralai/Mixtral-8x22B-Instruct-v0.1", "label": "Mixtral 8x22B MoE – High-capacity MoE", "size": "141B"},
    {"id": "mistralai/Mistral-7B-Instruct-v0.3", "label": "Mistral 7B Instruct v0.3 – Balanced", "size": "7B"},
    {"id": "microsoft/Phi-3.5-mini-instruct", "label": "Phi-3.5 Mini – Microsoft compact model", "size": "3.8B"},
    {"id": "microsoft/phi-2", "label": "Phi-2 (2.7B) – Fast & efficient", "size": "2.7B"},
    {"id": "google/gemma-2-9b-it", "label": "Gemma 2 9B Instruct – Google latest", "size": "9B"},
    {"id": "meta-llama/Meta-Llama-3-8B", "label": "Llama 3 8B – Latest Llama base", "size": "8B"},
    {"id": "tiiuae/falcon-7b", "label": "Falcon 7B – TII model", "size": "7B"},
    {"id": "scratch", "label": "Train from scratch (custom architecture)", "size": "custom"},
]

_VISION_MODELS = [
    {"id": "google/vit-base-patch16-224", "label": "ViT Base – Vision Transformer", "size": "86M"},
    {"id": "microsoft/resnet-50", "label": "ResNet-50 – Classic CNN", "size": "25M"},
    {"id": "openai/clip-vit-base-patch32", "label": "CLIP – Zero-shot vision-language", "size": "151M"},
    {"id": "facebook/detr-resnet-50", "label": "DETR – Transformer object detection", "size": "41M"},
    {"id": "scratch", "label": "Train from scratch", "size": "custom"},
]

BASE_MODELS = {
    # Text & Language
    "chatbot": _LLM_GENERAL,
    "code_ai": [
        # SOTA coding models
        {"id": "Qwen/Qwen2.5-Coder-32B-Instruct", "label": "Qwen2.5 Coder 32B – #1 open-source code model", "size": "32B"},
        {"id": "Qwen/Qwen2.5-Coder-7B-Instruct", "label": "Qwen2.5 Coder 7B – Fast code model", "size": "7B"},
        {"id": "deepseek-ai/DeepSeek-Coder-V2-Instruct", "label": "DeepSeek Coder V2 – 236B MoE coder", "size": "236B"},
        {"id": "deepseek-ai/deepseek-coder-6.7b-instruct", "label": "DeepSeek Coder 6.7B – Efficient", "size": "6.7B"},
        {"id": "bigcode/starcoder2-15b", "label": "StarCoder 2 15B – State-of-the-art code LM", "size": "15B"},
        {"id": "bigcode/starcoder2-3b", "label": "StarCoder 2 3B – Lightweight code LM", "size": "3B"},
        {"id": "codellama/CodeLlama-34b-Instruct-hf", "label": "Code Llama 34B – Meta code model", "size": "34B"},
        {"id": "codellama/CodeLlama-7b-hf", "label": "Code Llama 7B – Compact", "size": "7B"},
        {"id": "microsoft/phi-2", "label": "Phi-2 (2.7B) – Great at code", "size": "2.7B"},
        {"id": "scratch", "label": "Train from scratch", "size": "custom"},
    ],
    "translator": [
        {"id": "Helsinki-NLP/opus-mt-en-de", "label": "OPUS MT EN→DE – Helsinki NLP", "size": "300M"},
        {"id": "Helsinki-NLP/opus-mt-en-fr", "label": "OPUS MT EN→FR – Helsinki NLP", "size": "300M"},
        {"id": "facebook/nllb-200-distilled-600M", "label": "NLLB-200 600M – Meta 200 languages", "size": "600M"},
        {"id": "google/mt5-base", "label": "mT5 Base – Multilingual T5", "size": "582M"},
        {"id": "scratch", "label": "Train from scratch", "size": "custom"},
    ],
    "summarizer": [
        {"id": "facebook/bart-large-cnn", "label": "BART Large CNN – News summarisation", "size": "400M"},
        {"id": "google/pegasus-xsum", "label": "PEGASUS XSum – Abstractive summary", "size": "568M"},
        {"id": "sshleifer/distilbart-cnn-12-6", "label": "DistilBART – Lightweight summary", "size": "306M"},
        {"id": "microsoft/phi-2", "label": "Phi-2 – LLM summarisation", "size": "2.7B"},
        {"id": "scratch", "label": "Train from scratch", "size": "custom"},
    ],
    "text_classifier": [
        {"id": "distilbert-base-uncased", "label": "DistilBERT – Fast classifier", "size": "66M"},
        {"id": "bert-base-uncased", "label": "BERT Base – Classic NLP", "size": "110M"},
        {"id": "roberta-base", "label": "RoBERTa Base – Robust BERT", "size": "125M"},
        {"id": "albert-base-v2", "label": "ALBERT Base – Compact BERT", "size": "12M"},
        {"id": "scratch", "label": "Train from scratch", "size": "custom"},
    ],
    "creative_writer": [
        {"id": "mistralai/Mistral-7B-v0.1", "label": "Mistral 7B – Excellent creative writing", "size": "7B"},
        {"id": "microsoft/phi-2", "label": "Phi-2 (2.7B) – Creative & fast", "size": "2.7B"},
        {"id": "EleutherAI/gpt-neo-1.3B", "label": "GPT-Neo 1.3B – Open GPT-style", "size": "1.3B"},
        {"id": "scratch", "label": "Train from scratch", "size": "custom"},
    ],

    # Vision & Images
    "image_generator": [
        {"id": "runwayml/stable-diffusion-v1-5", "label": "Stable Diffusion 1.5", "size": "860M"},
        {"id": "stabilityai/stable-diffusion-2-1", "label": "Stable Diffusion 2.1", "size": "900M"},
        {"id": "stabilityai/stable-diffusion-xl-base-1.0", "label": "SDXL Base 1.0", "size": "6.9B"},
        {"id": "scratch", "label": "Train from scratch", "size": "custom"},
    ],
    "image_recognition": _VISION_MODELS,
    "object_detector": [
        {"id": "facebook/detr-resnet-50", "label": "DETR ResNet-50 – Transformer detection", "size": "41M"},
        {"id": "hustvl/yolos-tiny", "label": "YOLOS Tiny – Tiny transformer", "size": "6.5M"},
        {"id": "microsoft/resnet-50", "label": "ResNet-50 – Backbone for detection", "size": "25M"},
        {"id": "scratch", "label": "Train from scratch (YOLO-style)", "size": "custom"},
    ],
    "document_ai": [
        {"id": "microsoft/layoutlmv3-base", "label": "LayoutLMv3 – Document understanding", "size": "133M"},
        {"id": "naver-clova-ix/donut-base", "label": "Donut – OCR-free document AI", "size": "200M"},
        {"id": "microsoft/trocr-base-printed", "label": "TrOCR – Printed text OCR", "size": "334M"},
        {"id": "scratch", "label": "Train from scratch", "size": "custom"},
    ],
    "face_ai": [
        {"id": "google/vit-base-patch16-224", "label": "ViT Base – Fine-tune on face data", "size": "86M"},
        {"id": "microsoft/resnet-50", "label": "ResNet-50 – Classic face backbone", "size": "25M"},
        {"id": "scratch", "label": "Train from scratch", "size": "custom"},
    ],

    # Audio & Speech
    "speech_to_text": [
        {"id": "openai/whisper-small", "label": "Whisper Small – Fast transcription", "size": "244M"},
        {"id": "openai/whisper-medium", "label": "Whisper Medium – Balanced accuracy", "size": "769M"},
        {"id": "openai/whisper-large-v3", "label": "Whisper Large v3 – Best accuracy", "size": "1.5B"},
        {"id": "facebook/wav2vec2-base-960h", "label": "Wav2Vec2 Base – English ASR", "size": "95M"},
        {"id": "scratch", "label": "Train from scratch", "size": "custom"},
    ],
    "audio_classifier": [
        {"id": "facebook/wav2vec2-base", "label": "Wav2Vec2 Base – Audio features", "size": "95M"},
        {"id": "MIT/ast-finetuned-audioset-10-10-0.4593", "label": "AST – Audio Spectrogram Transformer", "size": "88M"},
        {"id": "scratch", "label": "Train from scratch", "size": "custom"},
    ],
    "music_ai": [
        {"id": "facebook/musicgen-small", "label": "MusicGen Small – Meta music gen", "size": "300M"},
        {"id": "facebook/musicgen-medium", "label": "MusicGen Medium – Better quality", "size": "1.5B"},
        {"id": "scratch", "label": "Train from scratch", "size": "custom"},
    ],

    # Data & Analytics
    "trading_bot": [
        {"id": "scratch_lstm", "label": "LSTM Network – Sequence prediction", "size": "custom"},
        {"id": "scratch_transformer", "label": "Transformer – Attention-based", "size": "custom"},
        {"id": "scratch_rl", "label": "Reinforcement Learning Agent", "size": "custom"},
    ],
    "news_sentiment": [
        {"id": "distilbert-base-uncased", "label": "DistilBERT – Fast sentiment", "size": "66M"},
        {"id": "ProsusAI/finbert", "label": "FinBERT – Financial sentiment", "size": "110M"},
        {"id": "cardiffnlp/twitter-roberta-base-sentiment", "label": "RoBERTa Sentiment", "size": "125M"},
        {"id": "scratch", "label": "Train from scratch", "size": "custom"},
    ],
    "anomaly_detector": [
        {"id": "scratch_autoencoder", "label": "Autoencoder – Reconstruct & flag anomalies", "size": "custom"},
        {"id": "scratch_isolation_forest", "label": "Isolation Forest – Classic ML anomaly", "size": "custom"},
        {"id": "scratch_transformer", "label": "Transformer Time-Series", "size": "custom"},
    ],
    "data_analyst": [
        {"id": "scratch_xgboost", "label": "XGBoost – Tabular champion", "size": "custom"},
        {"id": "scratch_lstm", "label": "LSTM – Time-series forecasting", "size": "custom"},
        {"id": "scratch_transformer", "label": "Transformer – Tabular/time-series", "size": "custom"},
        {"id": "microsoft/phi-2", "label": "Phi-2 – LLM-powered analysis", "size": "2.7B"},
    ],
    "recommendation": [
        {"id": "scratch_collab_filter", "label": "Collaborative Filtering – Classic CF", "size": "custom"},
        {"id": "scratch_neural_cf", "label": "Neural CF – Deep learning CF", "size": "custom"},
        {"id": "scratch_transformer", "label": "Transformer – Sequential recommendation", "size": "custom"},
    ],

    # Domain Specific
    "medical_ai": [
        {"id": "google/vit-base-patch16-224", "label": "ViT Base – Medical imaging", "size": "86M"},
        {"id": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract", "label": "PubMedBERT – Biomedical text", "size": "110M"},
        {"id": "facebook/detr-resnet-50", "label": "DETR – Medical object detection", "size": "41M"},
        {"id": "scratch", "label": "Train from scratch", "size": "custom"},
    ],
    "space_ai": [
        {"id": "google/vit-base-patch16-224", "label": "ViT Base – Astronomical imaging", "size": "86M"},
        {"id": "openai/clip-vit-base-patch32", "label": "CLIP – Zero-shot classification", "size": "151M"},
        {"id": "scratch", "label": "Train from scratch", "size": "custom"},
    ],
    "legal_ai": [
        {"id": "nlpaueb/legal-bert-base-uncased", "label": "Legal-BERT – Legal domain BERT", "size": "110M"},
        {"id": "mistralai/Mistral-7B-v0.1", "label": "Mistral 7B – LLM for legal reasoning", "size": "7B"},
        {"id": "microsoft/phi-2", "label": "Phi-2 – Compact legal AI", "size": "2.7B"},
        {"id": "scratch", "label": "Train from scratch", "size": "custom"},
    ],
    "education_ai": [
        {"id": "microsoft/phi-2", "label": "Phi-2 – Great for Q&A and tutoring", "size": "2.7B"},
        {"id": "mistralai/Mistral-7B-v0.1", "label": "Mistral 7B – Rich educational responses", "size": "7B"},
        {"id": "distilbert-base-uncased", "label": "DistilBERT – Question answering", "size": "66M"},
        {"id": "scratch", "label": "Train from scratch", "size": "custom"},
    ],
    "customer_support": [
        {"id": "microsoft/phi-2", "label": "Phi-2 (2.7B) – Fast support responses", "size": "2.7B"},
        {"id": "mistralai/Mistral-7B-v0.1", "label": "Mistral 7B – Quality support bot", "size": "7B"},
        {"id": "distilbert-base-uncased", "label": "DistilBERT – Intent classification", "size": "66M"},
        {"id": "scratch", "label": "Train from scratch", "size": "custom"},
    ],
    "agriculture_ai": [
        {"id": "google/vit-base-patch16-224", "label": "ViT Base – Crop/leaf classifier", "size": "86M"},
        {"id": "microsoft/resnet-50", "label": "ResNet-50 – Disease detection", "size": "25M"},
        {"id": "facebook/detr-resnet-50", "label": "DETR – Pest/weed detection", "size": "41M"},
        {"id": "scratch", "label": "Train from scratch", "size": "custom"},
    ],
    "game_ai": [
        {"id": "scratch_ppo", "label": "PPO – Proximal Policy Optimisation (RL)", "size": "custom"},
        {"id": "scratch_dqn", "label": "DQN – Deep Q-Network", "size": "custom"},
        {"id": "scratch_a3c", "label": "A3C – Async Advantage Actor-Critic", "size": "custom"},
        {"id": "microsoft/phi-2", "label": "Phi-2 – LLM-based NPC dialogue", "size": "2.7B"},
    ],
    "cyber_ai": [
        {"id": "distilbert-base-uncased", "label": "DistilBERT – Log / text classifier", "size": "66M"},
        {"id": "microsoft/phi-2", "label": "Phi-2 – Threat analysis assistant", "size": "2.7B"},
        {"id": "scratch_autoencoder", "label": "Autoencoder – Anomaly detection", "size": "custom"},
        {"id": "scratch", "label": "Train from scratch", "size": "custom"},
    ],

    # Custom
    "custom": _LLM_GENERAL,
}
