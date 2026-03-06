"""Central configuration for OwnAI."""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths
MODELS_DIR = os.path.join(BASE_DIR, "models_dir")
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
DATABASE_PATH = os.path.join(BASE_DIR, "ownai.db")

# Ensure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

# AI types with display metadata
AI_TYPES = {
    "chatbot": {
        "label": "Chatbot / Conversational AI",
        "icon": "💬",
        "description": "Build a conversational AI that can answer questions, hold discussions, and assist users.",
        "color": "#6366f1",
        "capabilities": ["text", "multimodal", "voice", "reasoning", "tool_use", "mcp", "web_search", "pc_control"],
    },
    "image_generator": {
        "label": "Image Generator",
        "icon": "🎨",
        "description": "Create AI that generates images from text prompts using diffusion models.",
        "color": "#ec4899",
        "capabilities": ["text", "multimodal"],
    },
    "image_recognition": {
        "label": "Image Recognition / Vision AI",
        "icon": "👁️",
        "description": "Build AI that identifies objects, scenes, faces and patterns in images.",
        "color": "#f59e0b",
        "capabilities": ["multimodal", "tool_use"],
    },
    "trading_bot": {
        "label": "Trading Bot (Crypto / Forex)",
        "icon": "📈",
        "description": "Develop AI-powered trading bots for crypto or forex markets with strategy automation.",
        "color": "#10b981",
        "capabilities": ["text", "tool_use", "web_search", "pc_control"],
    },
    "news_sentiment": {
        "label": "News Sentiment Analyser",
        "icon": "📰",
        "description": "Create AI that analyses news articles and social media for market or topic sentiment.",
        "color": "#3b82f6",
        "capabilities": ["text", "web_search", "tool_use"],
    },
    "medical_ai": {
        "label": "Medical / Cancer Detection AI",
        "icon": "🏥",
        "description": "Build AI for medical image analysis, anomaly detection, or diagnostic assistance.",
        "color": "#ef4444",
        "capabilities": ["multimodal", "tool_use"],
    },
    "space_ai": {
        "label": "Space / Astronomy AI",
        "icon": "🔭",
        "description": "AI for analysing astronomical data, detecting celestial objects, or observing space events.",
        "color": "#8b5cf6",
        "capabilities": ["multimodal", "tool_use", "web_search"],
    },
    "custom": {
        "label": "Custom AI",
        "icon": "⚙️",
        "description": "Build a fully custom AI model with complete control over architecture and capabilities.",
        "color": "#6b7280",
        "capabilities": ["text", "multimodal", "voice", "reasoning", "tool_use", "mcp", "web_search", "pc_control"],
    },
}

# Capability metadata
CAPABILITIES = {
    "text": {"label": "Text Processing", "icon": "📝", "description": "Core natural language understanding and generation"},
    "multimodal": {"label": "Multimodal (Images + Text)", "icon": "🖼️", "description": "Process and generate both text and images"},
    "voice": {"label": "Voice / Speech Input", "icon": "🎙️", "description": "Accept spoken input and convert to text via Whisper"},
    "reasoning": {"label": "Chain-of-Thought Reasoning", "icon": "🧠", "description": "Step-by-step reasoning for complex problems"},
    "tool_use": {"label": "Tool Use / Function Calling", "icon": "🔧", "description": "Call external APIs and use tools dynamically"},
    "mcp": {"label": "MCP Protocol Support", "icon": "🔌", "description": "Model Context Protocol for extended tool ecosystems"},
    "web_search": {"label": "Web Search", "icon": "🌐", "description": "Search the internet in real-time for up-to-date info"},
    "pc_control": {"label": "PC / Computer Control", "icon": "🖥️", "description": "Agentic control of desktop: files, apps, browser (like OpenClaw/CoPaw)"},
}

# Base model options per AI type
BASE_MODELS = {
    "chatbot": [
        {"id": "microsoft/phi-2", "label": "Phi-2 (2.7B) – Fast & efficient", "size": "2.7B"},
        {"id": "mistralai/Mistral-7B-v0.1", "label": "Mistral 7B – Balanced performance", "size": "7B"},
        {"id": "meta-llama/Llama-2-7b-hf", "label": "Llama 2 7B – Strong general purpose", "size": "7B"},
        {"id": "meta-llama/Meta-Llama-3-8B", "label": "Llama 3 8B – Latest Llama", "size": "8B"},
        {"id": "google/gemma-7b", "label": "Gemma 7B – Google's open model", "size": "7B"},
        {"id": "tiiuae/falcon-7b", "label": "Falcon 7B – TII model", "size": "7B"},
        {"id": "scratch", "label": "Train from scratch (custom architecture)", "size": "custom"},
    ],
    "image_generator": [
        {"id": "runwayml/stable-diffusion-v1-5", "label": "Stable Diffusion 1.5", "size": "860M"},
        {"id": "stabilityai/stable-diffusion-2-1", "label": "Stable Diffusion 2.1", "size": "900M"},
        {"id": "stabilityai/stable-diffusion-xl-base-1.0", "label": "SDXL Base 1.0", "size": "6.9B"},
        {"id": "scratch", "label": "Train from scratch", "size": "custom"},
    ],
    "image_recognition": [
        {"id": "google/vit-base-patch16-224", "label": "ViT Base – Vision Transformer", "size": "86M"},
        {"id": "microsoft/resnet-50", "label": "ResNet-50 – Classic CNN", "size": "25M"},
        {"id": "openai/clip-vit-base-patch32", "label": "CLIP – Zero-shot vision", "size": "151M"},
        {"id": "scratch", "label": "Train from scratch", "size": "custom"},
    ],
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
    "medical_ai": [
        {"id": "google/vit-base-patch16-224", "label": "ViT Base – Medical imaging", "size": "86M"},
        {"id": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract", "label": "PubMedBERT – Biomedical text", "size": "110M"},
        {"id": "scratch", "label": "Train from scratch", "size": "custom"},
    ],
    "space_ai": [
        {"id": "google/vit-base-patch16-224", "label": "ViT Base – Astronomical imaging", "size": "86M"},
        {"id": "scratch", "label": "Train from scratch", "size": "custom"},
    ],
    "custom": [
        {"id": "microsoft/phi-2", "label": "Phi-2 (2.7B) – Start from pre-trained", "size": "2.7B"},
        {"id": "mistralai/Mistral-7B-v0.1", "label": "Mistral 7B", "size": "7B"},
        {"id": "scratch", "label": "Train fully from scratch", "size": "custom"},
    ],
}
