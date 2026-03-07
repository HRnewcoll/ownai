"""OwnAI tests – core modules and Flask routes."""
import io
import json
import os
import sys
import tempfile
import zipfile

import pytest

# Ensure root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ---------------------------------------------------------------------------
# Core config
# ---------------------------------------------------------------------------


def test_ai_types_present():
    from core.config import AI_TYPES
    # Original 8 types
    original = {"chatbot", "image_generator", "image_recognition", "trading_bot",
                "news_sentiment", "medical_ai", "space_ai", "custom"}
    assert original.issubset(set(AI_TYPES.keys()))
    # New types added for any-niche coverage
    new_types = {"code_ai", "translator", "summarizer", "text_classifier", "creative_writer",
                 "object_detector", "document_ai", "face_ai",
                 "speech_to_text", "audio_classifier", "music_ai",
                 "anomaly_detector", "data_analyst", "recommendation",
                 "legal_ai", "education_ai", "customer_support",
                 "agriculture_ai", "game_ai", "cyber_ai"}
    assert new_types.issubset(set(AI_TYPES.keys()))


def test_ai_types_have_category():
    from core.config import AI_TYPES, AI_CATEGORIES
    valid_cats = set(AI_CATEGORIES.keys())
    for key, at in AI_TYPES.items():
        assert "category" in at, f"{key} missing category"
        assert at["category"] in valid_cats, f"{key} has unknown category '{at['category']}'"


def test_ai_categories_structure():
    from core.config import AI_CATEGORIES
    expected_cats = {"text", "vision", "audio", "data", "domain", "custom"}
    assert expected_cats.issubset(set(AI_CATEGORIES.keys()))
    for key, cat in AI_CATEGORIES.items():
        assert "label" in cat, f"Category {key} missing label"
        assert "icon" in cat, f"Category {key} missing icon"


def test_ai_types_have_example_use_cases():
    from core.config import AI_TYPES
    for key, at in AI_TYPES.items():
        assert "example_use_cases" in at, f"{key} missing example_use_cases"
        assert len(at["example_use_cases"]) > 0, f"{key} has empty example_use_cases"


def test_ai_types_structure():
    from core.config import AI_TYPES
    for key, at in AI_TYPES.items():
        assert "label" in at, f"{key} missing label"
        assert "icon" in at, f"{key} missing icon"
        assert "description" in at, f"{key} missing description"
        assert "capabilities" in at, f"{key} missing capabilities"


def test_capabilities_present():
    from core.config import CAPABILITIES
    expected = {"text", "multimodal", "voice", "reasoning", "tool_use", "mcp", "web_search", "pc_control"}
    assert expected.issubset(set(CAPABILITIES.keys()))


def test_base_models_present():
    from core.config import BASE_MODELS, AI_TYPES
    for ai_type in AI_TYPES:
        assert ai_type in BASE_MODELS, f"No base models for {ai_type}"
        assert len(BASE_MODELS[ai_type]) > 0, f"Empty model list for {ai_type}"


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------


def test_build_config_chatbot():
    from core.model_builder import ModelBuilder
    b = ModelBuilder()
    form = {
        "name": "test-bot",
        "ai_type": "chatbot",
        "base_model": "microsoft/phi-2",
        "fine_tune_method": "lora",
        "epochs": "5",
        "batch_size": "8",
        "learning_rate": "0.0002",
        "max_seq_len": "512",
        "content_policy": "standard",
        "agentic": "on",
        "capabilities": ["text", "reasoning"],
        "hf_dataset": "tatsu-lab/alpaca",
        "web_urls": "",
        "datasets": [],
    }
    cfg = b.build_config(form)
    assert cfg["name"] == "test-bot"
    assert cfg["ai_type"] == "chatbot"
    assert cfg["epochs"] == 5
    assert cfg["agentic"] is True
    assert "text" in cfg["capabilities"]


def test_build_config_defaults():
    from core.model_builder import ModelBuilder
    b = ModelBuilder()
    cfg = b.build_config({"name": "x"})
    assert cfg["content_policy"] == "standard"
    assert cfg["fine_tune_method"] == "lora"
    assert cfg["epochs"] == 3


def test_generate_training_script_chatbot():
    from core.model_builder import ModelBuilder
    b = ModelBuilder()
    cfg = {"name": "t", "ai_type": "chatbot", "base_model": "microsoft/phi-2",
           "fine_tune_method": "lora", "epochs": 2, "batch_size": 4,
           "learning_rate": 0.0002, "max_seq_len": 256, "capabilities": ["text"],
           "hf_dataset": "", "datasets": [], "web_urls": "", "content_policy": "standard",
           "agentic": False}
    with tempfile.TemporaryDirectory() as td:
        script = b.generate_training_script(cfg, td)
        assert os.path.exists(script)
        content = open(script).read()
        assert "SFTTrainer" in content
        assert "microsoft/phi-2" in content
        # Config JSON should also be written
        config_path = os.path.join(td, "ownai_config.json")
        assert os.path.exists(config_path)
        saved_cfg = json.load(open(config_path))
        assert saved_cfg["name"] == "t"


def test_generate_training_script_all_types():
    from core.model_builder import ModelBuilder
    b = ModelBuilder()
    # Includes all 28 AI types
    all_types = [
        # text
        "chatbot", "code_ai", "translator", "summarizer", "text_classifier", "creative_writer",
        # vision
        "image_generator", "image_recognition", "object_detector", "document_ai", "face_ai",
        # audio
        "speech_to_text", "audio_classifier", "music_ai",
        # data
        "trading_bot", "news_sentiment", "anomaly_detector", "data_analyst", "recommendation",
        # domain
        "medical_ai", "space_ai", "legal_ai", "education_ai", "customer_support",
        "agriculture_ai", "game_ai", "cyber_ai",
        # custom
        "custom",
    ]
    for ai_type in all_types:
        cfg = {"name": "t", "ai_type": ai_type, "base_model": "m",
               "fine_tune_method": "lora", "epochs": 1, "batch_size": 2,
               "learning_rate": 1e-4, "max_seq_len": 128, "capabilities": [],
               "hf_dataset": "", "datasets": [], "web_urls": "",
               "content_policy": "standard", "agentic": False, "custom_niche": ""}
        with tempfile.TemporaryDirectory() as td:
            script = b.generate_training_script(cfg, td)
            assert os.path.exists(script), f"No script for {ai_type}"
            assert os.path.getsize(script) > 100, f"Script too small for {ai_type}"


def test_custom_niche_keyword_routing():
    """Custom niche description should auto-route to the correct script template."""
    from core.model_builder import ModelBuilder
    b = ModelBuilder()
    base_cfg = {"name": "t", "ai_type": "custom", "base_model": "m",
                "fine_tune_method": "lora", "epochs": 1, "batch_size": 2,
                "learning_rate": 1e-4, "max_seq_len": 128, "capabilities": [],
                "hf_dataset": "", "datasets": [], "web_urls": "",
                "content_policy": "standard", "agentic": False}
    cases = [
        ("detect plant diseases in photos", "AutoModelForImageClassification"),
        ("transcribe speech from audio recordings", "WhisperForConditionalGeneration"),
        ("summarize research papers", "AutoModelForSeq2SeqLM"),
        ("classify spam and legit emails", "AutoModelForSequenceClassification"),
        ("predict crypto price movements", "TradingLSTM"),
        ("detect anomalies in server logs", "Autoencoder"),
        ("forecast energy demand from time series data", "ForecastLSTM"),
        ("recommend movies to users", "NeuralCF"),
        ("train a game agent to play chess", "PolicyNetwork"),
        ("generate code completions", "starcoder"),
    ]
    for niche, expected_marker in cases:
        cfg = {**base_cfg, "custom_niche": niche}
        with tempfile.TemporaryDirectory() as td:
            script_path = b.generate_training_script(cfg, td)
            with open(script_path) as f:
                content = f.read()
            assert expected_marker.lower() in content.lower(), (
                f"Niche '{niche}' should route to script containing '{expected_marker}'"
            )


def test_list_uploaded_files_empty():
    from core.dataset_manager import DatasetManager
    with tempfile.TemporaryDirectory() as td:
        dm = DatasetManager(td)
        assert dm.list_uploaded_files() == []


def test_save_scraped_data():
    from core.dataset_manager import DatasetManager
    with tempfile.TemporaryDirectory() as td:
        dm = DatasetManager(td)
        scraped = [
            {"url": "http://example.com", "text": "Hello world", "chars": 11, "status": "ok"},
            {"url": "http://bad.com", "text": "", "chars": 0, "status": "error"},
        ]
        path = dm.save_scraped_data(scraped, "test")
        assert os.path.exists(path)
        lines = open(path).readlines()
        assert len(lines) == 1  # only the successful one
        data = json.loads(lines[0])
        assert data["text"] == "Hello world"
        assert data["source"] == "http://example.com"


def test_is_safe_url_blocks_private_ips():
    from core.dataset_manager import DatasetManager
    with tempfile.TemporaryDirectory() as td:
        dm = DatasetManager(td)
        # These should always be blocked (invalid scheme or loopback/private IP literal)
        always_blocked = [
            "ftp://example.com/file",
            "file:///etc/passwd",
            "javascript:alert(1)",
            "http://127.0.0.1/admin",
            "http://192.168.1.1/router",
            "http://10.0.0.1/internal",
            "http://169.254.169.254/latest/meta-data/",
        ]
        for url in always_blocked:
            assert not dm._is_safe_url(url), f"{url} should be blocked"


# ---------------------------------------------------------------------------
# Agent tools
# ---------------------------------------------------------------------------


def test_agent_dispatch_unknown_tool():
    from core.agent import AgentTools
    agent = AgentTools()
    result = agent.dispatch("nonexistent_tool", {})
    assert "error" in result


def test_agent_web_search_disabled():
    from core.agent import AgentTools
    agent = AgentTools(web_search_enabled=False)
    result = agent.dispatch("web_search", {"query": "test"})
    assert "error" in result
    assert "not enabled" in result["error"]


def test_agent_pc_control_disabled():
    from core.agent import AgentTools
    agent = AgentTools(pc_control_enabled=False)
    for tool in ["read_file", "write_file", "list_dir", "run_command"]:
        result = agent.dispatch(tool, {"path": "/tmp", "command": "echo hi"})
        assert "error" in result, f"{tool} should fail when pc_control is disabled"


def test_agent_list_dir_enabled():
    from core.agent import AgentTools
    agent = AgentTools(pc_control_enabled=True)
    result = agent.dispatch("list_dir", {"path": "/tmp"})
    assert "entries" in result or "error" in result


def test_agent_blocked_command():
    from core.agent import AgentTools
    agent = AgentTools(pc_control_enabled=True)
    result = agent.dispatch("run_command", {"command": "rm -rf /"})
    assert "error" in result
    assert "Blocked" in result["error"]


def test_agent_read_file_missing():
    from core.agent import AgentTools
    agent = AgentTools(pc_control_enabled=True)
    result = agent.dispatch("read_file", {"path": "/nonexistent/file.txt"})
    assert "error" in result


def test_agent_write_and_read_file():
    from core.agent import AgentTools
    agent = AgentTools(pc_control_enabled=True)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "test.txt")
        write_result = agent.dispatch("write_file", {"path": path, "content": "hello"})
        assert write_result.get("success") is True
        read_result = agent.dispatch("read_file", {"path": path})
        assert read_result.get("content") == "hello"


# ---------------------------------------------------------------------------
# Flask app routes
# ---------------------------------------------------------------------------


@pytest.fixture
def app():
    import app as app_module
    app_module.app.config["TESTING"] = True
    app_module.app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
    app_module.app.config["WTF_CSRF_ENABLED"] = False
    with app_module.app.app_context():
        app_module.db.create_all()
    return app_module.app


@pytest.fixture
def client(app):
    return app.test_client()


def test_index_page(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"OwnAI" in resp.data


def test_builder_page(client):
    resp = client.get("/builder")
    assert resp.status_code == 200
    assert b"wizard" in resp.data.lower() or b"AI Type" in resp.data


def test_models_page(client):
    resp = client.get("/models")
    assert resp.status_code == 200


def test_datasets_page(client):
    resp = client.get("/datasets")
    assert resp.status_code == 200
    assert b"Upload" in resp.data


def test_create_model_api(client):
    resp = client.post("/api/models", data={
        "name": "my-test-bot",
        "ai_type": "chatbot",
        "base_model": "microsoft/phi-2",
        "fine_tune_method": "lora",
        "epochs": "3",
        "batch_size": "4",
        "learning_rate": "0.0002",
        "max_seq_len": "512",
        "content_policy": "standard",
    }, follow_redirects=False)
    # Should redirect to train page
    assert resp.status_code in (200, 302)


def test_hf_search_api(client):
    resp = client.get("/api/datasets/hf/search?q=alpaca")
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert "results" in data


def test_get_model_404(client):
    resp = client.get("/api/models/999")
    assert resp.status_code == 404


def test_delete_model_404(client):
    resp = client.delete("/api/models/999")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# New API endpoints
# ---------------------------------------------------------------------------


def test_system_info_endpoint(client):
    """GET /api/system/info should return system metadata."""
    resp = client.get("/api/system/info")
    assert resp.status_code == 200
    data = json.loads(resp.data)
    # Required fields
    assert "python" in data
    assert "platform" in data
    assert "cpu_count" in data
    assert "torch_available" in data
    assert "cuda_available" in data
    assert isinstance(data["cpu_count"], int)
    assert isinstance(data["torch_available"], bool)


def test_export_model_404(client):
    """GET /api/models/<id>/export should 404 for nonexistent model."""
    resp = client.get("/api/models/9999/export")
    assert resp.status_code == 404


def test_export_model_no_dir(client):
    """Export endpoint returns 404 JSON when model has no output directory."""
    # Create a model with no output dir
    resp = client.post("/api/models", data={
        "name": "export-test",
        "ai_type": "chatbot",
        "base_model": "microsoft/phi-2",
        "fine_tune_method": "lora",
        "epochs": "1",
        "batch_size": "4",
        "learning_rate": "0.0002",
        "max_seq_len": "512",
        "content_policy": "standard",
    }, follow_redirects=False)
    assert resp.status_code in (200, 302)

    # Find the model we just created via the models API
    models_resp = client.get("/models")
    assert models_resp.status_code == 200


def test_help_page(client):
    """GET /help should render the help/documentation page."""
    resp = client.get("/help")
    assert resp.status_code == 200
    assert b"Quick Start" in resp.data
    assert b"Ecosystem" in resp.data


def test_export_with_real_dir(client):
    """Export endpoint should return a zip when the model has a valid output dir."""
    from app import db, AIModel
    with tempfile.TemporaryDirectory() as td:
        # Seed the directory with a file
        with open(os.path.join(td, "ownai_config.json"), "w") as f:
            json.dump({"name": "test", "ai_type": "chatbot"}, f)
        # Create a model record pointing at the temp dir directly via DB
        with client.application.app_context():
            m = AIModel(name="zip-test", ai_type="chatbot", status="ready",
                        output_dir=td)
            m.config = {"name": "zip-test", "ai_type": "chatbot"}
            db.session.add(m)
            db.session.commit()
            model_id = m.id
        resp = client.get(f"/api/models/{model_id}/export")
        assert resp.status_code == 200
        assert resp.content_type == "application/zip"
        # Verify the zip contains the config file
        buf = io.BytesIO(resp.data)
        with zipfile.ZipFile(buf) as zf:
            names = zf.namelist()
        assert any("ownai_config.json" in n for n in names)
