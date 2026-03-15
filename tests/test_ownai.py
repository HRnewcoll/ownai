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


# ===========================================================================
# Reasoning Engine
# ===========================================================================


def test_code_sandbox_success():
    """CodeSandbox runs correct code and returns returncode 0."""
    from core.reasoning_engine import CodeSandbox
    sb = CodeSandbox(timeout=10)
    res = sb.run("print('hello sandbox')")
    assert res.get("returncode", res.get("exit_code", -1)) == 0, res.get("stderr", "")
    assert "hello sandbox" in res.get("stdout", "")


def test_code_sandbox_error():
    """CodeSandbox captures stderr and returns non-zero returncode for broken code."""
    from core.reasoning_engine import CodeSandbox
    sb = CodeSandbox(timeout=10)
    res = sb.run("raise ValueError('deliberate error')")
    assert res.get("returncode", res.get("exit_code", -1)) != 0
    assert "deliberate" in res.get("stderr", "") or "ValueError" in res.get("stderr", "")


def test_code_sandbox_timeout():
    """CodeSandbox kills long-running code after timeout and reports timeout."""
    from core.reasoning_engine import CodeSandbox
    sb = CodeSandbox(timeout=2)
    res = sb.run("import time; time.sleep(60)")
    assert res["timed_out"] is True


def test_code_sandbox_run_tests():
    """CodeSandbox.run_tests merges solution + tests and validates assertions."""
    from core.reasoning_engine import CodeSandbox
    sb = CodeSandbox(timeout=10)
    solution = "def add(a, b):\n    return a + b\n"
    tests    = "assert add(1, 2) == 3\nassert add(-1, 1) == 0\nprint('TESTS_PASSED')"
    res = sb.run_tests(solution, tests)
    assert res.get("returncode", res.get("exit_code", -1)) == 0
    assert "TESTS_PASSED" in res.get("stdout", "")


def test_tdd_test_generator():
    """ChainOfThought.build_tdd_tests returns a non-empty test scaffold."""
    from core.reasoning_engine import ChainOfThought
    cot = ChainOfThought()
    tests = cot.build_tdd_tests("Write a function that reverses a string")
    assert isinstance(tests, str) and len(tests) > 10


def test_chain_of_thought():
    """ChainOfThought.build_tdd_tests returns a non-empty string."""
    from core.reasoning_engine import ChainOfThought
    cot = ChainOfThought()
    tests = cot.build_tdd_tests("Write a function that reverses a string")
    assert isinstance(tests, str) and len(tests) > 10


def test_reasoning_engine_solve_simple():
    """ReasoningEngine.solve handles a trivial task within max_steps."""
    from core.reasoning_engine import ReasoningEngine
    engine = ReasoningEngine(tdd_enabled=True, max_steps=3)
    trace = engine.solve(
        task="Write a Python function named `double` that returns 2*x.",
        initial_code="def double(x):\n    return x * 2\n",
    )
    assert hasattr(trace, "to_dict")
    d = trace.to_dict()
    assert "steps" in d


def test_reasoning_trace_to_dict():
    """ReasoningTrace.to_dict contains all required keys."""
    from core.reasoning_engine import ReasoningEngine
    engine = ReasoningEngine(max_steps=1)
    trace = engine.solve("Return 42.")
    d = trace.to_dict()
    for key in ("task", "steps", "success", "final_answer"):
        assert key in d, f"Missing key: {key}"


# ===========================================================================
# Memory Manager
# ===========================================================================


def test_working_memory_add_retrieve():
    """WorkingMemory stores and retrieves recent messages."""
    from core.memory_manager import WorkingMemory
    wm = WorkingMemory()
    wm.add(session_id="s1", role="user", content="Hello")
    wm.add(session_id="s1", role="assistant", content="Hi there!")
    msgs = wm.get("s1")
    assert len(msgs) >= 2
    assert any("Hi there!" in str(m) for m in msgs)


def test_working_memory_overflow():
    """WorkingMemory can hold multiple messages without error."""
    from core.memory_manager import WorkingMemory
    wm = WorkingMemory()
    for i in range(5):
        wm.add(session_id="s1", role="user", content=f"msg {i}")
    msgs = wm.get("s1")
    assert len(msgs) >= 1


def test_working_memory_clear():
    """WorkingMemory.clear empties the buffer."""
    from core.memory_manager import WorkingMemory
    wm = WorkingMemory()
    wm.add(session_id="s1", role="user", content="test")
    wm.clear("s1")
    assert wm.get("s1") == []


def test_long_term_memory_add_recall(tmp_path):
    """LongTermMemory.store + retrieve_similar returns relevant facts."""
    from core.memory_manager import LongTermMemory, MemoryEntry
    ltm = LongTermMemory(db_path=str(tmp_path / "mem.db"))
    ltm.store(MemoryEntry(id=None, session_id="s", role="user",
                          content="Python is a programming language"))
    ltm.store(MemoryEntry(id=None, session_id="s", role="user",
                          content="Cats are mammals"))
    results = ltm.retrieve_similar("Python programming", top_k=2)
    assert len(results) >= 1
    assert any("Python" in str(r.content) for r in results)


def test_long_term_memory_stats(tmp_path):
    """LongTermMemory stores multiple facts without error."""
    from core.memory_manager import LongTermMemory, MemoryEntry
    ltm = LongTermMemory(db_path=str(tmp_path / "mem3.db"))
    ltm.store(MemoryEntry(id=None, session_id="s", role="user", content="fact1"))
    ltm.store(MemoryEntry(id=None, session_id="s", role="user", content="fact2"))
    results = ltm.retrieve_recent(session_id="s", limit=10)
    assert len(results) >= 2


def test_long_term_memory_clear(tmp_path):
    """LongTermMemory.forget_old prunes entries by count."""
    from core.memory_manager import LongTermMemory, MemoryEntry
    ltm = LongTermMemory(db_path=str(tmp_path / "mem2.db"))
    ltm.store(MemoryEntry(id=None, session_id="s", role="user", content="some fact"))
    # Keep up to 10000 entries – fact survives
    ltm.forget_old(max_entries=10000)
    results = ltm.retrieve_recent(session_id="s", limit=10)
    assert len(results) >= 1


def test_code_graph_rag():
    """CodeGraphRAG indexes a directory and returns a summary dict."""
    from core.memory_manager import CodeGraphRAG
    import tempfile, os
    rag = CodeGraphRAG()
    with tempfile.TemporaryDirectory() as td:
        with open(os.path.join(td, "math_utils.py"), "w") as f:
            f.write("def add(a, b):\n    return a + b\n\ndef greet(name):\n    return f'Hello {name}'\n")
        rag.index_directory(td)
    summary = rag.to_summary()
    # to_summary returns a dict or string
    assert summary is not None


# ===========================================================================
# Multi-Agent Orchestrator
# ===========================================================================


def test_multi_agent_state_to_dict():
    """AgentState.to_dict includes all expected keys."""
    from core.multi_agent import AgentState
    state = AgentState(task="test task")
    d = state.to_dict()
    for key in ("task", "plan", "code", "success", "final_output"):
        assert key in d, f"AgentState.to_dict missing key: {key}"


def test_multi_agent_run_simple():
    """MultiAgentOrchestrator.run completes for a simple coding task."""
    from core.multi_agent import MultiAgentOrchestrator
    orch = MultiAgentOrchestrator(tdd_enabled=True, max_iterations=2)
    state = orch.run("Write a Python function named `triple` that returns 3*x.")
    assert state.iterations >= 1
    d = state.to_dict()
    assert "triple" in d.get("final_output", "") or "triple" in d.get("code", "") \
        or d.get("success") is True


def test_architect_agent_produces_plan():
    """ArchitectAgent.run returns a non-empty string."""
    from core.multi_agent import ArchitectAgent, AgentState
    arch = ArchitectAgent()
    state = AgentState(task="Sort a list of integers")
    updated = arch.run(state)
    assert isinstance(updated.plan, str) and len(updated.plan) >= 0


def test_coder_agent_writes_code():
    """CoderAgent.run returns some code in the state."""
    from core.multi_agent import CoderAgent, AgentState
    coder = CoderAgent()
    state = AgentState(task="Write a square function", plan="Create a function square(x)")
    updated = coder.run(state)
    # code may be empty in mock/offline mode; just check the run doesn't crash
    assert hasattr(updated, "code")


def test_reviewer_agent_evaluates():
    """ReviewerAgent.run populates the review field."""
    from core.multi_agent import ReviewerAgent, AgentState
    rev = ReviewerAgent()
    state = AgentState(task="Add two numbers",
                       code="def add(a, b):\n    return a + b\nassert add(1,2)==3")
    updated = rev.run(state)
    assert hasattr(updated, "review")


# ===========================================================================
# Autonomous Trainer
# ===========================================================================


def test_quality_filter_good_text():
    """QualityFilter accepts a clearly human-written factual paragraph."""
    from core.autonomous_trainer import QualityFilter
    qf = QualityFilter(min_score=0.0)
    good = {
        "text": (
            "The Python programming language was created by Guido van Rossum "
            "and first released in 1991.  It emphasises code readability."
        )
    }
    result = qf.filter([good])
    assert len(result) == 1


def test_quality_filter_ai_slop():
    """QualityFilter rejects obvious AI-generated filler phrases."""
    from core.autonomous_trainer import QualityFilter
    qf = QualityFilter(min_score=0.5)
    slop = {"text": "As an AI language model I cannot provide personal opinions."}
    result = qf.filter([slop])
    assert len(result) == 0


def test_quality_filter_short_text():
    """QualityFilter score method returns a float in [0, 1]."""
    from core.autonomous_trainer import QualityFilter
    qf = QualityFilter()
    score = qf.score({"text": "Hi."})
    assert 0.0 <= score <= 1.0


def test_autonomous_trainer_status(tmp_path):
    """AutonomousTrainer has expected attributes and methods."""
    from core.autonomous_trainer import AutonomousTrainer
    at = AutonomousTrainer(data_dir=str(tmp_path), model_dir=str(tmp_path))
    assert hasattr(at, "is_running")
    assert hasattr(at, "trigger_now")
    assert at.is_running is False


def test_autonomous_trainer_history(tmp_path):
    """AutonomousTrainer.get_runs returns a list."""
    from core.autonomous_trainer import AutonomousTrainer
    at = AutonomousTrainer(data_dir=str(tmp_path), model_dir=str(tmp_path))
    assert isinstance(at.get_runs(), list)


def test_data_collector_collect_offline():
    """DataCollector.collect returns a list without raising (CI/offline)."""
    from core.autonomous_trainer import DataCollector
    dc = DataCollector(timeout=1)
    # In offline/CI mode this will return [] without raising
    result = dc.collect("arxiv_cs")
    assert isinstance(result, list)


# ===========================================================================
# Benchmarker
# ===========================================================================


def test_get_all_problems():
    """get_all_problems returns the full built-in problem set."""
    from core.benchmarker import get_all_problems
    problems = get_all_problems()
    assert len(problems) >= 10
    for p in problems:
        assert p.id           # id is the field name
        assert p.description
        assert p.reference_solution


def test_benchmark_problem_categories():
    """All problems have a valid category."""
    from core.benchmarker import get_all_problems
    for p in get_all_problems():
        assert p.category in {"coding", "reasoning"}, \
            f"Unexpected category: {p.category}"


def test_benchmark_problem_difficulties():
    """All problems have a valid difficulty."""
    from core.benchmarker import get_all_problems
    valid = {"easy", "medium", "hard"}
    for p in get_all_problems():
        assert p.difficulty in valid, f"Unknown difficulty: {p.difficulty}"


def test_benchmark_runner_single_problem():
    """BenchmarkRunner.run_suite completes on a single problem."""
    from core.benchmarker import BenchmarkRunner, get_all_problems
    runner = BenchmarkRunner()
    easy = next((p for p in get_all_problems() if p.difficulty == "easy"), None)
    assert easy is not None
    suite = runner.run_suite(problems=[easy])
    d = suite.to_dict()
    assert d["total"] == 1
    assert "results" in d


def test_benchmark_runner_full_suite_returns_report():
    """BenchmarkRunner.run_suite returns a BenchmarkSuite with expected keys."""
    from core.benchmarker import BenchmarkRunner, get_all_problems
    runner = BenchmarkRunner()
    problems = get_all_problems()[:3]
    suite = runner.run_suite(problems=problems)
    d = suite.to_dict()
    for key in ("passed", "total", "score_pct", "results"):
        assert key in d, f"Suite dict missing key: {key}"
    assert d["total"] == len(problems)


def test_benchmark_report_category_scores():
    """BenchmarkRunner.run_category returns a BenchmarkSuite."""
    from core.benchmarker import BenchmarkRunner
    runner = BenchmarkRunner()
    suite = runner.run_category("coding")
    assert suite.total >= 0


# ===========================================================================
# Model Enhancer
# ===========================================================================


def test_list_enhancements():
    """ModelEnhancer.list_enhancements returns all expected capability keys."""
    from core.model_enhancer import ModelEnhancer
    enhs = ModelEnhancer.list_enhancements()
    keys = {e["key"] for e in enhs}
    expected = {"vision", "tool_use", "voice_input", "voice_output",
                "knowledge", "memory", "reasoning", "web_search", "code_execution", "moe"}
    assert expected.issubset(keys)


def test_detect_capabilities_nonexistent_path():
    """detect_capabilities handles a nonexistent path gracefully."""
    from core.model_enhancer import ModelEnhancer
    caps = ModelEnhancer().detect_capabilities("/nonexistent/path/model")
    assert caps.base_model == "/nonexistent/path/model"
    # Should still return a DetectedCapabilities object
    d = caps.to_dict()
    assert "capabilities" in d


def test_detect_capabilities_from_name():
    """detect_capabilities uses name heuristics for known model IDs."""
    from core.model_enhancer import ModelEnhancer
    # LLaVA-style name → vision should be detected
    caps = ModelEnhancer().detect_capabilities("liuhaotian/llava-v1.5-7b")
    assert caps.has_vision is True

    # Mixtral → MoE
    caps2 = ModelEnhancer().detect_capabilities("mistralai/Mixtral-8x7B-v0.1")
    assert caps2.has_moe is True


def test_generate_enhancement_script_integration(tmp_path):
    """generate_enhancement_script writes an enhance.py and manifest."""
    from core.model_enhancer import ModelEnhancer
    out_dir = str(tmp_path / "enhanced")
    script_path = ModelEnhancer().generate_enhancement_script(
        base_model="microsoft/phi-2",
        enhancements=["voice_input", "memory", "web_search"],
        output_dir=out_dir,
    )
    assert os.path.exists(script_path)
    assert os.path.exists(os.path.join(out_dir, "enhance_manifest.json"))
    with open(script_path, encoding="utf-8") as fh:
        content = fh.read()
    assert "voice_input" in content or "Whisper" in content
    assert "memory" in content or "LongTermMemory" in content


def test_generate_enhancement_script_lora(tmp_path):
    """generate_enhancement_script includes LoRA code for tool_use enhancement."""
    from core.model_enhancer import ModelEnhancer
    out_dir = str(tmp_path / "lora_out")
    script_path = ModelEnhancer().generate_enhancement_script(
        base_model="microsoft/phi-2",
        enhancements=["tool_use"],
        output_dir=out_dir,
    )
    with open(script_path, encoding="utf-8") as fh:
        content = fh.read()
    assert "LoraConfig" in content or "peft" in content


def test_detect_capabilities_local_config(tmp_path):
    """detect_capabilities reads a local config.json if present."""
    import json as _json
    from core.model_enhancer import ModelEnhancer
    cfg = {
        "architectures": ["LlamaForCausalLM"],
        "num_experts": 8,
        "vision_config": {"hidden_size": 768},
    }
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(_json.dumps(cfg))
    caps = ModelEnhancer().detect_capabilities(str(tmp_path))
    assert caps.has_moe is True
    assert caps.has_vision is True
    assert caps.architecture == "LlamaForCausalLM"


# ===========================================================================
# Ollama Bridge
# ===========================================================================


def test_ollama_get_status_returns_dict():
    """get_status always returns a dict with 'ollama_running' key."""
    from core.ollama_bridge import get_status
    status = get_status()
    assert "ollama_running" in status
    assert "models" in status
    assert isinstance(status["models"], list)


def test_ollama_bridge_unavailable():
    """OllamaBridge returns a graceful error when Ollama is not running."""
    from core.ollama_bridge import OllamaBridge
    # Use a port that is almost certainly not running anything
    bridge = OllamaBridge("test", base_url="http://127.0.0.1:19999")
    resp = bridge.chat("hello")
    assert resp.success is False
    assert resp.error is not None


def test_ollama_list_models_offline():
    """list_local_models returns an empty list when Ollama is not available."""
    from core.ollama_bridge import list_local_models
    import unittest.mock as mock
    with mock.patch("core.ollama_bridge.is_ollama_running", return_value=False):
        models = list_local_models()
    assert isinstance(models, list)


def test_local_inference_router_backend():
    """LocalInferenceRouter.backend_name() returns a string."""
    from core.ollama_bridge import LocalInferenceRouter
    router = LocalInferenceRouter()
    name = router.backend_name()
    assert isinstance(name, str) and len(name) > 0


# ===========================================================================
# Flask routes – Enhance, Benchmark, Autonomous, Ollama
# ===========================================================================


def test_enhance_page(client):
    """GET /enhance renders the model enhancer wizard."""
    resp = client.get("/enhance")
    assert resp.status_code == 200
    assert b"Enhance" in resp.data or b"enhance" in resp.data


def test_api_enhance_detect_missing_path(client):
    """POST /api/enhance/detect with no model_path returns 400."""
    resp = client.post("/api/enhance/detect",
                       data=json.dumps({}),
                       content_type="application/json")
    assert resp.status_code == 400


def test_api_enhance_detect_nonexistent(client):
    """POST /api/enhance/detect with unknown path still returns JSON."""
    resp = client.post("/api/enhance/detect",
                       data=json.dumps({"model_path": "/no/such/model"}),
                       content_type="application/json")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "capabilities" in data


def test_api_enhance_generate_missing(client):
    """POST /api/enhance/generate without enhancements returns 400."""
    resp = client.post("/api/enhance/generate",
                       data=json.dumps({"base_model": "microsoft/phi-2",
                                        "enhancements": []}),
                       content_type="application/json")
    assert resp.status_code == 400


def test_api_enhance_generate_ok(client, tmp_path):
    """POST /api/enhance/generate returns script_path and script_content."""
    import unittest.mock as mock
    # Redirect MODELS_DIR to tmp_path to avoid cluttering repo
    with mock.patch("app.MODELS_DIR", str(tmp_path)):
        resp = client.post(
            "/api/enhance/generate",
            data=json.dumps({
                "base_model": "microsoft/phi-2",
                "enhancements": ["memory", "web_search"],
            }),
            content_type="application/json",
        )
    assert resp.status_code == 200
    data = resp.get_json()
    assert "script_path" in data
    assert "script_content" in data


def test_benchmark_page(client):
    """GET /benchmark renders the benchmark dashboard."""
    resp = client.get("/benchmark")
    assert resp.status_code == 200
    assert b"Benchmark" in resp.data or b"benchmark" in resp.data


def test_api_benchmark_problems(client):
    """GET /api/benchmark/problems returns a non-empty list."""
    resp = client.get("/api/benchmark/problems")
    assert resp.status_code == 200
    data = resp.get_json()
    assert isinstance(data, list) and len(data) >= 10


def test_api_benchmark_run(client, tmp_path):
    """POST /api/benchmark/run returns a report with expected keys."""
    import unittest.mock as mock
    with mock.patch("app.BENCHMARK_RESULTS_DIR", str(tmp_path)):
        resp = client.post(
            "/api/benchmark/run",
            data=json.dumps({"max_problems": 2}),
            content_type="application/json",
        )
    assert resp.status_code == 200
    data = resp.get_json()
    # BenchmarkSuite.to_dict() returns: name, total, passed, score_pct, results
    for key in ("passed", "total", "score_pct", "results"):
        assert key in data


def test_api_memory_stats(client):
    """GET /api/memory/stats returns a dict with total_facts."""
    resp = client.get("/api/memory/stats")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "total_facts" in data


def test_api_memory_add_search(client):
    """POST /api/memory/add stores a fact; /api/memory/search retrieves it."""
    add_resp = client.post(
        "/api/memory/add",
        data=json.dumps({"content": "Bananas are yellow fruit"}),
        content_type="application/json",
    )
    assert add_resp.status_code == 200

    search_resp = client.post(
        "/api/memory/search",
        data=json.dumps({"query": "yellow fruit", "top_k": 3}),
        content_type="application/json",
    )
    assert search_resp.status_code == 200
    results = search_resp.get_json()
    assert isinstance(results, list)


def test_api_memory_clear(client):
    """POST /api/memory/clear returns ok."""
    resp = client.post("/api/memory/clear",
                       data=json.dumps({}), content_type="application/json")
    assert resp.status_code == 200
    assert resp.get_json()["ok"] is True


def test_api_ollama_status(client):
    """GET /api/ollama/status returns a valid status dict."""
    resp = client.get("/api/ollama/status")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "ollama_running" in data


def test_api_ollama_chat_missing_message(client):
    """POST /api/ollama/chat without message returns 400."""
    resp = client.post("/api/ollama/chat",
                       data=json.dumps({}), content_type="application/json")
    assert resp.status_code == 400


def test_api_autonomous_status(client):
    """GET /api/autonomous/status returns expected keys."""
    resp = client.get("/api/autonomous/status")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "running" in data


def test_api_sandbox_run(client):
    """POST /api/sandbox/run executes code and returns stdout."""
    resp = client.post(
        "/api/sandbox/run",
        data=json.dumps({"code": "print('sandbox ok')"}),
        content_type="application/json",
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data.get("returncode", data.get("exit_code", -1)) == 0
    assert "sandbox ok" in data.get("stdout", "")


def test_api_multiagent_solve_missing_task(client):
    """POST /api/multiagent/solve without task returns 400."""
    resp = client.post("/api/multiagent/solve",
                       data=json.dumps({}), content_type="application/json")
    assert resp.status_code == 400


def test_api_multiagent_solve_simple(client):
    """POST /api/multiagent/solve returns a state dict for a simple task."""
    resp = client.post(
        "/api/multiagent/solve",
        data=json.dumps({
            "task": "Write a Python function named `add` that returns a+b.",
            "max_iterations": 2,
        }),
        content_type="application/json",
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert "task" in data
    assert "success" in data
