"""OwnAI – The AI Development Platform for Everyone.

Run with:
    python app.py

Then open http://localhost:5000 in your browser.
"""
import io
import os
import json
import uuid
import zipfile
import logging
from datetime import datetime

from flask import Flask, render_template, request, redirect, url_for, jsonify, flash, send_from_directory, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_socketio import SocketIO, emit

from core.config import (
    MODELS_DIR, UPLOADS_DIR, DATABASE_PATH,
    AI_TYPES, AI_CATEGORIES, CAPABILITIES, BASE_MODELS,
)
from core.model_builder import ModelBuilder
from core.trainer import TrainingManager
from core.dataset_manager import DatasetManager
from core.agent import AgentTools, TOOLS

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s – %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", os.urandom(32).hex())
app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{DATABASE_PATH}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB upload limit

db = SQLAlchemy(app)
socketio = SocketIO(app, async_mode="eventlet", cors_allowed_origins="*")

builder = ModelBuilder()
dataset_mgr = DatasetManager(UPLOADS_DIR)
trainer_mgr = TrainingManager(socketio)

# ---------------------------------------------------------------------------
# Database models
# ---------------------------------------------------------------------------


class AIModel(db.Model):
    __tablename__ = "ai_models"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    ai_type = db.Column(db.String(64), nullable=False)
    status = db.Column(db.String(32), default="draft")  # draft | training | ready | failed
    config_json = db.Column(db.Text, default="{}")
    output_dir = db.Column(db.String(512))
    script_path = db.Column(db.String(512))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    training_started_at = db.Column(db.DateTime)
    training_finished_at = db.Column(db.DateTime)

    @property
    def config(self):
        try:
            return json.loads(self.config_json or "{}")
        except Exception:
            return {}

    @config.setter
    def config(self, value: dict):
        self.config_json = json.dumps(value)

    def to_dict(self):
        cfg = self.config
        return {
            "id": self.id,
            "name": self.name,
            "ai_type": self.ai_type,
            "status": self.status,
            "config": cfg,
            "output_dir": self.output_dir,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "training_started_at": self.training_started_at.isoformat() if self.training_started_at else None,
            "training_finished_at": self.training_finished_at.isoformat() if self.training_finished_at else None,
            "icon": AI_TYPES.get(self.ai_type, {}).get("icon", "🤖"),
            "color": AI_TYPES.get(self.ai_type, {}).get("color", "#6366f1"),
        }


class Dataset(db.Model):
    __tablename__ = "datasets"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    source = db.Column(db.String(32), default="upload")  # upload | hf | web
    path = db.Column(db.String(512))
    hf_id = db.Column(db.String(200))
    size_bytes = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "source": self.source,
            "path": self.path,
            "hf_id": self.hf_id,
            "size_bytes": self.size_bytes,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


# ---------------------------------------------------------------------------
# Routes – Pages
# ---------------------------------------------------------------------------


@app.route("/")
def index():
    models = AIModel.query.order_by(AIModel.created_at.desc()).all()
    datasets = Dataset.query.order_by(Dataset.created_at.desc()).all()
    stats = {
        "total_models": len(models),
        "ready_models": sum(1 for m in models if m.status == "ready"),
        "training_models": sum(1 for m in models if m.status == "training"),
        "total_datasets": len(datasets),
    }
    return render_template("index.html", models=models, stats=stats, ai_types=AI_TYPES, ai_categories=AI_CATEGORIES)


@app.route("/builder")
def builder_page():
    return render_template(
        "builder.html",
        ai_types=AI_TYPES,
        ai_categories=AI_CATEGORIES,
        capabilities=CAPABILITIES,
        base_models=BASE_MODELS,
    )


@app.route("/builder/<int:model_id>")
def builder_edit(model_id):
    model = db.get_or_404(AIModel, model_id)
    return render_template(
        "builder.html",
        ai_types=AI_TYPES,
        ai_categories=AI_CATEGORIES,
        capabilities=CAPABILITIES,
        base_models=BASE_MODELS,
        editing=model,
    )


@app.route("/models")
def models_page():
    models = AIModel.query.order_by(AIModel.created_at.desc()).all()
    return render_template("models.html", models=models, ai_types=AI_TYPES)


@app.route("/train/<int:model_id>")
def train_page(model_id):
    model = db.get_or_404(AIModel, model_id)
    datasets = Dataset.query.order_by(Dataset.created_at.desc()).all()
    log_lines = trainer_mgr.get_log(model.output_dir or "") if model.output_dir else []
    return render_template(
        "train.html",
        model=model,
        datasets=datasets,
        log_lines=log_lines,
        is_running=trainer_mgr.is_running(model_id),
        ai_types=AI_TYPES,
    )


@app.route("/run/<int:model_id>")
def run_page(model_id):
    model = db.get_or_404(AIModel, model_id)
    cfg = model.config
    tools_available = {k: v for k, v in TOOLS.items()
                       if k in ("web_search", "read_file", "write_file", "run_command", "list_dir", "open_browser")}
    return render_template(
        "run.html",
        model=model,
        tools=tools_available,
        ai_types=AI_TYPES,
    )


@app.route("/datasets")
def datasets_page():
    datasets = Dataset.query.order_by(Dataset.created_at.desc()).all()
    return render_template("datasets.html", datasets=datasets)


# ---------------------------------------------------------------------------
# API – Model CRUD
# ---------------------------------------------------------------------------


@app.route("/api/models", methods=["POST"])
def api_create_model():
    """Create a new AI model from wizard form data."""
    try:
        cfg = builder.build_config(request.form)
        name = cfg["name"] or f"ai_{uuid.uuid4().hex[:6]}"

        # Create output directory
        safe_name = "".join(c if c.isalnum() else "_" for c in name)
        out_dir = os.path.join(MODELS_DIR, f"{safe_name}_{uuid.uuid4().hex[:6]}")
        os.makedirs(out_dir, exist_ok=True)
        cfg["output_dir"] = out_dir

        # Generate training script
        script_path = builder.generate_training_script(cfg, out_dir)

        model = AIModel(
            name=name,
            ai_type=cfg["ai_type"],
            status="draft",
            output_dir=out_dir,
            script_path=script_path,
        )
        model.config = cfg
        db.session.add(model)
        db.session.commit()

        flash(f'✅ AI model "{name}" created successfully!', "success")
        return redirect(url_for("train_page", model_id=model.id))
    except Exception as e:
        logger.error("Error creating model: %s", e)
        flash(f"❌ Error creating model: {e}", "danger")
        return redirect(url_for("builder_page"))


@app.route("/api/models/<int:model_id>", methods=["DELETE"])
def api_delete_model(model_id):
    model = db.get_or_404(AIModel, model_id)
    # Stop training if running
    trainer_mgr.stop_training(model_id)
    # Remove output directory
    import shutil
    if model.output_dir and os.path.exists(model.output_dir):
        try:
            shutil.rmtree(model.output_dir)
        except Exception as e:
            logger.warning("Could not delete model dir: %s", e)
    db.session.delete(model)
    db.session.commit()
    return jsonify({"success": True})


@app.route("/api/models/<int:model_id>", methods=["GET"])
def api_get_model(model_id):
    model = db.get_or_404(AIModel, model_id)
    return jsonify(model.to_dict())


@app.route("/api/models/<int:model_id>/script")
def api_get_script(model_id):
    model = db.get_or_404(AIModel, model_id)
    if not model.script_path or not os.path.exists(model.script_path):
        return jsonify({"error": "Script not found"}), 404
    with open(model.script_path, encoding="utf-8") as f:
        content = f.read()
    return jsonify({"script": content, "path": model.script_path})


# ---------------------------------------------------------------------------
# API – Training
# ---------------------------------------------------------------------------


@app.route("/api/train/<int:model_id>/start", methods=["POST"])
def api_start_training(model_id):
    model = db.get_or_404(AIModel, model_id)
    if not model.script_path:
        return jsonify({"error": "No training script found. Save the model first."}), 400
    ok = trainer_mgr.start_training(model_id, model.script_path, model.output_dir, db, AIModel)
    if ok:
        return jsonify({"success": True, "message": "Training started"})
    return jsonify({"error": "Training already running or could not start"}), 400


@app.route("/api/train/<int:model_id>/stop", methods=["POST"])
def api_stop_training(model_id):
    ok = trainer_mgr.stop_training(model_id)
    if ok:
        model = db.get_or_404(AIModel, model_id)
        model.status = "draft"
        db.session.commit()
    return jsonify({"success": ok})


@app.route("/api/train/<int:model_id>/logs")
def api_get_logs(model_id):
    model = db.get_or_404(AIModel, model_id)
    lines = trainer_mgr.get_log(model.output_dir or "") if model.output_dir else []
    return jsonify({"lines": lines, "is_running": trainer_mgr.is_running(model_id)})


@app.route("/api/train/<int:model_id>/status")
def api_training_status(model_id):
    model = db.get_or_404(AIModel, model_id)
    return jsonify({
        "status": model.status,
        "is_running": trainer_mgr.is_running(model_id),
        "started": model.training_started_at.isoformat() if model.training_started_at else None,
        "finished": model.training_finished_at.isoformat() if model.training_finished_at else None,
    })


# ---------------------------------------------------------------------------
# API – Datasets
# ---------------------------------------------------------------------------


@app.route("/api/datasets", methods=["POST"])
def api_upload_dataset():
    name = request.form.get("name", "")
    file = request.files.get("file")
    if file and file.filename:
        meta = dataset_mgr.save_uploaded_file(file, name)
        ds = Dataset(
            name=name or file.filename,
            source="upload",
            path=meta["path"],
            size_bytes=meta["size_bytes"],
        )
        db.session.add(ds)
        db.session.commit()
        return jsonify({"success": True, "dataset": ds.to_dict()})
    return jsonify({"error": "No file provided"}), 400


@app.route("/api/datasets/hf/search")
def api_hf_search():
    query = request.args.get("q", "")
    results = dataset_mgr.search_hf_datasets(query)
    return jsonify({"results": results})


@app.route("/api/datasets/hf/add", methods=["POST"])
def api_hf_add():
    data = request.get_json() or {}
    hf_id = data.get("hf_id", "")
    if not hf_id:
        return jsonify({"error": "No HuggingFace dataset ID provided"}), 400
    ds = Dataset(name=hf_id, source="hf", hf_id=hf_id)
    db.session.add(ds)
    db.session.commit()
    return jsonify({"success": True, "dataset": ds.to_dict()})


@app.route("/api/datasets/web/scrape", methods=["POST"])
def api_web_scrape():
    data = request.get_json() or {}
    urls = data.get("urls", [])
    name = data.get("name", "web_dataset")
    if not urls:
        return jsonify({"error": "No URLs provided"}), 400
    scraped = dataset_mgr.scrape_urls(urls)
    path = dataset_mgr.save_scraped_data(scraped, name)
    total_chars = sum(s.get("chars", 0) for s in scraped)
    ds = Dataset(
        name=name,
        source="web",
        path=path,
        size_bytes=os.path.getsize(path) if os.path.exists(path) else 0,
    )
    db.session.add(ds)
    db.session.commit()
    return jsonify({
        "success": True,
        "dataset": ds.to_dict(),
        "pages_scraped": len([s for s in scraped if s.get("status") == "ok"]),
        "total_chars": total_chars,
    })


@app.route("/api/datasets/<int:dataset_id>", methods=["DELETE"])
def api_delete_dataset(dataset_id):
    ds = db.get_or_404(Dataset, dataset_id)
    if ds.path and os.path.exists(ds.path):
        try:
            os.remove(ds.path)
        except Exception:
            pass
    db.session.delete(ds)
    db.session.commit()
    return jsonify({"success": True})


# ---------------------------------------------------------------------------
# API – Agent / Inference
# ---------------------------------------------------------------------------


@app.route("/api/chat/<int:model_id>", methods=["POST"])
def api_chat(model_id):
    """Simple inference endpoint – falls back to a demo response if model not loaded."""
    model = db.get_or_404(AIModel, model_id)
    data = request.get_json() or {}
    user_message = data.get("message", "").strip()
    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    cfg = model.config
    capabilities = cfg.get("capabilities", [])
    agentic = cfg.get("agentic", False)

    # Tool execution
    tool_results = []
    if agentic and "web_search" in capabilities and user_message.startswith("/search "):
        query = user_message[8:].strip()
        agent = AgentTools(
            pc_control_enabled="pc_control" in capabilities,
            web_search_enabled="web_search" in capabilities,
        )
        result = agent.dispatch("web_search", {"query": query})
        tool_results.append({"tool": "web_search", "result": result})

    # Try to load model for real inference
    response_text = _run_inference(model, user_message)

    return jsonify({
        "response": response_text,
        "model_id": model_id,
        "model_name": model.name,
        "tool_results": tool_results,
    })


def _run_inference(model: AIModel, message: str) -> str:
    """Attempt real inference; fall back to placeholder if model not loaded."""
    cfg = model.config
    output_dir = model.output_dir or ""
    base_model = cfg.get("base_model", "")

    # Only attempt if model is ready and has a checkpoint
    if model.status != "ready":
        return (
            f"🤖 **{model.name}** is not ready yet (status: {model.status}). "
            "Start training first, then come back to chat!"
        )

    try:
        from transformers import pipeline as hf_pipeline
        # Look for saved model in output_dir
        model_path = output_dir if os.path.isdir(output_dir) else base_model
        pipe = hf_pipeline("text-generation", model=model_path, max_new_tokens=200)
        result = pipe(message)[0]["generated_text"]
        # Return only the newly generated part
        if result.startswith(message):
            result = result[len(message):].strip()
        return result or "(No response generated)"
    except Exception as e:
        return f"⚠️ Could not load model for inference: {e}\n\nOnce your model is fully trained you can chat here."


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOADS_DIR, filename)
def api_agent_tool():
    """Execute an agent tool call."""
    data = request.get_json() or {}
    model_id = data.get("model_id")
    tool_name = data.get("tool")
    params = data.get("params", {})

    if not model_id or not tool_name:
        return jsonify({"error": "model_id and tool are required"}), 400

    model = db.get_or_404(AIModel, model_id)
    cfg = model.config
    capabilities = cfg.get("capabilities", [])

    agent = AgentTools(
        pc_control_enabled="pc_control" in capabilities,
        web_search_enabled="web_search" in capabilities,
    )
    result = agent.dispatch(tool_name, params)
    return jsonify(result)


# ---------------------------------------------------------------------------
# Static file helper
# ---------------------------------------------------------------------------


@app.route("/help")
def help_page():
    return render_template("help.html", ai_types=AI_TYPES, ai_categories=AI_CATEGORIES)


# ---------------------------------------------------------------------------
# API – System info
# ---------------------------------------------------------------------------


@app.route("/api/system/info")
def api_system_info():
    """Return system capability info: GPU, CPU, memory, Python, torch."""
    import sys
    import platform as _platform
    info: dict = {
        "python": sys.version.split()[0],
        "platform": _platform.system(),
        "cpu_count": os.cpu_count() or 1,
        "torch_available": False,
        "cuda_available": False,
        "cuda_device": None,
        "cuda_memory_gb": None,
        "ram_gb": None,
    }
    # RAM
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / (1024 ** 3), 1)
        info["ram_used_gb"] = round(psutil.virtual_memory().used / (1024 ** 3), 1)
    except ImportError:
        pass
    # Torch / CUDA
    try:
        import torch
        info["torch_available"] = True
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_device"] = torch.cuda.get_device_name(0)
            info["cuda_memory_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 1
            )
    except ImportError:
        pass
    return jsonify(info)


# ---------------------------------------------------------------------------
# API – Model export (download as zip)
# ---------------------------------------------------------------------------


@app.route("/api/models/<int:model_id>/export")
def api_export_model(model_id):
    """Package the model output directory as a downloadable zip file."""
    model = db.get_or_404(AIModel, model_id)
    out_dir = model.output_dir
    if not out_dir or not os.path.isdir(out_dir):
        return jsonify({"error": "No output directory found. Train the model first."}), 404

    # Stream zip into memory to avoid large temp files
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _dirs, files in os.walk(out_dir):
            for fname in files:
                full_path = os.path.join(root, fname)
                arcname = os.path.relpath(full_path, os.path.dirname(out_dir))
                zf.write(full_path, arcname)
    buf.seek(0)
    safe_name = "".join(c if c.isalnum() else "_" for c in model.name)
    return send_file(
        buf,
        mimetype="application/zip",
        as_attachment=True,
        download_name=f"{safe_name}_model.zip",
    )


# ---------------------------------------------------------------------------
# SocketIO events
# ---------------------------------------------------------------------------


@socketio.on("connect")
def handle_connect():
    logger.info("Client connected: %s", request.sid)


@socketio.on("disconnect")
def handle_disconnect():
    logger.info("Client disconnected: %s", request.sid)


@socketio.on("join_training")
def handle_join(data):
    model_id = data.get("model_id")
    if model_id:
        # Send existing log lines to newly-connected client
        model = db.session.get(AIModel, model_id)
        if model and model.output_dir:
            lines = trainer_mgr.get_log(model.output_dir)
            emit("training_history", {"model_id": model_id, "lines": lines})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def create_app():
    with app.app_context():
        db.create_all()
    return app


if __name__ == "__main__":
    create_app()
    port = int(os.environ.get("PORT", 5000))
    print(f"\n🚀 OwnAI is running at http://localhost:{port}\n")
    socketio.run(app, host="0.0.0.0", port=port, debug=False)
