"""Training manager – runs training scripts in background threads with live log streaming."""
import os
import json
import subprocess
import threading
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


class TrainingManager:
    """Manages background training processes with live log streaming via SocketIO."""

    def __init__(self, socketio=None):
        self.socketio = socketio
        self._processes: dict[int, subprocess.Popen] = {}  # model_id -> process
        self._threads: dict[int, threading.Thread] = {}

    def start_training(self, model_id: int, script_path: str, output_dir: str, db, Model) -> bool:
        """Launch training script as a subprocess and stream logs."""
        if model_id in self._processes:
            proc = self._processes[model_id]
            if proc.poll() is None:
                logger.warning("Training already running for model %s", model_id)
                return False

        model = db.session.get(Model, model_id)
        if not model:
            return False

        model.status = "training"
        model.training_started_at = datetime.utcnow()
        db.session.commit()

        def _run():
            try:
                proc = subprocess.Popen(
                    ["python", script_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    cwd=output_dir,
                )
                self._processes[model_id] = proc
                log_path = os.path.join(output_dir, "training.log")
                with open(log_path, "w") as log_file:
                    for line in proc.stdout:
                        log_file.write(line)
                        log_file.flush()
                        if self.socketio:
                            self.socketio.emit("training_log", {
                                "model_id": model_id,
                                "line": line.rstrip(),
                            })
                proc.wait()
                # Update model status
                with db.app.app_context():
                    m = db.session.get(Model, model_id)
                    if m:
                        m.status = "ready" if proc.returncode == 0 else "failed"
                        m.training_finished_at = datetime.utcnow()
                        db.session.commit()
                if self.socketio:
                    self.socketio.emit("training_complete", {
                        "model_id": model_id,
                        "success": proc.returncode == 0,
                    })
            except Exception as e:
                logger.error("Training error for model %s: %s", model_id, e)
                with db.app.app_context():
                    m = db.session.get(Model, model_id)
                    if m:
                        m.status = "failed"
                        db.session.commit()

        t = threading.Thread(target=_run, daemon=True)
        self._threads[model_id] = t
        t.start()
        return True

    def stop_training(self, model_id: int) -> bool:
        """Terminate a running training process."""
        proc = self._processes.get(model_id)
        if proc and proc.poll() is None:
            proc.terminate()
            return True
        return False

    def get_log(self, output_dir: str, tail: int = 200) -> list:
        """Return the last `tail` lines of the training log."""
        log_path = os.path.join(output_dir, "training.log")
        if not os.path.exists(log_path):
            return []
        with open(log_path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        return [l.rstrip() for l in lines[-tail:]]

    def is_running(self, model_id: int) -> bool:
        proc = self._processes.get(model_id)
        return proc is not None and proc.poll() is None
