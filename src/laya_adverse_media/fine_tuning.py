from __future__ import annotations

import os
import re
import subprocess
import threading
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROGRESS_PATTERN = re.compile(
    r"epoch=(?P<epoch>\d+)/(?P<epochs>\d+)\s+batch=(?P<batch>\d+)\s+"
    r"loss=(?P<loss>[\d.]+)\s+sigma=(?P<sigma>[\d.]+)"
)
EPOCH_METRICS_PATTERN = re.compile(
    r"epoch_metrics\s+epoch=(?P<epoch>\d+)\s+"
    r"train_loss=(?P<train_loss>[\d.eE+-]+)\s+"
    r"validation_loss=(?P<validation_loss>[\d.eE+-]+)\s+"
    r"(?:train_accuracy=(?P<train_accuracy>[\d.eE+-]+)\s+)?"
    r"validation_accuracy=(?P<validation_accuracy>[\d.eE+-]+)"
)


class FineTuningJob:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._process: subprocess.Popen[str] | None = None
        self._logs: list[dict[str, Any]] = []
        self.status = "idle"
        self.phase: str | None = None
        self.started_at: str | None = None
        self.command: list[str] = []
        self.exit_code: int | None = None
        self.error: str | None = None
        self.progress: dict[str, Any] = {}
        self.loss_history: list[dict[str, Any]] = []

    def start(
        self,
        command: list[str],
        phase: str,
        cwd: Path,
        env: Mapping[str, str] | None = None,
    ) -> None:
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                raise RuntimeError("a fine-tuning job is already running")
            self._logs = []
            self.status = "running"
            self.phase = phase
            self.started_at = datetime.now(timezone.utc).isoformat()
            self.command = command
            self.exit_code = None
            self.error = None
            self.progress = {}
            self.loss_history = []
            try:
                process_env = os.environ.copy()
                if env:
                    process_env.update(env)
                self._process = subprocess.Popen(
                    command,
                    cwd=cwd,
                    env=process_env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                )
            except OSError as exc:
                self.status = "failed"
                self.error = str(exc)
                raise
            process = self._process
        threading.Thread(
            target=self._collect_output,
            args=(process,),
            name="laya-fine-tuning",
            daemon=True,
        ).start()

    def _collect_output(self, process: subprocess.Popen[str]) -> None:
        assert process.stdout is not None
        for line in process.stdout:
            message = line.rstrip()
            with self._lock:
                self._logs.append({"index": len(self._logs) + 1, "message": message})
                match = PROGRESS_PATTERN.search(message)
                if match:
                    values = match.groupdict()
                    self.progress = {
                        "epoch": int(values["epoch"]),
                        "epochs": int(values["epochs"]),
                        "batch": int(values["batch"]),
                        "loss": float(values["loss"]),
                        "sigma": float(values["sigma"]),
                    }
                epoch_metrics = EPOCH_METRICS_PATTERN.search(message)
                if epoch_metrics:
                    values = epoch_metrics.groupdict()
                    self.loss_history.append(
                        {
                            "epoch": int(values["epoch"]),
                            "train_loss": float(values["train_loss"]),
                            "validation_loss": float(values["validation_loss"]),
                            "train_accuracy": (
                                float(values["train_accuracy"])
                                if values["train_accuracy"] is not None
                                else None
                            ),
                            "validation_accuracy": float(values["validation_accuracy"]),
                        }
                    )
        exit_code = process.wait()
        with self._lock:
            self.exit_code = exit_code
            if self.status == "stopping":
                self.status = "stopped"
            else:
                self.status = "complete" if exit_code == 0 else "failed"
                if exit_code != 0:
                    self.error = f"{self.phase or 'job'} exited with code {exit_code}"

    def stop(self) -> None:
        with self._lock:
            process = self._process
            if process is None or process.poll() is not None:
                return
            self.status = "stopping"
            process.terminate()

    def snapshot(self, logs_after: int = 0) -> dict[str, Any]:
        with self._lock:
            process = self._process
            return {
                "status": self.status,
                "phase": self.phase,
                "pid": process.pid if process is not None and process.poll() is None else None,
                "started_at": self.started_at,
                "command": list(self.command),
                "exit_code": self.exit_code,
                "error": self.error,
                "progress": dict(self.progress),
                "loss_history": [row.copy() for row in self.loss_history],
                "logs": [row.copy() for row in self._logs if row["index"] > logs_after],
            }