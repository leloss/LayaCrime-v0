from __future__ import annotations

import os
import re
import subprocess
import threading
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PHASE_PATTERN = re.compile(r"^phase=(?P<phase>[a-z_-]+)$")
TOTAL_PATTERN = re.compile(r"^download_total=(?P<total>\d+)$")
PATH_PATTERN = re.compile(r"^installed_path=(?P<path>.+)$")


class ModelInstallationJob:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._process: subprocess.Popen[str] | None = None
        self._logs: list[dict[str, Any]] = []
        self.status = "idle"
        self.phase: str | None = None
        self.model_id: str | None = None
        self.started_at: str | None = None
        self.exit_code: int | None = None
        self.error: str | None = None
        self.total_bytes = 0
        self.installed_path: Path | None = None
        self.models_dir: Path | None = None

    def start(
        self,
        command: list[str],
        model_id: str,
        models_dir: Path,
        cwd: Path,
        *,
        stdin_data: str,
        env: Mapping[str, str] | None = None,
    ) -> None:
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                raise RuntimeError("another model installation is already running")
            process_env = dict(env) if env is not None else os.environ.copy()
            self._logs = []
            self.status = "running"
            self.phase = "preparing"
            self.model_id = model_id
            self.started_at = datetime.now(timezone.utc).isoformat()
            self.exit_code = None
            self.error = None
            self.total_bytes = 0
            self.installed_path = None
            self.models_dir = models_dir
            try:
                self._process = subprocess.Popen(
                    command,
                    cwd=cwd,
                    env=process_env,
                    stdin=subprocess.PIPE,
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
            assert process.stdin is not None
            process.stdin.write(stdin_data)
            process.stdin.close()
        threading.Thread(
            target=self._collect_output,
            args=(process,),
            name="laya-model-installation",
            daemon=True,
        ).start()

    def _collect_output(self, process: subprocess.Popen[str]) -> None:
        assert process.stdout is not None
        for line in process.stdout:
            message = line.rstrip()
            with self._lock:
                phase = PHASE_PATTERN.match(message)
                total = TOTAL_PATTERN.match(message)
                installed = PATH_PATTERN.match(message)
                if phase:
                    self.phase = phase.group("phase").replace("_", " ")
                elif total:
                    self.total_bytes = int(total.group("total"))
                elif installed:
                    self.installed_path = Path(installed.group("path"))
                elif message:
                    self._logs.append({
                        "index": len(self._logs) + 1,
                        "message": message,
                    })
        exit_code = process.wait()
        with self._lock:
            self.exit_code = exit_code
            if self.status == "stopping":
                self.status = "stopped"
            elif (
                exit_code == 0
                and self.installed_path is not None
                and self.installed_path.is_file()
            ):
                self.status = "complete"
                self.phase = "ready"
            else:
                self.status = "failed"
                self.error = self._logs[-1]["message"] if self._logs else (
                    "installation finished without producing a verified model"
                    if exit_code == 0
                    else f"installation exited with code {exit_code}"
                )

    def stop(self) -> None:
        with self._lock:
            process = self._process
            if process is None or process.poll() is not None:
                return
            self.status = "stopping"
            process.terminate()

    def _downloaded_bytes(self) -> int:
        if self.installed_path is not None and self.installed_path.is_file():
            return self.installed_path.stat().st_size
        if self.models_dir is None or not self.models_dir.is_dir():
            return 0
        candidates = list(self.models_dir.rglob("*.incomplete"))
        return max((path.stat().st_size for path in candidates), default=0)

    def snapshot(self, logs_after: int = 0) -> dict[str, Any]:
        with self._lock:
            downloaded = self._downloaded_bytes()
            fraction = (
                min(1.0, downloaded / self.total_bytes)
                if self.total_bytes > 0 else None
            )
            process = self._process
            return {
                "status": self.status,
                "phase": self.phase,
                "model_id": self.model_id,
                "started_at": self.started_at,
                "pid": process.pid if process is not None and process.poll() is None else None,
                "exit_code": self.exit_code,
                "error": self.error,
                "downloaded_bytes": downloaded,
                "total_bytes": self.total_bytes,
                "progress": fraction,
                "installed_path": str(self.installed_path) if self.installed_path else None,
                "logs": [row.copy() for row in self._logs if row["index"] > logs_after],
            }