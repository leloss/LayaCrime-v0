from __future__ import annotations

import csv
import importlib.util
import json
import os
import re
import subprocess
import threading
import zipfile
from pathlib import Path
from typing import Any, Mapping

ACADEMIC_KHANDPUR_ID = "academic-khandpur-entity-relevance"
ACADEMIC_NEWSMTSC_ID = "academic-newsmtsc-grutsc-v1"
ACADEMIC_TARTU_ID = "academic-tartu-tfidf-mnb"
KHANDPUR_SEED = 20261005
TARTU_CLEANUP_PATTERN = re.compile(
    r"(http\S+)|(#(\w+))|(@(\w+))|[^\w\s]|(\w*\d\w*)"
)
TARTU_WHITESPACE_PATTERN = re.compile(r"(\s+)|(\n+)")
ACADEMIC_SETUP_HINT = "run scripts/setup_academic_models.sh"


def _modules_available(*names: str) -> bool:
    return all(importlib.util.find_spec(name) is not None for name in names)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def _target_context(article: str, entity: str, radius: int = 900) -> str:
    compact = re.sub(r"\s+", " ", article).strip()
    matches = list(re.finditer(re.escape(entity.strip()), compact, re.IGNORECASE))
    if not matches:
        return f"[TARGET] {entity.strip()} [/TARGET] [SEP] {compact[: radius * 2]}"
    windows = []
    for match in matches[:3]:
        start = max(0, match.start() - radius)
        end = min(len(compact), match.end() + radius)
        windows.append(
            compact[start : match.start()]
            + f" [TARGET] {match.group(0)} [/TARGET] "
            + compact[match.end() : end]
        )
    return " [SEP] ".join(windows) + f" [SEP] {entity.strip()}"


def _clean_tartu_text(text: str) -> str:
    cleaned = TARTU_CLEANUP_PATTERN.sub("", text)
    return TARTU_WHITESPACE_PATTERN.sub(" ", cleaned).strip().lower()


def _read_zip_csv(path: Path) -> list[dict[str, str]]:
    csv.field_size_limit(2_147_483_647)
    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist() if not name.endswith("/")]
        if len(members) != 1:
            raise ValueError(f"expected one CSV in {path}")
        with archive.open(members[0]) as raw_source:
            return list(csv.DictReader(line.decode("utf-8-sig") for line in raw_source))


def _newsmtsc_python(project_root: Path) -> Path | None:
    configured = os.environ.get("LAYA_NEWSMTSC_PYTHON")
    candidates = [
        Path(configured).expanduser() if configured else None,
        project_root / ".venv-newsmtsc" / "Scripts" / "python.exe",
        project_root / ".venv-newsmtsc" / "bin" / "python",
        Path.home() / ".cache" / "laya-adverse-media" / "newsmtsc-venv" / "bin" / "python",
    ]
    return next((path for path in candidates if path is not None and path.is_file()), None)


class _NewsMtscWorker:
    def __init__(self, project_root: Path, python: Path) -> None:
        self._process = subprocess.Popen(
            [str(python), str(project_root / "scripts" / "newsmtsc_worker.py")],
            cwd=project_root,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        self._recent_output: list[str] = []
        ready = self._read_protocol_line("ready=")
        if ready != "1":
            self.close()
            raise RuntimeError(f"NewsMTSC worker did not become ready: {ready}")

    def _read_protocol_line(self, prefix: str) -> str:
        assert self._process.stdout is not None
        for line in self._process.stdout:
            message = line.rstrip()
            if message.startswith(prefix):
                return message.removeprefix(prefix)
            if message:
                self._recent_output = [*self._recent_output[-19:], message]
        detail = self._recent_output[-1] if self._recent_output else "worker exited"
        raise RuntimeError(f"NewsMTSC worker failed: {detail}")

    def predict(self, article: str, entity_name: str) -> Mapping[str, Any]:
        if self._process.poll() is not None:
            raise RuntimeError("NewsMTSC worker is not running")
        assert self._process.stdin is not None
        self._process.stdin.write(json.dumps({
            "article": article,
            "entity_name": entity_name,
        }, ensure_ascii=False) + "\n")
        self._process.stdin.flush()
        response = json.loads(self._read_protocol_line("result="))
        if response.get("error"):
            raise RuntimeError(str(response["error"]))
        return response

    def close(self) -> None:
        if self._process.poll() is not None:
            return
        try:
            if self._process.stdin is not None:
                self._process.stdin.write('{"command":"shutdown"}\n')
                self._process.stdin.flush()
            self._process.wait(timeout=5)
        except (BrokenPipeError, subprocess.TimeoutExpired):
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()


class AcademicModelRuntime:
    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self._lock = threading.RLock()
        self._models: dict[str, tuple[Any, Any, float]] = {}
        self._newsmtsc_worker: _NewsMtscWorker | None = None
        self.active_model: str | None = None

    def options(self) -> list[dict[str, Any]]:
        tuning = self.project_root / "datasets" / "adverse-media-public-tuning-2000"
        khandpur_error = None
        if not ((tuning / "corpus.jsonl").is_file() and (
            tuning / "annotations" / "human.jsonl"
        ).is_file()):
            khandpur_error = "public tuning dataset is missing"
        elif not _modules_available("numpy", "sklearn"):
            khandpur_error = f"scikit-learn is not installed; {ACADEMIC_SETUP_HINT}"
        khandpur_available = khandpur_error is None
        tartu = self.project_root / "third_party" / "ut-ml-adverse-media"
        tartu_error = None
        if not all(
            (tartu / filename).is_file()
            for filename in (
                "adverse_media_training.csv.zip",
                "non_adverse_media_training.csv.zip",
            )
        ):
            tartu_error = f"released Tartu training data is missing; {ACADEMIC_SETUP_HINT}"
        elif not _modules_available("numpy", "sklearn", "spacy", "en_core_web_sm"):
            tartu_error = (
                f"scikit-learn, spaCy, or en_core_web_sm is not installed; {ACADEMIC_SETUP_HINT}"
            )
        tartu_available = tartu_error is None
        newsmtsc_python = _newsmtsc_python(self.project_root)
        newsmtsc_source = self.project_root / "third_party" / "NewsMTSC" / "NewsSentiment"
        newsmtsc_encoder = (
            newsmtsc_source
            / "pretrained_models"
            / os.environ.get("LAYA_NEWSMTSC_PRETRAINED_MODEL", "roberta-base")
        )
        newsmtsc_available = (
            newsmtsc_python is not None and (newsmtsc_encoder / "config.json").is_file()
        )
        return [{
            "id": ACADEMIC_KHANDPUR_ID,
            "label": "Khandpur · Entity-relevance component",
            "family": "academic",
            "category": "decision",
            "deletable": False,
            "runtime_available": khandpur_available,
            "availability_error": khandpur_error,
            "prompt": None,
            "method": (
                "Logistic regression over word n-grams around each mention of the entity, "
                "trained on the public 2,000-article tuning set."
            ),
        }, {
            "id": ACADEMIC_NEWSMTSC_ID,
            "label": "NewsMTSC · GRU-TSC v1 sentiment transfer",
            "family": "academic",
            "category": "decision",
            "deletable": False,
            "runtime_available": newsmtsc_available,
            "availability_error": (
                None
                if newsmtsc_available
                else f"isolated NewsMTSC environment is missing; {ACADEMIC_SETUP_HINT}"
            ),
            "prompt": None,
            "method": (
                "The published GRU-TSC news sentiment model scores the sentence that mentions "
                "the entity; negative sentiment toward the entity is reported as negative."
            ),
        }, {
            "id": ACADEMIC_TARTU_ID,
            "label": "Tartu · TF-IDF + multinomial NB",
            "family": "academic",
            "category": "decision",
            "deletable": False,
            "runtime_available": tartu_available,
            "availability_error": tartu_error,
            "prompt": None,
            "method": (
                "TF-IDF with multinomial naive Bayes trained on the Tartu project's released "
                "adverse-media articles; it classifies the whole article, not the entity."
            ),
        }]

    def activate(self, model_id: str) -> None:
        if model_id not in {
            ACADEMIC_KHANDPUR_ID,
            ACADEMIC_NEWSMTSC_ID,
            ACADEMIC_TARTU_ID,
        }:
            raise ValueError(f"unknown academic model {model_id!r}")
        with self._lock:
            if model_id == ACADEMIC_NEWSMTSC_ID and self._newsmtsc_worker is None:
                python = _newsmtsc_python(self.project_root)
                if python is None:
                    raise RuntimeError("isolated NewsMTSC Python executable is unavailable")
                self._newsmtsc_worker = _NewsMtscWorker(self.project_root, python)
            elif model_id not in self._models:
                self._models[model_id] = (
                    self._fit_khandpur()
                    if model_id == ACADEMIC_KHANDPUR_ID
                    else self._fit_tartu()
                )
            self.active_model = model_id

    def _fit_khandpur(self) -> tuple[Any, Any, float]:
        import numpy as np
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import f1_score
        from sklearn.model_selection import train_test_split

        root = self.project_root / "datasets" / "adverse-media-public-tuning-2000"
        rows = _read_jsonl(root / "corpus.jsonl")
        labels_by_id = {
            row["article_id"]: int(row["label"] == 2)
            for row in _read_jsonl(root / "annotations" / "human.jsonl")
        }
        rows = [row for row in rows if row["article_id"] in labels_by_id]
        labels = np.array([labels_by_id[row["article_id"]] for row in rows])
        contexts = [
            _target_context(str(row["article"]), str(row["entity_name"]))
            for row in rows
        ]
        train, validation = train_test_split(
            np.arange(len(rows)), test_size=0.20, random_state=KHANDPUR_SEED, stratify=labels
        )
        best: tuple[float, float, Any, Any] | None = None
        for regularization in (0.25, 1.0, 4.0):
            vectorizer = CountVectorizer(
                lowercase=True,
                binary=True,
                ngram_range=(1, 2),
                min_df=2,
                max_features=75_000,
            )
            classifier = LogisticRegression(
                C=regularization,
                class_weight="balanced",
                max_iter=2_000,
                random_state=KHANDPUR_SEED,
                solver="liblinear",
            )
            classifier.fit(vectorizer.fit_transform([contexts[index] for index in train]), labels[train])
            scores = classifier.predict_proba(
                vectorizer.transform([contexts[index] for index in validation])
            )[:, 1]
            threshold, validation_f1 = max(
                (
                    [
                        float(threshold),
                        float(
                            f1_score(
                                labels[validation],
                                scores >= threshold,
                                zero_division=0,
                            )
                        ),
                    ]
                    for threshold in np.linspace(0.10, 0.90, 161)
                ),
                key=lambda item: (item[1], -abs(item[0] - 0.5)),
            )
            candidate = (validation_f1, -abs(threshold - 0.5), vectorizer, classifier)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
                best_threshold = threshold
        assert best is not None
        return best[2], best[3], best_threshold

    def _fit_tartu(self) -> tuple[Any, Any, Any, float]:
        import numpy as np
        import spacy
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.naive_bayes import MultinomialNB

        root = self.project_root / "third_party" / "ut-ml-adverse-media"
        examples: list[tuple[str, int]] = []
        for filename in (
            "adverse_media_training.csv.zip",
            "non_adverse_media_training.csv.zip",
        ):
            for row in _read_zip_csv(root / filename):
                label = row.get("label", "").strip()
                if label in {"am", "nam", "random"}:
                    examples.append(
                        (f"{row.get('title', '')} {row.get('article', '')}", int(label == "am"))
                    )
        nlp = spacy.load("en_core_web_sm")
        cleaned = [_clean_tartu_text(text) for text, _ in examples]
        documents = [
            " ".join(token.lemma_ for token in document if not token.is_stop)
            for document in nlp.pipe(cleaned, batch_size=64)
        ]
        vectorizer = TfidfVectorizer(
            max_features=40_000,
            min_df=5,
            max_df=0.5,
            analyzer="word",
            stop_words="english",
            ngram_range=(1, 3),
        )
        classifier = MultinomialNB(alpha=0.3)
        classifier.fit(
            vectorizer.fit_transform(documents),
            np.array([label for _, label in examples], dtype=np.int64),
        )
        return nlp, vectorizer, classifier, 0.5

    def predict(self, article: str, entity_name: str, model_id: str) -> Mapping[str, Any]:
        self.activate(model_id)
        if model_id == ACADEMIC_NEWSMTSC_ID:
            assert self._newsmtsc_worker is not None
            return self._newsmtsc_worker.predict(article, entity_name)
        if model_id == ACADEMIC_TARTU_ID:
            nlp, vectorizer, classifier, threshold = self._models[model_id]
            cleaned = _clean_tartu_text(article)
            document = nlp(cleaned)
            model_input = " ".join(
                token.lemma_ for token in document if not token.is_stop
            )
        else:
            vectorizer, classifier, threshold = self._models[model_id]
            model_input = _target_context(article, entity_name)
        adverse_probability = float(
            classifier.predict_proba(
                vectorizer.transform([model_input])
            )[0, 1]
        )
        choice = "A" if adverse_probability >= threshold else "B"
        confidence = max(adverse_probability, 1.0 - adverse_probability)
        return {
            "answers": {
                "criminal_association": {
                    "choice": choice,
                    "confidence": confidence,
                    "probabilities": {
                        "A": adverse_probability,
                        "B": 1.0 - adverse_probability,
                    },
                }
            },
            "routing": {
                "model": model_id,
                "reason": "explicit academic baseline",
                "decision_threshold": threshold,
                "uses_target_entity": model_id != ACADEMIC_TARTU_ID,
            },
        }

    def close(self) -> None:
        with self._lock:
            if self._newsmtsc_worker is not None:
                self._newsmtsc_worker.close()
                self._newsmtsc_worker = None
            self.active_model = None