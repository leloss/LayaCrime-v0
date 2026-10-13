from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "third_party" / "NewsMTSC"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from benchmark_newsmtsc_checkpoint import target_sentence  # noqa: E402


def _result(classifier: Any, article: str, entity_name: str) -> dict[str, Any]:
    context = target_sentence(article, entity_name, maximum_chars_per_side=300)
    if context is None:
        adverse_probability = 0.0
        choice = "B"
        reason = "no literal target mention; official fallback"
    else:
        output = classifier.infer(
            text_left=context[0],
            target_mention=context[1],
            text_right=context[2],
            disable_tqdm=True,
        )
        probabilities = {
            row["class_label"]: float(row["class_prob"])
            for row in output
        }
        adverse_probability = probabilities["negative"]
        choice = "A" if max(output, key=lambda row: row["class_prob"])["class_label"] == "negative" else "B"
        reason = "official negative target-sentiment argmax mapping"
    return {
        "answers": {
            "criminal_association": {
                "choice": choice,
                "confidence": max(adverse_probability, 1.0 - adverse_probability),
                "probabilities": {
                    "A": adverse_probability,
                    "B": 1.0 - adverse_probability,
                },
            }
        },
        "routing": {
            "model": "academic-newsmtsc-grutsc-v1",
            "reason": reason,
            "decision_rule": "negative three-class argmax",
            "uses_target_entity": True,
        },
    }


def main() -> None:
    import truststore
    from NewsSentiment.infer import TargetSentimentClassifier, parse_arguments

    truststore.inject_into_ssl()
    options = parse_arguments(override_args=True)
    options.pretrained_model_name = os.environ.get(
        "LAYA_NEWSMTSC_PRETRAINED_MODEL", "roberta-base"
    )
    classifier = TargetSentimentClassifier(opts_from_infer=options)
    print("ready=1", flush=True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get("command") == "shutdown":
                return
            response = _result(
                classifier,
                str(request["article"]),
                str(request["entity_name"]),
            )
        except Exception as exc:
            response = {"error": f"{type(exc).__name__}: {exc}"}
        print("result=" + json.dumps(response, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()