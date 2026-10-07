from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download a pinned Hugging Face model snapshot outside Git."
    )
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", required=True)
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--cache-dir", type=Path)
    destination.add_argument("--local-dir", type=Path)
    parser.add_argument("--allow-pattern", action="append", dest="allow_patterns")
    parser.add_argument("--required-file", action="append", dest="required_files")
    parser.add_argument("--read-token-stdin", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    from huggingface_hub import snapshot_download

    token = os.getenv("HF_TOKEN") or None
    if args.read_token_stdin:
        token = json.load(sys.stdin).get("token") or None

    options = {
        "repo_id": args.repo_id,
        "revision": args.revision,
        "allow_patterns": args.allow_patterns,
        "token": token,
    }
    if args.cache_dir is not None:
        options["cache_dir"] = args.cache_dir.resolve()
    else:
        options["local_dir"] = args.local_dir.resolve()

    print("phase=downloading", flush=True)
    snapshot = Path(snapshot_download(**options)).resolve()
    missing = [
        relative_path
        for relative_path in (args.required_files or [])
        if not (snapshot / relative_path).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Downloaded snapshot is incomplete; missing: " + ", ".join(missing)
        )
    print(f"installed_snapshot={snapshot}")
    if args.required_files:
        print(f"installed_path={snapshot / args.required_files[0]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
