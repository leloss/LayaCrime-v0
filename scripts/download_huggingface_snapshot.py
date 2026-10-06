from __future__ import annotations

import argparse
import os
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
    return parser


def main() -> int:
    args = build_parser().parse_args()
    from huggingface_hub import snapshot_download

    options = {
        "repo_id": args.repo_id,
        "revision": args.revision,
        "allow_patterns": args.allow_patterns,
        "token": os.getenv("HF_TOKEN") or None,
    }
    if args.cache_dir is not None:
        options["cache_dir"] = args.cache_dir.resolve()
    else:
        options["local_dir"] = args.local_dir.resolve()

    snapshot = snapshot_download(**options)
    print(f"installed_snapshot={snapshot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
