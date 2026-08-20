"""CLI for recoverable Kimodo benchmark jobs."""

import argparse
import json
from pathlib import Path

from kimodo.benchmark_job import run_job, load_job_config


def main():
    parser = argparse.ArgumentParser(description="Run or resume an auditable Kimodo benchmark pipeline")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--from-stage", choices=("prepare", "generate", "embed", "evaluate", "summarize", "publish"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--force-unlock", action="store_true",
        help="Remove a stale job lock after independently confirming no worker is active",
    )
    args = parser.parse_args()
    result = run_job(
        load_job_config(args.config.resolve()), resume=args.resume,
        dry_run=args.dry_run, from_stage=args.from_stage, force_unlock=args.force_unlock,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
