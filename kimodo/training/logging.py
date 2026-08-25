"""Dependency-optional experiment logging for long-running training jobs."""

import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from omegaconf import OmegaConf


def _git_value(*args):
    try:
        return subprocess.check_output(
            ["git", *args], stderr=subprocess.DEVNULL, text=True, timeout=5
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def write_run_metadata(output: Path, config, argv=None, extra_metadata=None):
    """Write immutable-enough provenance without leaking environment secrets."""
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "argv": list(sys.argv if argv is None else argv),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_dirty": bool(_git_value("status", "--porcelain")),
    }
    metadata.update(extra_metadata or {})
    temporary = output / "run-metadata.json.tmp"
    temporary.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    temporary.replace(output / "run-metadata.json")
    OmegaConf.save(config, output / "training-config.yaml")


class ExperimentLogger:
    """Fan out scalar events to JSONL and optional TensorBoard/W&B backends."""

    def __init__(self, output: Path, config, resume=False):
        self.output = output
        self.output.mkdir(parents=True, exist_ok=True)
        self.stream = (output / "events.jsonl").open("a" if resume else "w", encoding="utf-8")
        self.tensorboard = None
        self.wandb = None
        logging = config.training.get("logging", {})
        backends = {str(x).lower() for x in logging.get("backends", ["jsonl"])}
        if "tensorboard" in backends:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError as error:
                raise ImportError("TensorBoard logging requested; install kimodo[train]") from error
            self.tensorboard = SummaryWriter(str(output / "tensorboard"), purge_step=None)
        if "wandb" in backends:
            try:
                import wandb
            except ImportError as error:
                raise ImportError("W&B logging requested; install kimodo[train]") from error
            self.wandb = wandb
            run_id_path = output / "wandb-run-id.txt"
            if resume and run_id_path.exists():
                run_id = run_id_path.read_text(encoding="utf-8").strip()
            else:
                run_id = wandb.util.generate_id()
                run_id_path.write_text(run_id, encoding="utf-8")
            wandb.init(
                project=logging.get("project", "kimodo"),
                entity=logging.get("entity"),
                name=logging.get("run_name"),
                group=logging.get("group"),
                tags=list(logging.get("tags", [])),
                dir=str(output),
                id=run_id,
                resume="allow" if resume else None,
                config=OmegaConf.to_container(config, resolve=True),
            )

    def log(self, metrics: dict, batch_step: int, event="train"):
        values = {}
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().item()
            if isinstance(value, (int, float, bool, str)) or value is None:
                values[key] = value
        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "batch_step": int(batch_step),
            **values,
        }
        self.stream.write(json.dumps(record, allow_nan=False) + "\n")
        self.stream.flush()
        scalars = {key: value for key, value in values.items() if isinstance(value, (int, float, bool))}
        if self.tensorboard:
            for key, value in scalars.items():
                self.tensorboard.add_scalar(f"{event}/{key}", value, batch_step)
        if self.wandb:
            self.wandb.log({f"{event}/{key}": value for key, value in scalars.items()}, step=batch_step)

    def close(self):
        if self.stream.closed:
            return
        if self.tensorboard:
            self.tensorboard.close()
        if self.wandb:
            self.wandb.finish()
        self.stream.close()
