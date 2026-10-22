"""Dependency-optional experiment logging for long-running training jobs."""

import json
import os
import platform
import socket
import subprocess
import sys
import time
from collections import deque
from datetime import datetime, timedelta, timezone
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


# Short console names for KimodoLoss components; unknown keys print as-is.
_COMPONENT_LABELS = {
    "smooth_root_pos": "root",
    "global_root_heading": "heading",
    "local_joints_positions": "joints",
    "velocities": "vel",
    "global_rot_data": "rot",
    "foot_contacts": "contacts",
    "fk": "fk",
}


def format_duration(seconds: float) -> str:
    """``1d 02:03:04`` / ``02:03:04`` style duration."""
    seconds = max(int(round(seconds)), 0)
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    clock = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{days}d {clock}" if days else clock


class ProgressReporter:
    """Rank-zero console lines with throughput, elapsed time and an ETA.

    The ETA uses the step rate over the last ``window`` log intervals, so it
    follows phase changes and includes validation and checkpoint time.
    """

    def __init__(self, total_steps: int, start_step: int, global_batch_size: int, window: int = 20, stream=None):
        self.total_steps = int(total_steps)
        self.global_batch_size = int(global_batch_size)
        self.stream = stream or sys.stdout
        self.started = time.perf_counter()
        self.history = deque([(self.started, int(start_step))], maxlen=window + 1)
        self.first_log = True

    def _print(self, text: str) -> None:
        print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {text}", file=self.stream, flush=True)

    def banner(self, lines) -> None:
        for line in lines:
            self._print(line)

    def timing(self, step: int) -> dict:
        """Record a log point and return elapsed/rate/ETA values for it."""
        now = time.perf_counter()
        previous_time, previous_step = self.history[-1]
        interval_steps = max(step - previous_step, 1)
        if self.first_log:
            # The first interval includes DataLoader worker startup and warm-up;
            # start the ETA window after it rather than skew the estimate.
            self.first_log = False
            self.history.clear()
            self.history.append((now, int(step)))
            eta = None
        else:
            self.history.append((now, int(step)))
            oldest_time, oldest_step = self.history[0]
            rate = (step - oldest_step) / max(now - oldest_time, 1e-9)
            eta = (self.total_steps - step) / rate if rate > 0 else None
        return {
            "elapsed_seconds": now - self.started,
            "interval_seconds": now - previous_time,
            "seconds_per_batch_step": (now - previous_time) / interval_steps,
            "samples_per_second": interval_steps * self.global_batch_size / max(now - previous_time, 1e-9),
            "eta_seconds": eta,
        }

    def step(self, step: int, metrics: dict) -> None:
        width = len(f"{self.total_steps:,}")
        eta = metrics["eta_seconds"]
        parts = [
            f"step {step:>{width},}/{self.total_steps:,} {100 * step / self.total_steps:5.1f}% P{metrics['phase']}",
            f"loss {metrics['loss']:.4f}  gnorm {metrics['gradient_norm']:.3g}  lr {metrics['learning_rate']:.2e}",
            f"{metrics['seconds_per_batch_step']:.2f}s/step  {metrics['samples_per_second']:,.0f} samples/s"
            f"  data {100 * metrics['data_wait_fraction']:.0f}%",
        ]
        if "gpu_memory_allocated_gib" in metrics:
            parts.append(
                f"mem {metrics['gpu_memory_allocated_gib']:.1f}/{metrics['gpu_memory_reserved_gib']:.1f}G"
            )
        if eta is None:
            eta_text = "ETA estimating"
        else:
            finish = datetime.now() + timedelta(seconds=eta)
            eta_text = f"ETA {format_duration(eta)} ({finish:%m-%d %H:%M})"
        parts.append(f"{format_duration(metrics['elapsed_seconds'])} elapsed  {eta_text}")
        self._print(" | ".join(parts))
        components = "  ".join(
            f"{_COMPONENT_LABELS.get(name, name)} {value:.4f}" for name, value in metrics["components"].items()
        )
        print(f"{'':22}{components}", file=self.stream, flush=True)

    def validation(self, step: int, metrics: dict, best: float, monitor: str, improved: bool) -> None:
        note = "new best, saved model-best" if improved else f"best {best:.4f}"
        components = "  ".join(
            f"{_COMPONENT_LABELS.get(name, name)} {value:.4f}" for name, value in metrics.items() if name != "loss"
        )
        self._print(f"validation @ step {step:,} | {monitor} {metrics[monitor]:.4f} ({note}) | {components}")

    def message(self, text: str) -> None:
        self._print(text)
