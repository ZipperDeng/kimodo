"""Recoverable orchestration around Kimodo's existing benchmark scripts."""

import hashlib
import json
import os
import platform
import re
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from omegaconf import OmegaConf
from kimodo.model.text_encoder_registry import read_checkpoint_text_model


STAGES = ("prepare", "generate", "embed", "evaluate", "summarize", "publish")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def suite_fingerprint(root: Path):
    """Hash lightweight benchmark definitions, deliberately excluding large NPZ files."""
    digest = hashlib.sha256()
    if not root.is_dir():
        return None
    paths = []
    for name in ("meta.json", "seed_motion.json", "seed_constraints.json"):
        paths.extend(root.rglob(name))
    for path in sorted(set(paths)):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


class JobLock:
    def __init__(self, path: Path):
        self.path = path
        self.acquired = False

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({"pid": os.getpid(), "host": socket.gethostname(), "created_at_utc": utc_now()})
        try:
            descriptor = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as error:
            owner = self.path.read_text(encoding="utf-8", errors="replace")
            raise RuntimeError(f"Benchmark job is locked by {owner}") from error
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
        self.acquired = True
        return self

    def __exit__(self, *_):
        if self.acquired:
            self.path.unlink(missing_ok=True)


def _resolve(base: Path, value):
    path = Path(str(value)).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def load_job_config(path: Path):
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    base = path.resolve().parent
    raw["job_dir"] = str(_resolve(base, raw["job_dir"]))
    raw["suite"] = str(_resolve(base, raw["suite"]))
    if raw.get("dataset"):
        raw["dataset"] = str(_resolve(base, raw["dataset"]))
    model = raw.setdefault("model", {})
    if model.get("checkpoint"):
        model["checkpoint"] = str(_resolve(base, model["checkpoint"]))
    return raw


def _flag(command, name, enabled):
    if enabled:
        command.append(name)


def build_commands(config, repo_root: Path):
    python = sys.executable
    suite = config["suite"]
    job_dir = Path(config["job_dir"])
    results = str(job_dir / "results")
    commands = {}
    prepare = config.get("prepare", {})
    if prepare.get("enabled", False):
        commands["prepare"] = [
            python, "-m", "benchmark.create_benchmark", suite,
            "--dataset", config["dataset"], "--workers", str(prepare.get("workers", 1)),
        ]
        _flag(commands["prepare"], "--overwrite", prepare.get("overwrite", False))
    generation = config.get("generation", {})
    model = config.get("model", {})
    model_reference = model.get("checkpoint") or model.get("name")
    if generation.get("enabled", True) and not model_reference:
        raise ValueError("model.checkpoint or model.name is required")
    if generation.get("enabled", True):
        commands["generate"] = [
            python, "-m", "benchmark.generate_eval",
            "--benchmark", suite, "--output", results, "--model", str(model_reference),
            "--batch_size", str(generation.get("batch_size", 1)),
            "--num_workers", str(generation.get("num_workers", 0)),
            "--diffusion_steps", str(generation.get("diffusion_steps", 100)),
        ]
        _flag(commands["generate"], "--postprocess", generation.get("postprocess", False))
        _flag(commands["generate"], "--text_encoder_fp32", generation.get("text_encoder_fp32", False))
        if generation.get("text_model"):
            commands["generate"] += ["--text_encoder", str(generation["text_model"])]
    embedding = config.get("embedding", {})
    if embedding.get("enabled", True):
        commands["embed"] = [
            python, "-m", "benchmark.embed_folder", results,
            "--model", str(embedding.get("model", "tmr-soma-rp")),
        ]
        if embedding.get("device"):
            commands["embed"] += ["--device", str(embedding["device"])]
        _flag(commands["embed"], "--text_encoder_fp32", embedding.get("text_encoder_fp32", True))
        if embedding.get("text_model"):
            commands["embed"] += ["--text_encoder", str(embedding["text_model"])]
    evaluate = config.get("evaluation", {})
    if evaluate.get("enabled", True):
        commands["evaluate"] = [python, "-m", "benchmark.evaluate_folder", results]
        if evaluate.get("device"):
            commands["evaluate"] += ["--device", str(evaluate["device"])]
    summary = config.get("summary", {})
    if summary.get("enabled", True):
        commands["summarize"] = [
            python, "-m", "benchmark.parse_folder", results,
            "--output", str(job_dir / "summary_rows.json"),
            "--format", str(summary.get("format", "terminal")),
        ]
    return commands


def validate_stage(stage, config):
    suite = Path(config["suite"])
    results = Path(config["job_dir"]) / "results"
    expected = len(list(suite.rglob("meta.json")))
    if stage == "prepare":
        seeds = len(list(suite.rglob("seed_motion.json")))
        actual = len(list(suite.rglob("gt_motion.npz")))
        if seeds and actual < seeds:
            raise RuntimeError(f"Prepare incomplete: {actual}/{seeds} ground-truth motions")
        return {"seed_cases": seeds, "ground_truth": actual}
    if expected == 0:
        raise RuntimeError(f"No meta.json cases under benchmark suite: {suite}")
    pattern = {
        "generate": "motion.npz",
        "embed": "motion_embedding.npy",
        "evaluate": "metrics.json",
    }.get(stage)
    if pattern:
        actual = len(list(results.rglob(pattern)))
        if actual != expected:
            raise RuntimeError(f"{stage} incomplete: {actual}/{expected} {pattern} files")
        validation = {"expected_cases": expected, "completed_cases": actual}
        provenance_name = {
            "generate": "generation-text-model.json", "embed": "evaluator-text-model.json"
        }.get(stage)
        if provenance_name and (results / provenance_name).is_file():
            validation["text_model"] = json.loads((results / provenance_name).read_text(encoding="utf-8"))
        return validation
    if stage == "summarize":
        summary = Path(config["job_dir"]) / "summary_rows.json"
        if not summary.is_file():
            raise RuntimeError(f"Missing benchmark summary: {summary}")
        return {"summary": str(summary)}
    return {}


def run_command(stage, command, job_dir: Path, environment=None):
    log_path = job_dir / "logs" / f"{stage}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(f"\n[{utc_now()}] $ {subprocess.list2cmdline(command)}\n")
        stream.flush()
        result = subprocess.run(
            command,
            cwd=str(Path(__file__).resolve().parents[1]),
            env={**os.environ, **{str(key): str(value) for key, value in (environment or {}).items()}},
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command)
    return log_path


def flatten_numbers(value, prefix=""):
    output = {}
    if isinstance(value, dict):
        for key, child in value.items():
            output.update(flatten_numbers(child, f"{prefix}/{key}" if prefix else str(key)))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        output[prefix] = value
    return output


def publish_wandb(config, manifest):
    publish = config.get("publish", {})
    if not publish.get("wandb", {}).get("enabled", False):
        return {"enabled": False}
    try:
        import wandb
    except ImportError as error:
        raise ImportError("W&B publishing requested; install kimodo[train]") from error
    settings = publish["wandb"]
    summary_path = Path(config["job_dir"]) / "summary_rows.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    step = settings.get("step")
    if step is None:
        match = re.search(r"(\d+)(?=\.safetensors$)", str(config.get("model", {}).get("checkpoint", "")))
        step = int(match.group(1)) if match else 0
    run = wandb.init(
        project=settings.get("project", "kimodo"), entity=settings.get("entity"),
        id=settings.get("run_id"), resume="allow" if settings.get("run_id") else None,
        job_type="benchmark", name=settings.get("name"),
    )
    metrics = {f"benchmark/{key}": value for key, value in flatten_numbers(summary).items()}
    metrics["benchmark/job_id"] = manifest["job_id"]
    wandb.log(metrics, step=int(step))
    artifact = wandb.Artifact(f"benchmark-{manifest['job_id']}", type="benchmark")
    artifact.add_file(str(summary_path))
    artifact.add_file(str(Path(config["job_dir"]) / "job-manifest.json"))
    run.log_artifact(artifact)
    wandb.finish()
    return {"enabled": True, "metrics": len(metrics), "step": int(step)}


def create_manifest(config, commands):
    checkpoint = config.get("model", {}).get("checkpoint")
    generation_text_model = {
        "preset": config.get("generation", {}).get("text_model") or "llm2vec",
        "source": "job-config" if config.get("generation", {}).get("text_model") else "implicit-default",
    }
    if checkpoint:
        provenance_path = Path(checkpoint).parent / "text-model.json"
        if provenance_path.is_file():
            generation_text_model = json.loads(provenance_path.read_text(encoding="utf-8"))
        else:
            generation_text_model = read_checkpoint_text_model(checkpoint) or generation_text_model
    evaluator_text_model = {
        "preset": config.get("embedding", {}).get("text_model"),
        "tmr_model": config.get("embedding", {}).get("model", "tmr-soma-rp"),
        "text_encoder_fp32": config.get("embedding", {}).get("text_encoder_fp32", True),
    }
    identity = {
        "schema_version": 1,
        "suite": config["suite"],
        "suite_fingerprint": suite_fingerprint(Path(config["suite"])),
        "checkpoint": checkpoint,
        "checkpoint_sha256": sha256_file(Path(checkpoint)) if checkpoint else None,
        "model_name": config.get("model", {}).get("name"),
        "generation_text_model": generation_text_model,
        "evaluator_text_model": evaluator_text_model,
        "commands": commands,
    }
    encoded = json.dumps(identity, sort_keys=True).encode()
    return {
        **identity,
        "job_id": hashlib.sha256(encoded).hexdigest()[:16],
        "created_at_utc": utc_now(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
    }


def run_job(config, resume=False, dry_run=False, from_stage=None, force_unlock=False):
    repo_root = Path(__file__).resolve().parents[1]
    job_dir = Path(config["job_dir"])
    commands = build_commands(config, repo_root)
    if dry_run:
        return {stage: subprocess.list2cmdline(command) for stage, command in commands.items()}
    job_dir.mkdir(parents=True, exist_ok=True)
    manifest_path, state_path = job_dir / "job-manifest.json", job_dir / "job-state.json"
    proposed = create_manifest(config, commands)
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        comparable = (
            "suite", "suite_fingerprint", "checkpoint", "checkpoint_sha256", "model_name",
            "generation_text_model", "evaluator_text_model", "commands",
        )
        if any(existing.get(key) != proposed.get(key) for key in comparable):
            raise RuntimeError("Job manifest does not match current config/checkpoint/suite")
        if not resume:
            raise FileExistsError(f"Benchmark job already exists; pass --resume: {job_dir}")
        manifest = existing
        state = json.loads(state_path.read_text(encoding="utf-8"))
    else:
        manifest = proposed
        atomic_json(manifest_path, manifest)
        state = {"schema_version": 1, "job_id": manifest["job_id"], "status": "PENDING", "stages": {}}
        atomic_json(state_path, state)
    enabled = [stage for stage in STAGES if stage in commands]
    if config.get("publish", {}).get("wandb", {}).get("enabled", False):
        enabled.append("publish")
    if from_stage:
        if from_stage not in enabled:
            raise ValueError(f"--from-stage must be one of {enabled}")
        start = enabled.index(from_stage)
        for stage in enabled[start:]:
            state["stages"].pop(stage, None)
    lock_path = job_dir / ".benchmark.lock"
    if force_unlock and lock_path.exists():
        previous_lock = lock_path.read_text(encoding="utf-8", errors="replace")
        lock_path.unlink()
        state.setdefault("lock_recovery", []).append({"at_utc": utc_now(), "previous_owner": previous_lock})
        atomic_json(state_path, state)
    with JobLock(lock_path):
        try:
            state["status"] = "RUNNING"
            atomic_json(state_path, state)
            for stage in enabled:
                if state["stages"].get(stage, {}).get("status") == "COMPLETED":
                    continue
                record = state["stages"].setdefault(stage, {})
                record.update({"status": "RUNNING", "started_at_utc": utc_now(), "attempt": record.get("attempt", 0) + 1})
                atomic_json(state_path, state)
                if stage == "publish":
                    validation = publish_wandb(config, manifest)
                    log_path = None
                else:
                    effective_command = list(commands[stage])
                    if from_stage and stage in ("prepare", "generate", "embed") and "--overwrite" not in effective_command:
                        effective_command.append("--overwrite")
                    record["command"] = effective_command
                    atomic_json(state_path, state)
                    log_path = run_command(stage, effective_command, job_dir, config.get("environment"))
                    validation = validate_stage(stage, config)
                record.update({
                    "status": "COMPLETED", "finished_at_utc": utc_now(),
                    "log": str(log_path) if log_path else None, "validation": validation,
                })
                atomic_json(state_path, state)
            state["status"] = "COMPLETED"
            state["finished_at_utc"] = utc_now()
            atomic_json(state_path, state)
        except BaseException as error:
            if "record" in locals() and record.get("status") == "RUNNING":
                record.update({
                    "status": "FAILED", "failed_at_utc": utc_now(),
                    "error": {"type": type(error).__name__, "message": str(error)},
                })
            state["status"] = "FAILED"
            state["failed_at_utc"] = utc_now()
            state["error"] = {"type": type(error).__name__, "message": str(error)}
            atomic_json(state_path, state)
            raise
    return state
