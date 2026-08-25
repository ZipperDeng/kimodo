"""Train the released Kimodo two-stage diffusion denoiser.

Example (one GPU)::

    python -m kimodo.scripts.train --config configs/train_soma.yaml \
        --dataset-root /data/dataset-root --output runs/kimodo-soma

Use ``torchrun --nproc_per_node=N`` for distributed training. Dataset layouts
are documented in ``docs/source/training.md``.
"""

import argparse
import contextlib
import functools
import json
import math
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from hydra.utils import instantiate
from omegaconf import OmegaConf, open_dict
from safetensors.torch import save_file
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from kimodo.model.diffusion import Diffusion
from kimodo.model.text_encoder_registry import (
    TEXT_ENCODER_PRESETS,
    build_text_encoder_config,
    text_encoder_provenance_from_config,
)
from kimodo.training import (
    AdamAtan2,
    ExponentialMovingAverage,
    ExperimentLogger,
    KimodoLoss,
    MotionDataset,
    capture_rng_state,
    collate_motion_batch,
    compute_gradient_accumulation,
    restore_rng_state,
    read_text_vector_metadata,
    skeleton_dataset_root,
    set_dropout_probability,
    write_run_metadata,
    validate_dataset_contract,
)
from kimodo.training.constraints import sample_constraints


def arguments():
    parser = argparse.ArgumentParser(description="Train Kimodo")
    parser.add_argument("--config", required=True, help="Training YAML (see configs/train_soma.yaml)")
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Prepared Kimodo motion/text dataset directory",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", help="Path to a training-state .pt file")
    parser.add_argument(
        "--max-batch-steps", type=int,
        help="Override total global-batch steps",
    )
    parser.add_argument(
        "--text-encoder", choices=sorted(TEXT_ENCODER_PRESETS),
        help="Override the training text encoder preset and model input dimension",
    )
    return parser.parse_args()


def distributed_setup():
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return world, local_rank, device


@torch.no_grad()
def _text_embeddings_to_device(text, text_mask, device):
    """Validate the offline-cache batch contract and move it to the training device."""
    if not isinstance(text, torch.Tensor) or not isinstance(text_mask, torch.Tensor):
        raise TypeError(
            "Training requires tensor text vectors from an AssetCMS mmap snapshot; "
            "publish the selected encoder and then run kimodo_stats."
        )
    return text.to(device), text_mask.to(device)


def evaluate_validation(
    denoiser, loader, motion_rep, diffusion, objective, device, precision, max_batches, seed
):
    """Deterministic text-only denoising validation on a bounded holdout."""
    was_training = denoiser.training
    denoiser.eval()
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    totals, total_items = {}, 0
    cuda_devices = [device.index] if device.type == "cuda" else []
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    try:
        random.seed(seed)
        np.random.seed(seed)
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(seed)
            for batch_index, (rotations, roots, lengths, valid, text, text_mask) in enumerate(loader):
                if batch_index >= max_batches:
                    break
                rotations, roots = rotations.to(device), roots.to(device)
                lengths, valid = lengths.to(device), valid.to(device)
                text, text_mask = _text_embeddings_to_device(text, text_mask, device)
                clean = motion_rep(rotations, roots, to_normalize=False, lengths=lengths)
                clean = motion_rep.translate_2d_to_zero(clean)
                clean = motion_rep.normalize(clean)
                first_heading = motion_rep.get_root_heading_angle(motion_rep.unnormalize(clean))[:, 0]
                timesteps = torch.randint(diffusion.num_base_steps, (len(clean),), device=device)
                noisy = diffusion.q_sample(clean, timesteps)
                autocast = torch.autocast(device.type, dtype=amp_dtype, enabled=device.type == "cuda" and precision != "fp32")
                with autocast:
                    pred = denoiser(noisy, valid, text, text_mask, timesteps, first_heading, None, None)
                    loss, components = objective(pred, clean, valid)
                values = {"loss": loss, **components}
                for key, value in values.items():
                    totals[key] = totals.get(key, 0.0) + float(value.detach()) * len(clean)
                total_items += len(clean)
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        denoiser.train(was_training)
    if not total_items:
        raise ValueError("Validation loader produced no samples")
    return {key: value / total_items for key, value in totals.items()}


def prune_periodic_checkpoints(output: Path, keep: int):
    if keep <= 0:
        return
    checkpoints = sorted(output.glob("model-[0-9][0-9][0-9][0-9][0-9][0-9][0-9].safetensors"))
    for path in checkpoints[:-keep]:
        path.unlink()


def write_text_model_provenance(output: Path, provenance):
    if not provenance:
        return
    temporary = output / "text-model.json.tmp"
    temporary.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output / "text-model.json")


def save_checkpoint(
    output,
    denoiser,
    optimizer,
    scaler,
    ema,
    batch_step,
    micro_batch_step,
    epoch,
    config,
    final=False,
    best=False,
    best_metric=None,
):
    output.mkdir(parents=True, exist_ok=True)
    raw = denoiser.module if isinstance(denoiser, DistributedDataParallel) else denoiser
    inference_state = ema.averaged_state_dict(raw) if ema is not None else raw.state_dict()
    state = {
        f"denoiser.backbone.{key}": value.detach().cpu().contiguous()
        for key, value in inference_state.items()
    }
    model_name = "model.safetensors" if final else ("model-best.safetensors" if best else f"model-{batch_step:07d}.safetensors")
    model_path = output / model_name
    model_temporary = model_path.with_suffix(model_path.suffix + ".tmp")
    safetensors_metadata = {"batch_step": str(batch_step)}
    text_model_path = output / "text-model.json"
    if text_model_path.is_file():
        safetensors_metadata["text_model"] = json.dumps(
            json.loads(text_model_path.read_text(encoding="utf-8")), sort_keys=True
        )
    save_file(state, str(model_temporary), metadata=safetensors_metadata)
    model_temporary.replace(model_path)
    training_state = {
            "schema_version": 1,
            "batch_step": batch_step,
            "micro_batch_step": micro_batch_step,
            "epoch": epoch,
            "model": raw.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "ema": ema.state_dict() if ema is not None else None,
            "rng": capture_rng_state(),
            "best_metric": best_metric,
            "text_model": json.loads(text_model_path.read_text(encoding="utf-8")) if text_model_path.is_file() else None,
    }
    # Best snapshots are inference-only; the resumable latest state remains
    # tied to the current global batch step.
    if not best:
        state_path = output / "training-state.pt"
        state_temporary = state_path.with_suffix(".pt.tmp")
        torch.save(training_state, state_temporary)
        state_temporary.replace(state_path)
    export_config = OmegaConf.create(OmegaConf.to_container(config.model, resolve=False))
    with open_dict(export_config):
        export_config.denoiser.ckpt_path = "${oc.select:checkpoint_dir}/model.safetensors"
        export_config.denoiser.motion_rep.stats_path = "${oc.select:checkpoint_dir}/stats/motion"
    OmegaConf.save(export_config, output / "config.yaml")
    return model_path


def main():
    args = arguments()
    cfg = OmegaConf.load(args.config)
    if args.text_encoder:
        with open_dict(cfg):
            cfg.training.text_encoder = build_text_encoder_config(args.text_encoder)
            cfg.model.denoiser.llm_shape[-1] = TEXT_ENCODER_PRESETS[args.text_encoder]["embedding_dim"]
    if args.max_batch_steps is not None:
        cfg.training.total_batch_steps = args.max_batch_steps
        cfg.training.phase1_batch_steps = min(cfg.training.phase1_batch_steps, args.max_batch_steps)
    world, rank, device = distributed_setup()
    accumulation = compute_gradient_accumulation(
        cfg.training.batch_size,
        world,
        cfg.training.expected_global_batch_size,
    )
    seed = int(cfg.training.seed) + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    output = Path(args.output).resolve()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    if not args.resume and (output / "training-state.pt").exists():
        raise FileExistsError(
            f"Output already contains a training run: {output}. Pass --resume or choose a new directory."
        )
    configured_text_model = text_encoder_provenance_from_config(cfg.training.text_encoder)
    text_encoder_name = args.text_encoder or configured_text_model.get("preset", "llm2vec")
    _index_metadata, text_model = read_text_vector_metadata(dataset_root, text_encoder_name)
    logger = None
    if rank == 0:
        if not args.resume or not (output / "run-metadata.json").exists():
            write_run_metadata(
                output,
                cfg,
                extra_metadata={
                    "text_model": text_model,
                    "runtime_batching": {
                        "per_device_batch_size": int(cfg.training.batch_size),
                        "world_size": world,
                        "global_micro_batch_size": int(cfg.training.batch_size) * world,
                        "gradient_accumulation": accumulation,
                        "global_batch_size": int(cfg.training.expected_global_batch_size),
                    },
                },
            )
            write_text_model_provenance(output, text_model)
        logger = ExperimentLogger(output, cfg, resume=bool(args.resume))
    # Instantiate the configured skeleton name without loading model weights so
    # all derived paths select one variant below the shared dataset root.
    skeleton_probe = instantiate(cfg.model.denoiser.motion_rep.skeleton)
    variant_root = skeleton_dataset_root(dataset_root, skeleton_probe.name)
    motion_stats_dir = variant_root / "stats" / "motion"
    if not motion_stats_dir.is_dir():
        raise FileNotFoundError(f"Missing motion statistics: {motion_stats_dir}. Run kimodo_stats first.")
    if rank == 0:
        output_stats = output / "stats" / "motion"
        if motion_stats_dir.is_dir() and not output_stats.exists():
            shutil.copytree(motion_stats_dir, output_stats)
    if world > 1:
        dist.barrier()
    model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=False))
    with open_dict(model_cfg):
        model_cfg.denoiser.ckpt_path = None
        model_cfg.denoiser.motion_rep.stats_path = str(motion_stats_dir)
    OmegaConf.resolve(model_cfg)
    expected_embedding_dim = int(model_cfg.denoiser.llm_shape[-1])
    if text_model and text_model.get("embedding_dim") is not None:
        cached_embedding_dim = int(text_model["embedding_dim"])
        if cached_embedding_dim != expected_embedding_dim:
            raise ValueError(
                f"Dataset text embeddings have dimension {cached_embedding_dim}, "
                f"but the denoiser expects {expected_embedding_dim}."
            )
    denoiser = instantiate(model_cfg.denoiser).to(device)
    motion_rep = denoiser.motion_rep
    validate_dataset_contract(dataset_root, motion_rep.skeleton, motion_rep.fps)
    diffusion = Diffusion(int(model_cfg.num_base_steps)).to(device)
    objective = KimodoLoss(motion_rep)
    optimizer = AdamAtan2(
        denoiser.parameters(), lr=float(cfg.training.learning_rate),
        weight_decay=float(cfg.training.weight_decay),
    )
    ema = ExponentialMovingAverage(denoiser, decay=float(cfg.training.ema.decay)) if cfg.training.ema.enabled else None

    batch_step, micro_batch_step, start_epoch, best_metric = 0, 0, 0, float("inf")
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        denoiser.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        batch_step = int(state["batch_step"])
        # Checkpoints are saved only at accumulation boundaries. A changed
        # WORLD_SIZE therefore starts a fresh accumulation window safely.
        micro_batch_step = batch_step * accumulation
        start_epoch = int(state.get("epoch", 0))
        best_metric = float(state.get("best_metric", float("inf")))
        if state.get("ema") is not None and ema is not None:
            ema.load_state_dict(state["ema"], denoiser)
        if state.get("rng") is not None:
            if world == 1:
                restore_rng_state(state["rng"])
            else:
                # A single rank-zero checkpoint cannot contain every rank's
                # RNG stream. Re-seed ranks independently to avoid correlated
                # augmentation after distributed resume.
                resume_seed = seed + batch_step * world
                random.seed(resume_seed)
                np.random.seed(resume_seed)
                torch.manual_seed(resume_seed)
    if world > 1:
        denoiser = DistributedDataParallel(
            denoiser,
            device_ids=[device.index] if device.type == "cuda" else None,
        )

    train_manifest = variant_root / "train.jsonl"
    if not train_manifest.is_file():
        raise FileNotFoundError(f"Missing generated manifest: {train_manifest}. Run kimodo_stats first.")
    dataset = MotionDataset(
        train_manifest,
        text_encoder=text_encoder_name,
        expected_joints=motion_rep.nbjoints,
    )
    val_manifest = variant_root / "val.jsonl"
    validation_dataset = (
        MotionDataset(
            val_manifest,
            text_encoder=text_encoder_name,
            expected_joints=motion_rep.nbjoints,
        )
        if val_manifest.is_file() else None
    )
    sampler = DistributedSampler(dataset, shuffle=True) if world > 1 else None
    collate = functools.partial(collate_motion_batch, max_frames=int(cfg.training.max_frames))
    num_workers = int(cfg.training.num_workers)
    loader = DataLoader(
        dataset, batch_size=int(cfg.training.batch_size), shuffle=sampler is None, sampler=sampler,
        num_workers=num_workers, pin_memory=device.type == "cuda", drop_last=True,
        persistent_workers=num_workers > 0, collate_fn=collate,
    )
    validation_loader = None
    validation_cfg = cfg.training.get("validation", {})
    if validation_cfg.get("enabled", False) and validation_dataset is not None and len(validation_dataset):
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=int(validation_cfg.get("batch_size", cfg.training.batch_size)),
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
            drop_last=False,
            collate_fn=collate,
        )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and cfg.training.precision == "fp16")
    if args.resume and state.get("scaler") is not None:
        scaler.load_state_dict(state["scaler"])
    amp_dtype = torch.bfloat16 if cfg.training.precision == "bf16" else torch.float16
    global_batch = int(cfg.training.batch_size) * world * accumulation
    if rank == 0:
        startup = {
            "global_batch_size": global_batch,
            "global_micro_batch_size": int(cfg.training.batch_size) * world,
            "world_size": world,
            "gradient_accumulation": accumulation,
            "train_samples": len(dataset),
            "validation_samples": len(validation_dataset) if validation_dataset is not None else 0,
            "parameters": sum(parameter.numel() for parameter in denoiser.parameters()),
            "text_model": text_model.get("preset") if text_model else None,
            "text_embedding_dim": text_model.get("embedding_dim") if text_model else None,
        }
        print(json.dumps(startup))
        logger.log(startup, batch_step, event="run")
    epoch = start_epoch
    optimizer.zero_grad(set_to_none=True)
    active_phase = None
    last_log_time = time.perf_counter()
    last_log_step = batch_step
    while batch_step < int(cfg.training.total_batch_steps):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for rotations, roots, lengths, valid, text, text_mask in loader:
            rotations, roots = rotations.to(device), roots.to(device)
            lengths, valid = lengths.to(device), valid.to(device)
            text, text_mask = _text_embeddings_to_device(text, text_mask, device)
            with torch.no_grad():
                clean = motion_rep(rotations, roots, to_normalize=False, lengths=lengths)
                clean = motion_rep.translate_2d_to_zero(clean)
                headings = torch.rand(len(clean), device=device) * (2 * math.pi)
                clean = motion_rep.rotate_to(clean, headings)
                clean = motion_rep.normalize(clean)
                first_heading = motion_rep.get_root_heading_angle(motion_rep.unnormalize(clean))[:, 0]
                observed = motion_mask = None
                phase = 1 if batch_step < int(cfg.training.phase1_batch_steps) else 2
                if phase != active_phase:
                    dropout = cfg.training.phase1_dropout if phase == 1 else cfg.training.phase2_dropout
                    raw = denoiser.module if isinstance(denoiser, DistributedDataParallel) else denoiser
                    set_dropout_probability(raw, float(dropout))
                    active_phase = phase
                    if rank == 0:
                        print(json.dumps({"batch_step": batch_step, "phase": phase, "dropout": float(dropout)}))
                if phase == 2:
                    phase2_steps = max(
                        1,
                        int(cfg.training.total_batch_steps) - int(cfg.training.phase1_batch_steps),
                    )
                    curriculum_progress = (
                        batch_step - int(cfg.training.phase1_batch_steps)
                    ) / phase2_steps
                    observed, motion_mask = sample_constraints(
                        clean,
                        lengths,
                        motion_rep,
                        curriculum_progress=curriculum_progress,
                        max_sparse_keyframes=int(cfg.training.constraints.max_sparse_keyframes),
                        no_constraint_probability=float(cfg.training.constraints.no_constraint_probability),
                        mixed_constraint_probability=float(cfg.training.constraints.mixed_constraint_probability),
                    )
                    motion_mask = motion_mask.to(clean.dtype)
                drop_text = torch.rand(len(text), device=device) < float(cfg.training.text_dropout)
                text[drop_text] = 0
                text_mask[drop_text] = False
                t = torch.randint(diffusion.num_base_steps, (len(clean),), device=device)
                noisy = diffusion.q_sample(clean, t)

            raw = denoiser.module if isinstance(denoiser, DistributedDataParallel) else denoiser
            autocast = torch.autocast(
                device.type,
                dtype=amp_dtype,
                enabled=device.type == "cuda" and cfg.training.precision != "fp32",
            )
            completes_update = (micro_batch_step + 1) % accumulation == 0
            sync_context = contextlib.nullcontext()
            if isinstance(denoiser, DistributedDataParallel) and not completes_update:
                sync_context = denoiser.no_sync()
            with sync_context:
                with autocast:
                    pred = denoiser(noisy, valid, text, text_mask, t, first_heading, motion_mask, observed)
                    loss, components = objective(pred, clean, valid)
                    scaled_loss = loss / accumulation
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss at batch step {batch_step}: {float(loss)}")
                scaler.scale(scaled_loss).backward()
            micro_batch_step += 1
            if completes_update:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(raw.parameters(), float(cfg.training.gradient_clip))
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError(f"Non-finite gradient norm at batch step {batch_step}")
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                batch_step += 1
                if ema is not None and batch_step % int(cfg.training.ema.update_every_batch_steps) == 0:
                    ema.update(raw)
                if rank == 0 and batch_step % int(cfg.training.log_every_batch_steps) == 0:
                    now = time.perf_counter()
                    elapsed = max(now - last_log_time, 1e-9)
                    steps_since_log = batch_step - last_log_step
                    values = {key: float(value.detach()) for key, value in components.items()}
                    metrics = {
                        "loss": float(loss.detach()),
                        **values,
                        "learning_rate": float(optimizer.param_groups[0]["lr"]),
                        "gradient_norm": float(grad_norm),
                        "samples_per_second": steps_since_log * global_batch / elapsed,
                        "seconds_per_batch_step": elapsed / max(steps_since_log, 1),
                        "epoch": epoch,
                        "phase": active_phase,
                        "loss_scale": float(scaler.get_scale()),
                    }
                    if device.type == "cuda":
                        metrics["gpu_memory_allocated_gib"] = torch.cuda.memory_allocated(device) / 2**30
                        metrics["gpu_memory_reserved_gib"] = torch.cuda.memory_reserved(device) / 2**30
                    print(
                        json.dumps({"batch_step": batch_step, **{k: round(v, 5) if isinstance(v, float) else v for k, v in metrics.items()}}),
                        flush=True,
                    )
                    logger.log(metrics, batch_step)
                    last_log_time, last_log_step = now, batch_step
                validation_every = int(validation_cfg.get("every_batch_steps", 0))
                validation_due = validation_loader is not None and validation_every > 0 and batch_step % validation_every == 0
                if validation_due:
                    if world > 1:
                        dist.barrier()
                    if rank == 0:
                        raw = denoiser.module if isinstance(denoiser, DistributedDataParallel) else denoiser
                        ema_context = ema.apply_to(raw) if ema is not None else contextlib.nullcontext(raw)
                        with ema_context:
                            validation_metrics = evaluate_validation(
                                raw, validation_loader, motion_rep, diffusion, objective,
                                device, str(cfg.training.precision), int(validation_cfg.get("max_batches", 16)),
                                int(cfg.training.seed) + 100000,
                            )
                        logger.log(validation_metrics, batch_step, event="validation")
                        print(json.dumps({"batch_step": batch_step, "validation": validation_metrics}), flush=True)
                        monitor_name = str(validation_cfg.get("monitor", "loss"))
                        if monitor_name not in validation_metrics:
                            raise KeyError(
                                f"validation.monitor={monitor_name!r} is not one of {sorted(validation_metrics)}"
                            )
                        monitored = float(validation_metrics[monitor_name])
                        if monitored < best_metric:
                            best_metric = monitored
                            save_checkpoint(
                                output, denoiser, optimizer, scaler, ema, batch_step, micro_batch_step,
                                epoch, cfg, best=True, best_metric=best_metric,
                            )
                            logger.log({"metric": best_metric}, batch_step, event="best_checkpoint")
                    if world > 1:
                        dist.barrier()
                if rank == 0 and batch_step % int(cfg.training.save_every_batch_steps) == 0:
                    path = save_checkpoint(
                        output, denoiser, optimizer, scaler, ema, batch_step, micro_batch_step, epoch, cfg,
                        best_metric=best_metric,
                    )
                    prune_periodic_checkpoints(output, int(cfg.training.get("keep_checkpoints", 3)))
                    logger.log({"path": str(path)}, batch_step, event="checkpoint")
            if batch_step >= int(cfg.training.total_batch_steps):
                break
        epoch += 1
    if rank == 0:
        save_checkpoint(
            output, denoiser, optimizer, scaler, ema, batch_step, micro_batch_step, epoch, cfg,
            final=True, best_metric=best_metric,
        )
        logger.close()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
