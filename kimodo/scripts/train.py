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
    BonesSeedDataset,
    ExponentialMovingAverage,
    ExperimentLogger,
    KimodoLoss,
    MotionDataset,
    capture_rng_state,
    collate_motion_batch,
    is_seed_dataset_root,
    restore_rng_state,
    read_embedding_cache_provenance,
    set_dropout_probability,
    write_run_metadata,
)
from kimodo.training.constraints import sample_constraints
from kimodo.training.data import resolve_dataset_root


def arguments():
    parser = argparse.ArgumentParser(description="Train Kimodo")
    parser.add_argument("--config", required=True, help="Training YAML (see configs/train_soma.yaml)")
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Prepared SEED or generic dataset directory",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", help="Path to a training-state .pt file")
    parser.add_argument("--max-steps", type=int, help="Override curriculum total steps")
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
def evaluate_validation(
    denoiser, loader, text_encoder, motion_rep, diffusion, objective, device, precision, max_batches, seed
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
                if isinstance(text, list):
                    text, text_lengths = text_encoder(text)
                    text = text.to(device)
                    text_lengths = torch.as_tensor(text_lengths, device=device)
                    text_mask = torch.arange(text.shape[1], device=device)[None] < text_lengths[:, None]
                else:
                    text, text_mask = text.to(device), text_mask.to(device)
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
                    totals[key] = totals.get(key, 0.0) + float(value) * len(clean)
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
    optimizer_step,
    micro_step,
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
    model_name = "model.safetensors" if final else ("model-best.safetensors" if best else f"model-{optimizer_step:07d}.safetensors")
    model_path = output / model_name
    model_temporary = model_path.with_suffix(model_path.suffix + ".tmp")
    safetensors_metadata = {"optimizer_step": str(optimizer_step)}
    text_model_path = output / "text-model.json"
    if text_model_path.is_file():
        safetensors_metadata["text_model"] = json.dumps(
            json.loads(text_model_path.read_text(encoding="utf-8")), sort_keys=True
        )
    save_file(state, str(model_temporary), metadata=safetensors_metadata)
    model_temporary.replace(model_path)
    training_state = {
            "schema_version": 1,
            "optimizer_step": optimizer_step,
            "micro_step": micro_step,
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
    # tied to the current optimizer step.
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
    if args.max_steps is not None:
        cfg.training.total_steps = args.max_steps
        cfg.training.phase1_steps = min(cfg.training.phase1_steps, args.max_steps)
    world, rank, device = distributed_setup()
    seed = int(cfg.training.seed) + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    output = Path(args.output).resolve()
    dataset_root = resolve_dataset_root(args.dataset_root)
    seed_layout = is_seed_dataset_root(dataset_root)
    embedding_cache = dataset_root / "text-embeddings.sqlite" if seed_layout else None
    if not args.resume and (output / "training-state.pt").exists():
        raise FileExistsError(
            f"Output already contains a training run: {output}. Pass --resume or choose a new directory."
        )
    cache_provenance = read_embedding_cache_provenance(embedding_cache) if embedding_cache else None
    generic_text_model_path = dataset_root / "text-model.json"
    generic_text_model = (
        json.loads(generic_text_model_path.read_text(encoding="utf-8"))
        if generic_text_model_path.is_file() else None
    )
    if seed_layout and cache_provenance is None:
        raise ValueError(
            "Embedding cache has no text-model metadata. Rebuild it with kimodo_cache_text "
            "so the training run is attributable."
        )
    if not seed_layout and generic_text_model is None:
        raise FileNotFoundError(f"Missing text-model metadata: {generic_text_model_path}. Run kimodo_cache_text first.")
    text_model = (
        cache_provenance
        or generic_text_model
        or text_encoder_provenance_from_config(cfg.training.get("text_encoder"))
    )
    logger = None
    if rank == 0:
        if not args.resume or not (output / "run-metadata.json").exists():
            write_run_metadata(output, cfg, extra_metadata={"text_model": text_model})
            write_text_model_provenance(output, text_model)
        logger = ExperimentLogger(output, cfg, resume=bool(args.resume))
    motion_stats_dir = dataset_root / "stats" / "motion"
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
    diffusion = Diffusion(int(model_cfg.num_base_steps)).to(device)
    objective = KimodoLoss(motion_rep)
    optimizer = AdamAtan2(
        denoiser.parameters(), lr=float(cfg.training.learning_rate),
        weight_decay=float(cfg.training.weight_decay),
    )
    ema = ExponentialMovingAverage(denoiser, decay=float(cfg.training.ema.decay)) if cfg.training.ema.enabled else None

    optimizer_step, micro_step, start_epoch, best_metric = 0, 0, 0, float("inf")
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        denoiser.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        optimizer_step = int(state.get("optimizer_step", state.get("step", 0)))
        micro_step = int(state.get("micro_step", optimizer_step * int(cfg.training.gradient_accumulation)))
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
                resume_seed = seed + optimizer_step * world
                random.seed(resume_seed)
                np.random.seed(resume_seed)
                torch.manual_seed(resume_seed)
    if world > 1:
        denoiser = DistributedDataParallel(
            denoiser,
            device_ids=[device.index] if device.type == "cuda" else None,
        )

    if seed_layout:
        dataset = BonesSeedDataset(
            dataset_root,
            split="train",
            fps=int(motion_rep.fps),
            include_mirrored=bool(cfg.training.include_mirrored),
            temporal_probability=float(cfg.training.temporal_probability),
            embedding_cache=embedding_cache,
            expected_embedding_dim=expected_embedding_dim,
        )
        text_encoder = None
        dataset_root_input = True
        validation_dataset = BonesSeedDataset(
            dataset_root,
            split="val",
            fps=int(motion_rep.fps),
            include_mirrored=bool(cfg.training.include_mirrored),
            temporal_probability=0.0,
            embedding_cache=embedding_cache,
            expected_embedding_dim=expected_embedding_dim,
        )
    else:
        train_manifest = dataset_root / "train.jsonl"
        if not train_manifest.is_file():
            raise FileNotFoundError(f"Missing generated manifest: {train_manifest}. Run kimodo_stats first.")
        dataset = MotionDataset(train_manifest)
        text_encoder = None
        dataset_root_input = False
        val_manifest = dataset_root / "val.jsonl"
        validation_dataset = MotionDataset(val_manifest) if val_manifest.is_file() else None
    if rank == 0 and text_encoder is not None:
        resolved_text_model = getattr(text_encoder, "provenance", text_model)
        write_text_model_provenance(output, resolved_text_model)
        logger.log(
            {
                "preset": resolved_text_model.get("preset"),
                "revision": resolved_text_model.get("revision"),
                "embedding_dim": resolved_text_model.get("embedding_dim"),
            },
            optimizer_step,
            event="text_model",
        )
    sampler = DistributedSampler(dataset, shuffle=True) if world > 1 else None
    collate = functools.partial(collate_motion_batch, max_frames=int(cfg.training.max_frames))
    num_workers = int(cfg.training.num_workers)
    if os.name == "nt" and dataset_root_input:
        # Spawn would duplicate the large metadata/temporal-label index in
        # every Windows worker. Main-process BVH loading is slower but bounded.
        num_workers = 0
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
            num_workers=0 if os.name == "nt" and dataset_root_input else num_workers,
            pin_memory=device.type == "cuda",
            drop_last=False,
            collate_fn=collate,
        )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and cfg.training.precision == "fp16")
    if args.resume and state.get("scaler") is not None:
        scaler.load_state_dict(state["scaler"])
    amp_dtype = torch.bfloat16 if cfg.training.precision == "bf16" else torch.float16
    accumulation = int(cfg.training.gradient_accumulation)
    if accumulation < 1:
        raise ValueError("training.gradient_accumulation must be at least one")
    global_batch = int(cfg.training.batch_size) * world * accumulation
    if rank == 0:
        startup = {
            "global_batch_size": global_batch,
            "world_size": world,
            "accumulation": accumulation,
            "train_samples": len(dataset),
            "validation_samples": len(validation_dataset) if validation_dataset is not None else 0,
            "parameters": sum(parameter.numel() for parameter in denoiser.parameters()),
            "text_model": text_model.get("preset") if text_model else None,
            "text_embedding_dim": text_model.get("embedding_dim") if text_model else None,
        }
        print(json.dumps(startup))
        logger.log(startup, optimizer_step, event="run")
        expected_batch = int(cfg.training.expected_global_batch_size)
        if expected_batch and global_batch != expected_batch:
            print(f"Warning: global batch is {global_batch}; report configuration uses {expected_batch}.")
    epoch = start_epoch
    optimizer.zero_grad(set_to_none=True)
    active_phase = None
    last_log_time = time.perf_counter()
    last_log_step = optimizer_step
    while optimizer_step < int(cfg.training.total_steps):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for rotations, roots, lengths, valid, text, text_mask in loader:
            rotations, roots = rotations.to(device), roots.to(device)
            lengths, valid = lengths.to(device), valid.to(device)
            if isinstance(text, list):
                with torch.no_grad():
                    text, text_lengths = text_encoder(text)
                text = text.to(device)
                text_lengths = torch.as_tensor(text_lengths, device=device)
                text_mask = torch.arange(text.shape[1], device=device)[None] < text_lengths[:, None]
            else:
                text, text_mask = text.to(device), text_mask.to(device)
            with torch.no_grad():
                clean = motion_rep(rotations, roots, to_normalize=False, lengths=lengths)
                clean = motion_rep.translate_2d_to_zero(clean)
                headings = torch.rand(len(clean), device=device) * (2 * math.pi)
                clean = motion_rep.rotate_to(clean, headings)
                clean = motion_rep.normalize(clean)
                first_heading = motion_rep.get_root_heading_angle(motion_rep.unnormalize(clean))[:, 0]
                observed = motion_mask = None
                phase = 1 if optimizer_step < int(cfg.training.phase1_steps) else 2
                if phase != active_phase:
                    dropout = cfg.training.phase1_dropout if phase == 1 else cfg.training.phase2_dropout
                    raw = denoiser.module if isinstance(denoiser, DistributedDataParallel) else denoiser
                    set_dropout_probability(raw, float(dropout))
                    active_phase = phase
                    if rank == 0:
                        print(json.dumps({"optimizer_step": optimizer_step, "phase": phase, "dropout": float(dropout)}))
                if phase == 2:
                    phase2_steps = max(1, int(cfg.training.total_steps) - int(cfg.training.phase1_steps))
                    curriculum_progress = (optimizer_step - int(cfg.training.phase1_steps)) / phase2_steps
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
            completes_update = (micro_step + 1) % accumulation == 0
            sync_context = contextlib.nullcontext()
            if isinstance(denoiser, DistributedDataParallel) and not completes_update:
                sync_context = denoiser.no_sync()
            with sync_context:
                with autocast:
                    pred = denoiser(noisy, valid, text, text_mask, t, first_heading, motion_mask, observed)
                    loss, components = objective(pred, clean, valid)
                    scaled_loss = loss / accumulation
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss at optimizer step {optimizer_step}: {float(loss)}")
                scaler.scale(scaled_loss).backward()
            micro_step += 1
            if completes_update:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(raw.parameters(), float(cfg.training.gradient_clip))
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError(f"Non-finite gradient norm at optimizer step {optimizer_step}")
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1
                if ema is not None and optimizer_step % int(cfg.training.ema.update_every) == 0:
                    ema.update(raw)
                if rank == 0 and optimizer_step % int(cfg.training.log_every) == 0:
                    now = time.perf_counter()
                    elapsed = max(now - last_log_time, 1e-9)
                    steps_since_log = optimizer_step - last_log_step
                    values = {key: float(value.detach()) for key, value in components.items()}
                    metrics = {
                        "loss": float(loss.detach()),
                        **values,
                        "learning_rate": float(optimizer.param_groups[0]["lr"]),
                        "gradient_norm": float(grad_norm),
                        "samples_per_second": steps_since_log * global_batch / elapsed,
                        "seconds_per_step": elapsed / max(steps_since_log, 1),
                        "epoch": epoch,
                        "phase": active_phase,
                        "loss_scale": float(scaler.get_scale()),
                    }
                    if device.type == "cuda":
                        metrics["gpu_memory_allocated_gib"] = torch.cuda.memory_allocated(device) / 2**30
                        metrics["gpu_memory_reserved_gib"] = torch.cuda.memory_reserved(device) / 2**30
                    print(
                        json.dumps({"step": optimizer_step, **{k: round(v, 5) if isinstance(v, float) else v for k, v in metrics.items()}}),
                        flush=True,
                    )
                    logger.log(metrics, optimizer_step)
                    last_log_time, last_log_step = now, optimizer_step
                validation_every = int(validation_cfg.get("every", 0))
                validation_due = validation_loader is not None and validation_every > 0 and optimizer_step % validation_every == 0
                if validation_due:
                    if world > 1:
                        dist.barrier()
                    if rank == 0:
                        raw = denoiser.module if isinstance(denoiser, DistributedDataParallel) else denoiser
                        ema_context = ema.apply_to(raw) if ema is not None else contextlib.nullcontext(raw)
                        with ema_context:
                            validation_metrics = evaluate_validation(
                                raw, validation_loader, text_encoder, motion_rep, diffusion, objective,
                                device, str(cfg.training.precision), int(validation_cfg.get("max_batches", 16)),
                                int(cfg.training.seed) + 100000,
                            )
                        logger.log(validation_metrics, optimizer_step, event="validation")
                        print(json.dumps({"step": optimizer_step, "validation": validation_metrics}), flush=True)
                        monitor_name = str(validation_cfg.get("monitor", "loss"))
                        if monitor_name not in validation_metrics:
                            raise KeyError(
                                f"validation.monitor={monitor_name!r} is not one of {sorted(validation_metrics)}"
                            )
                        monitored = float(validation_metrics[monitor_name])
                        if monitored < best_metric:
                            best_metric = monitored
                            save_checkpoint(
                                output, denoiser, optimizer, scaler, ema, optimizer_step, micro_step,
                                epoch, cfg, best=True, best_metric=best_metric,
                            )
                            logger.log({"metric": best_metric}, optimizer_step, event="best_checkpoint")
                    if world > 1:
                        dist.barrier()
                if rank == 0 and optimizer_step % int(cfg.training.save_every) == 0:
                    path = save_checkpoint(
                        output, denoiser, optimizer, scaler, ema, optimizer_step, micro_step, epoch, cfg,
                        best_metric=best_metric,
                    )
                    prune_periodic_checkpoints(output, int(cfg.training.get("keep_checkpoints", 3)))
                    logger.log({"path": str(path)}, optimizer_step, event="checkpoint")
            if optimizer_step >= int(cfg.training.total_steps):
                break
        epoch += 1
    if rank == 0:
        save_checkpoint(
            output, denoiser, optimizer, scaler, ema, optimizer_step, micro_step, epoch, cfg,
            final=True, best_metric=best_metric,
        )
        logger.close()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
