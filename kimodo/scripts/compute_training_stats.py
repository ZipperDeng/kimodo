"""Compute split Kimodo normalization statistics for a training corpus."""

import argparse
import functools
import json
import math
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf, open_dict
from torch.utils.data import DataLoader

from kimodo.training import (
    BonesSeedDataset,
    MotionDataset,
    RunningMoments,
    build_generic_manifest,
    collate_motion_batch,
    is_seed_dataset_root,
)
from kimodo.training.data import resolve_dataset_root


def arguments():
    parser = argparse.ArgumentParser(description="Compute Kimodo motion normalization statistics")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-root", required=True, help="Prepared SEED or generic dataset directory")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-random-heading", action="store_true")
    return parser.parse_args()


def _save(folder: Path, moments: RunningMoments) -> None:
    folder.mkdir(parents=True, exist_ok=False)
    mean, std = moments.compute()
    np.save(folder / "mean.npy", mean.numpy())
    np.save(folder / "std.npy", std.numpy())


def main():
    args = arguments()
    cfg = OmegaConf.load(args.config)
    seed = int(args.seed if args.seed is not None else cfg.training.seed)
    max_frames = int(args.max_frames if args.max_frames is not None else cfg.training.max_frames)
    torch.manual_seed(seed)
    motion_rep_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model.denoiser.motion_rep, resolve=False))
    with open_dict(motion_rep_cfg):
        motion_rep_cfg.stats_path = None
    motion_rep = instantiate(motion_rep_cfg)
    device = torch.device(args.device)
    motion_rep.skeleton.to(device)

    root = resolve_dataset_root(args.dataset_root)
    seed_layout = is_seed_dataset_root(root)
    if seed_layout:
        if not (root / "text-embeddings.sqlite").is_file():
            raise FileNotFoundError("Missing text-embeddings.sqlite. Run kimodo_cache_text first.")
        dataset = BonesSeedDataset(root, split="train", fps=int(motion_rep.fps))
    else:
        train_manifest = build_generic_manifest(root, "train")
        if train_manifest is None:
            raise FileNotFoundError(f"Expected generic motions under {root / 'motions' / 'train'}")
        build_generic_manifest(root, "val")
        dataset = MotionDataset(train_manifest)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=functools.partial(collate_motion_batch, max_frames=max_frames),
    )

    global_root = RunningMoments(motion_rep.global_root_dim)
    local_root = RunningMoments(motion_rep.local_root_dim)
    body = RunningMoments(motion_rep.body_dim)
    for rotations, roots, lengths, valid, _text, _text_mask in loader:
        rotations, roots = rotations.to(device), roots.to(device)
        lengths, valid = lengths.to(device), valid.to(device)
        with torch.no_grad():
            features = motion_rep(rotations, roots, to_normalize=False, lengths=lengths)
            features = motion_rep.translate_2d_to_zero(features)
            if not args.no_random_heading:
                headings = torch.rand(len(features), device=device) * (2 * math.pi)
                features = motion_rep.rotate_to(features, headings)
            root_features = features[..., motion_rep.root_slice]
            local_features = motion_rep.global_root_to_local_root(root_features, normalized=False, lengths=lengths)
        global_root.update(root_features, valid)
        local_root.update(local_features, valid)
        body.update(features[..., motion_rep.body_slice], valid)

    output = root / "stats" / "motion"
    output.mkdir(parents=True, exist_ok=False)
    _save(output / "global_root", global_root)
    _save(output / "local_root", local_root)
    _save(output / "body", body)
    manifest = {
        "schema_version": 1,
        "format": "seed" if seed_layout else "generic",
        "motion_stats": "stats/motion",
        "text_embeddings": "text-embeddings.sqlite" if seed_layout else "embeddings",
        "splits": ["train", "val"] if (seed_layout or (root / "val.jsonl").is_file()) else ["train"],
    }
    (root / "dataset.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Saved normalization statistics for {global_root.count} valid frames to {output}")


if __name__ == "__main__":
    main()
