"""Compute split Kimodo normalization statistics for a training corpus."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf, open_dict
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from kimodo.training import (
    MotionDataset,
    RunningMoments,
    build_generic_manifest,
    collate_motion_batch,
    skeleton_dataset_root,
    validate_dataset_contract,
)


def arguments():
    parser = argparse.ArgumentParser(description="Compute Kimodo motion normalization statistics")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-root", required=True, help="Prepared Kimodo motion/text dataset directory")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--no-random-heading", action="store_true")
    return parser.parse_args()


def _save(folder: Path, moments: RunningMoments) -> None:
    folder.mkdir(parents=True, exist_ok=False)
    mean, std = moments.compute()
    np.save(folder / "mean.npy", mean.numpy())
    np.save(folder / "std.npy", std.numpy())


def _worker_init(_worker_id: int) -> None:
    # Batches are already parallel across processes; nested Torch thread pools
    # would oversubscribe the host.
    torch.set_num_threads(1)


def _partial_moments(values: torch.Tensor, valid: torch.Tensor):
    selected = values.detach()[valid].double().cpu()
    return len(selected), selected.sum(dim=0), selected.square().sum(dim=0)


class _StatsCollator:
    """Compute CPU feature statistics inside DataLoader worker processes."""

    def __init__(self, motion_rep_cfg, max_frames: int, random_heading: bool):
        self.motion_rep_cfg = motion_rep_cfg
        self.max_frames = max_frames
        self.random_heading = random_heading
        self.motion_rep = None

    def __call__(self, samples):
        if self.motion_rep is None:
            self.motion_rep = instantiate(self.motion_rep_cfg)
            self.motion_rep.skeleton.to("cpu")
        rotations, roots, lengths, valid, _text, _text_mask = collate_motion_batch(
            samples, max_frames=self.max_frames
        )
        with torch.no_grad():
            features = self.motion_rep(rotations, roots, to_normalize=False, lengths=lengths)
            features = self.motion_rep.translate_2d_to_zero(features)
            if self.random_heading:
                headings = torch.rand(len(features)) * (2 * math.pi)
                features = self.motion_rep.rotate_to(features, headings)
            root_features = features[..., self.motion_rep.root_slice]
            local_features = self.motion_rep.global_root_to_local_root(
                root_features, normalized=False, lengths=lengths
            )
        return (
            len(samples),
            _partial_moments(root_features, valid),
            _partial_moments(local_features, valid),
            _partial_moments(features[..., self.motion_rep.body_slice], valid),
        )


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
    motion_rep.skeleton.to("cpu")

    root = Path(args.dataset_root).expanduser().resolve()
    variant_root = skeleton_dataset_root(root, motion_rep.skeleton.name)
    validate_dataset_contract(root, motion_rep.skeleton, motion_rep.fps)
    train_manifest = build_generic_manifest(root, "train", motion_rep.skeleton.name)
    if train_manifest is None:
        raise FileNotFoundError(f"Expected prepared motions under {variant_root / 'motions' / 'train'}")
    build_generic_manifest(root, "val", motion_rep.skeleton.name)
    dataset = MotionDataset(
        train_manifest,
        load_text=False,
        motion_cache_size=8,
        expected_joints=motion_rep.nbjoints,
    )
    global_root = RunningMoments(motion_rep.global_root_dim)
    local_root = RunningMoments(motion_rep.local_root_dim)
    body = RunningMoments(motion_rep.body_dim)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=_StatsCollator(motion_rep_cfg, max_frames, not args.no_random_heading),
        worker_init_fn=_worker_init if args.num_workers else None,
    )

    with tqdm(total=len(dataset), desc="Computing stats", unit="motion", dynamic_ncols=True) as progress:
        for batch_count, global_partial, local_partial, body_partial in loader:
            global_root.merge(*global_partial)
            local_root.merge(*local_partial)
            body.merge(*body_partial)
            progress.update(batch_count)
            progress.set_postfix(valid_frames=global_root.count, refresh=False)

    output = variant_root / "stats" / "motion"
    output.mkdir(parents=True, exist_ok=False)
    _save(output / "global_root", global_root)
    _save(output / "local_root", local_root)
    _save(output / "body", body)
    manifest = {
        "schema_version": 1,
        "format": "kimodo-motion-text-v1",
        "skeleton": motion_rep.skeleton.name,
        "joint_count": motion_rep.nbjoints,
        "fps": motion_rep.fps,
        "motion_stats": "stats/motion",
        "text_vectors": "../../motion-text-index.json",
        "splits": ["train", "val"] if (variant_root / "val.jsonl").is_file() else ["train"],
    }
    (variant_root / "dataset.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Saved normalization statistics for {global_root.count} valid frames to {output}")


if __name__ == "__main__":
    main()
