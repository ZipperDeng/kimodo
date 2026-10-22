"""Motion normalization statistics computed by the trainer before the first step."""

import functools
import math
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from .data import collate_motion_batch

STATS_PARTS = ("global_root", "local_root", "body")


class RunningMoments:
    """Accumulate population mean and standard deviation without retaining data."""

    def __init__(self, dimension: int):
        self.count = 0
        self.sum = torch.zeros(dimension, dtype=torch.float64)
        self.sum_squares = torch.zeros(dimension, dtype=torch.float64)

    def update(self, values: torch.Tensor, valid: torch.Tensor) -> None:
        selected = values.detach()[valid].double().cpu()
        if not len(selected):
            return
        self.merge(len(selected), selected.sum(dim=0), selected.square().sum(dim=0))

    def merge(self, count: int, total: torch.Tensor, total_squares: torch.Tensor) -> None:
        """Merge sufficient statistics computed by another process."""
        if not count:
            return
        self.count += int(count)
        self.sum += total.detach().double().cpu()
        self.sum_squares += total_squares.detach().double().cpu()

    def compute(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.count == 0:
            raise ValueError("Cannot compute statistics without valid frames")
        mean = self.sum / self.count
        variance = (self.sum_squares / self.count - mean.square()).clamp_min(0)
        return mean.float(), variance.sqrt().float()


@torch.no_grad()
def compute_motion_stats(
    dataset,
    motion_rep,
    device: torch.device,
    *,
    max_samples: int,
    batch_size: int,
    num_workers: int,
    max_frames: int,
    seed: int,
) -> dict[str, RunningMoments]:
    """Feature moments under the training transform (random crop and heading).

    A fixed random subset of ``max_samples`` rows is split across distributed
    ranks; each rank featurizes its shard on ``device`` and the partial sums are
    all-reduced, so every rank returns identical moments.
    """
    distributed = dist.is_available() and dist.is_initialized()
    rank, world = (dist.get_rank(), dist.get_world_size()) if distributed else (0, 1)
    generator = torch.Generator().manual_seed(seed)
    selected = torch.randperm(len(dataset), generator=generator)[: min(max_samples, len(dataset))]
    shard = Subset(dataset, selected[rank::world].tolist())
    loader = DataLoader(
        shard,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=functools.partial(collate_motion_batch, max_frames=max_frames),
        generator=torch.Generator().manual_seed(seed + rank),
    )
    dims = {
        "global_root": motion_rep.global_root_dim,
        "local_root": motion_rep.local_root_dim,
        "body": motion_rep.body_dim,
    }
    sums = {name: torch.zeros(3, dim, dtype=torch.float64, device=device) for name, dim in dims.items()}
    headings = torch.Generator(device=device).manual_seed(seed + rank)
    progress = tqdm(total=len(selected), desc="Computing motion stats", unit="motion", disable=rank != 0)
    # Random crops in an in-process collate use the global RNG; fork it so the
    # statistics are reproducible and the caller's RNG stream is untouched.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed + rank)
        _accumulate(loader, motion_rep, device, headings, sums, progress, world)
    progress.close()
    moments = {}
    for name, values in sums.items():
        if distributed:
            dist.all_reduce(values)
        moments[name] = RunningMoments(dims[name])
        moments[name].merge(int(values[0, 0].item()), values[1], values[2])
    return moments


def _accumulate(loader, motion_rep, device, headings, sums, progress, world):
    for rotations, roots, lengths, valid, _text, _text_mask, smooth_roots in loader:
        rotations, roots = rotations.to(device), roots.to(device)
        lengths, valid = lengths.to(device), valid.to(device)
        features = motion_rep(
            rotations, roots, to_normalize=False, lengths=lengths, smooth_root_pos=smooth_roots.to(device)
        )
        features = motion_rep.translate_2d_to_zero(features)
        angles = torch.rand(len(features), device=device, generator=headings) * (2 * math.pi)
        features = motion_rep.rotate_to(features, angles)
        root_features = features[..., motion_rep.root_slice]
        parts = {
            "global_root": root_features,
            "local_root": motion_rep.global_root_to_local_root(root_features, normalized=False, lengths=lengths),
            "body": features[..., motion_rep.body_slice],
        }
        for name, values in parts.items():
            selected_values = values[valid].double()
            sums[name][0] += len(selected_values)
            sums[name][1] += selected_values.sum(dim=0)
            sums[name][2] += selected_values.square().sum(dim=0)
        progress.update(len(rotations) * world)


def save_motion_stats(folder: Path, moments: dict[str, RunningMoments]) -> None:
    """Write ``<part>/{mean,std}.npy`` atomically, in the layout ``KimodoMotionRep`` loads."""
    folder = Path(folder)
    temporary = folder.with_name(folder.name + ".tmp")
    for name in STATS_PARTS:
        mean, std = moments[name].compute()
        (temporary / name).mkdir(parents=True, exist_ok=True)
        np.save(temporary / name / "mean.npy", mean.numpy())
        np.save(temporary / name / "std.npy", std.numpy())
    folder.parent.mkdir(parents=True, exist_ok=True)
    temporary.replace(folder)
