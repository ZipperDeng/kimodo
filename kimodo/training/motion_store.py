"""Read-only mmap motion stores published by AssetCMS."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from kimodo.geometry import quaternion_to_matrix

MOTION_STORE_FORMAT = "kimodo-motion-mmap-v1"


def read_motion_index(variant_root: str | Path) -> tuple[Path, dict]:
    """Validate ``skeletons/<skeleton>/motions/index.json`` without mapping the data."""
    path = Path(variant_root).expanduser().resolve() / "motions"
    index_path = path / "index.json"
    if not index_path.is_file():
        raise FileNotFoundError(
            f"Missing motion store: {index_path}. Republish the release with an AssetCMS version that packs motions."
        )
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if (
        index.get("schema_version") != 1
        or index.get("format") != MOTION_STORE_FORMAT
        or index.get("dtype") != "float32"
        or index.get("rotation") != "quaternion_wxyz"
    ):
        raise ValueError(f"Unsupported motion store: {index_path}")
    frames, joints = int(index["frames"]), int(index["joints"])
    for name, width in (("rotations", joints * 4), ("roots", 3)):
        file = path / index[name]
        if not file.is_file() or file.stat().st_size != frames * width * 4:
            raise ValueError(f"Incomplete motion store file: {file}")
    if len(index["motions"]) != int(index["count"]):
        raise ValueError(f"Motion store index count mismatch: {index_path}")
    return path, index


class MotionStore:
    """Slice motions from one contiguous float32 mmap without decompression."""

    def __init__(self, variant_root: str | Path):
        self.path, self.index = read_motion_index(variant_root)
        frames, self.joints = int(self.index["frames"]), int(self.index["joints"])
        self.motions = self.index["motions"]
        self.rotations = np.memmap(
            self.path / self.index["rotations"], dtype="<f4", mode="r", shape=(frames, self.joints, 4)
        )
        self.roots = np.memmap(self.path / self.index["roots"], dtype="<f4", mode="r", shape=(frames, 3))

    def get(self, motion_index: int, start: int = 0, end: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``local_rot_mats [T,J,3,3]`` and ``root_positions [T,3]`` for a frame range."""
        motion = self.motions[motion_index]
        end = motion["frames"] if end is None else end
        first, last = motion["offset"] + start, motion["offset"] + end
        quaternions = torch.from_numpy(np.array(self.rotations[first:last]))
        roots = torch.from_numpy(np.array(self.roots[first:last]))
        return quaternion_to_matrix(quaternions), roots

    def close(self):
        for values in (self.rotations, self.roots):
            mmap = getattr(values, "_mmap", None)
            if mmap is not None:
                mmap.close()


__all__ = ["MotionStore", "read_motion_index"]
