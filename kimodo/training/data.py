"""Manifest based dataset used by the Kimodo training entry point.

Kimodo consumes only self-contained AssetCMS releases. Each
``skeletons/<skeleton>/<split>.jsonl`` row selects a motion by ``motion_index``
in the skeleton's float32 mmap motion store (optionally a frame slice of it)
and one or more content-addressed text vectors in the release's float16 mmap
snapshot. The frozen encoder never runs inside Kimodo training.
"""

import json
import random
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset

from kimodo.motion_rep.smooth_root import get_smooth_root_pos

from .motion_store import MotionStore, read_motion_index
from .text_vectors import TextVectorStore, read_snapshot_metadata, read_text_vector_metadata


def skeleton_dataset_root(root: Path, skeleton_name: str) -> Path:
    """Return the skeleton-specific directory below a release root."""
    return Path(root).expanduser().resolve() / "skeletons" / str(skeleton_name)


def read_release_manifest(root: Path) -> dict:
    """Return the AssetCMS release manifest, rejecting any other directory layout."""
    path = Path(root).expanduser().resolve() / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    if "release" not in manifest:
        raise FileNotFoundError(
            f"{path.parent} is not an AssetCMS release; pass library/releases/<release> as --dataset-root"
        )
    return manifest


def validate_dataset_contract(root: Path, skeleton, fps: float) -> dict:
    """Validate release and skeleton metadata against the motion representation."""
    root = Path(root).expanduser().resolve()
    release = read_release_manifest(root)
    variant_root = skeleton_dataset_root(root, skeleton.name)
    selected = (release.get("selection") or {}).get("skeleton")
    if selected and selected != skeleton.name:
        raise ValueError(
            f"Release {release['release']!r} was published for skeleton {selected!r}, "
            f"not configured skeleton {skeleton.name!r}"
        )
    if not variant_root.is_dir():
        raise FileNotFoundError(
            f"Release {release['release']!r} contains no {skeleton.name!r} variant: {variant_root}"
        )
    metadata = {}
    for path in (root / "source.json", variant_root / "source.json"):
        if path.is_file():
            metadata.update(json.loads(path.read_text(encoding="utf-8")))
    expected = {
        "skeleton": str(skeleton.name),
        "joint_count": int(skeleton.nbjoints),
        "fps": float(fps),
    }
    for key, value in expected.items():
        if key not in metadata:
            continue
        actual = metadata[key]
        matches = abs(float(actual) - value) < 1e-3 if key == "fps" else actual == value
        if not matches:
            raise ValueError(
                f"Dataset {key}={actual!r} is incompatible with configured {key}={value!r}"
            )
    return metadata


class MotionDataset(Dataset):
    def __init__(
        self,
        manifest: str,
        text_encoder: str = "llm2vec",
        load_text: bool = True,
        expected_joints: Optional[int] = None,
    ):
        self.manifest = Path(manifest).resolve()
        self.load_text = load_text
        variant_root = self.manifest.parent
        if variant_root.parent.name != "skeletons":
            raise ValueError(f"Manifest must be under <dataset-root>/skeletons/<skeleton>: {self.manifest}")
        self.variant_root = variant_root
        self.dataset_root = variant_root.parent.parent
        self.text_encoder = text_encoder
        self._text_vectors = None
        self._motions = None
        with self.manifest.open(encoding="utf-8") as stream:
            self.items = [json.loads(line) for line in stream if line.strip()]
        if not self.items:
            raise ValueError(f"Empty manifest: {self.manifest}")
        # Validate every row against the store index before model construction
        # and worker startup, so malformed releases fail immediately.
        _, index = read_motion_index(variant_root)
        if expected_joints is not None and int(index["joints"]) != expected_joints:
            raise ValueError(
                f"Motion store {variant_root} has {index['joints']} joints; "
                f"configured skeleton requires {expected_joints}"
            )
        motions = index["motions"]
        for row, item in enumerate(self.items, 1):
            motion_index = item.get("motion_index")
            if not isinstance(motion_index, int) or not 0 <= motion_index < len(motions):
                raise ValueError(
                    f"Manifest {self.manifest} row {row} has no valid motion_index; "
                    "republish the release with an AssetCMS version that packs motions."
                )
            start, end = int(item.get("start_frame", 0)), int(item.get("end_frame", motions[motion_index]["frames"]))
            if not 0 <= start < end <= motions[motion_index]["frames"]:
                raise ValueError(f"Manifest {self.manifest} row {row} has invalid frame range [{start}, {end})")
        if self.load_text:
            missing = [index + 1 for index, item in enumerate(self.items) if not item.get("text_vector_keys")]
            if missing:
                raise ValueError(
                    f"Manifest {self.manifest} has no text_vector_keys at row {missing[0]}; train from an AssetCMS release."
                )
            read_text_vector_metadata(self.dataset_root, self.text_encoder)
            read_snapshot_metadata(self.dataset_root, self.text_encoder)

    def __len__(self):
        return len(self.items)

    def _vector_store(self):
        if self._text_vectors is None:
            self._text_vectors = TextVectorStore(self.dataset_root, self.text_encoder)
        return self._text_vectors

    def _motion_store(self):
        if self._motions is None:
            self._motions = MotionStore(self.variant_root)
        return self._motions

    def __getstate__(self):
        # Each DataLoader worker maps the stores itself; the OS page cache is shared.
        state = self.__dict__.copy()
        state["_text_vectors"] = None
        state["_motions"] = None
        return state

    def close(self):
        for name in ("_text_vectors", "_motions"):
            store = getattr(self, name, None)
            if store is not None:
                store.close()
                setattr(self, name, None)

    def __del__(self):
        self.close()

    def __getitem__(self, index):
        item = self.items[index]
        rotations, roots = self._motion_store().get(
            item["motion_index"], int(item.get("start_frame", 0)), item.get("end_frame")
        )
        sample = {
            "local_rot_mats": rotations,
            "root_positions": roots,
            "name": item.get("name", f"motion-{item['motion_index']}"),
        }
        if self.load_text:
            key = random.choice(item["text_vector_keys"])
            embedded = self._vector_store().get(key)
            sample["text_feat"] = torch.from_numpy(embedded)
        return sample


def collate_motion_batch(samples, max_frames=300):
    """Randomly crop and pad a variable-length batch.

    Also returns the smoothed root trajectory of each cropped clip. The smoother
    is CPU-bound, so computing it here moves it into DataLoader workers, in
    parallel with the training step.
    """
    lengths = torch.tensor([min(len(x["root_positions"]), max_frames) for x in samples])
    if torch.any(lengths < 2):
        raise ValueError("Kimodo training clips must contain at least two frames")
    batch, max_len = len(samples), int(lengths.max())
    joints = samples[0]["local_rot_mats"].shape[1]
    # Pad with identity rotations, not zeros: padding is masked from the loss,
    # but zero matrices become zero 6D vectors that turn into NaN when features
    # are re-orthonormalized (e.g. by the random-heading rotate_to), and NaN in
    # padding still propagates through attention.
    rotations = torch.eye(3).expand(batch, max_len, joints, 3, 3).clone()
    roots = torch.zeros(batch, max_len, 3)
    embedded = "text_feat" in samples[0]
    if embedded:
        text_len = torch.tensor([len(x["text_feat"]) for x in samples])
        max_text = int(text_len.max())
        text_dim = samples[0]["text_feat"].shape[-1]
        text = torch.zeros(batch, max_text, text_dim)
        text_mask = torch.arange(max_text)[None] < text_len[:, None]
    else:
        text = None
        text_mask = None
    for i, sample in enumerate(samples):
        n = int(lengths[i])
        available = len(sample["root_positions"])
        start = 0 if available == n else int(torch.randint(available - n + 1, ()).item())
        rotations[i, :n] = sample["local_rot_mats"][start : start + n]
        roots[i, :n] = sample["root_positions"][start : start + n]
        if embedded:
            text[i, : text_len[i]] = sample["text_feat"]
    frame_mask = torch.arange(max_len)[None] < lengths[:, None]
    smooth_roots = get_smooth_root_pos(roots, lengths)
    return rotations, roots, lengths, frame_mask, text, text_mask, smooth_roots
