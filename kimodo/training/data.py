"""Manifest based dataset used by the Kimodo training entry point.

Each JSONL row points at a motion ``.npz`` and one or more content-addressed
text vectors in an AssetCMS-published mmap snapshot. The frozen encoder never
runs inside Kimodo training.
"""

import hashlib
import json
import random
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .text_vectors import TextVectorStore, read_snapshot_metadata, read_text_vector_metadata


def skeleton_dataset_root(root: Path, skeleton_name: str) -> Path:
    """Return the skeleton-specific derived-data directory below a shared dataset root."""
    return Path(root).expanduser().resolve() / "skeletons" / str(skeleton_name)


def validate_dataset_contract(root: Path, skeleton, fps: float) -> dict:
    """Validate shared and skeleton-specific metadata against the motion representation."""
    root = Path(root).expanduser().resolve()
    metadata = {}
    variant_root = skeleton_dataset_root(root, skeleton.name)
    for path in (root / "source.json", variant_root / "source.json", variant_root / "dataset.json"):
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


def text_embedding_key(text: str) -> str:
    """Return the stable content key used by the text-embedding cache."""
    canonical = text.strip()
    if not canonical:
        raise ValueError("Text description is empty")
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _write_manifest(variant_root: Path, split: str, rows) -> Path:
    manifest = variant_root / f"{split}.jsonl"
    temporary = manifest.with_suffix(".jsonl.tmp")
    temporary.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    temporary.replace(manifest)
    return manifest


def build_generic_manifest(root: Path, split: str, skeleton_name: str) -> Optional[Path]:
    """Build a motion manifest whose text values reference the SQLite cache."""
    root = Path(root).expanduser().resolve()
    variant_root = skeleton_dataset_root(root, skeleton_name)
    motion_root = variant_root / "motions" / split
    if not motion_root.is_dir():
        return None
    catalog = root / "samples" / f"{split}.jsonl"
    if catalog.is_file():
        rows, referenced_keys = [], set()
        with catalog.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                sample = json.loads(line)
                motion = variant_root / sample["motion"]
                if not motion.is_file():
                    raise FileNotFoundError(f"Missing catalog motion at {catalog}:{line_number}: {motion}")
                texts = [str(text).strip() for text in sample["texts"] if str(text).strip()]
                if not texts:
                    raise ValueError(f"No text descriptions at {catalog}:{line_number}")
                keys = [text_embedding_key(text) for text in texts]
                referenced_keys.update(keys)
                row = {
                    "motion": motion.relative_to(variant_root).as_posix(),
                    "name": sample["name"],
                    "text_vector_keys": keys,
                }
                if "start_frame" in sample:
                    start, end = int(sample["start_frame"]), int(sample["end_frame"])
                    if start < 0 or end - start < 2:
                        raise ValueError(f"Invalid frame range at {catalog}:{line_number}: [{start}, {end})")
                    row.update(start_frame=start, end_frame=end)
                rows.append(row)
        if not rows:
            raise ValueError(f"Empty sample catalog: {catalog}")
        return _write_manifest(variant_root, split, rows)
    rows = []
    referenced_keys = set()
    for motion in sorted(motion_root.rglob("*.npz")):
        relative_motion = motion.relative_to(motion_root)
        relative = relative_motion.with_suffix("")
        text = root / "texts" / split / relative_motion.with_suffix(".txt")
        text_dir = root / "texts" / split / relative
        if text.is_file():
            text_files = [text]
        elif text_dir.is_dir():
            text_files = sorted(text_dir.rglob("*.txt"))
        else:
            raise FileNotFoundError(f"Missing text paired with {motion}")
        if not text_files:
            raise ValueError(f"No text descriptions paired with {motion}")
        keys = [text_embedding_key(path.read_text(encoding="utf-8")) for path in text_files]
        referenced_keys.update(keys)
        row = {"motion": motion.relative_to(variant_root).as_posix(), "name": relative.as_posix()}
        row["text_vector_keys"] = keys
        rows.append(row)
    segments = root / "segments" / f"{split}.jsonl"
    if segments.is_file():
        with segments.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                segment = json.loads(line)
                motion = variant_root / segment["motion"]
                if not motion.is_file():
                    raise FileNotFoundError(f"Missing segment source motion at {segments}:{line_number}: {motion}")
                start, end = int(segment["start_frame"]), int(segment["end_frame"])
                if start < 0 or end - start < 2:
                    raise ValueError(f"Invalid frame range at {segments}:{line_number}: [{start}, {end})")
                key = text_embedding_key(segment["text"])
                referenced_keys.add(key)
                rows.append({
                    "motion": motion.relative_to(variant_root).as_posix(),
                    "name": segment["name"],
                    "start_frame": start,
                    "end_frame": end,
                    "text_vector_keys": [key],
                })
    if not rows:
        raise ValueError(f"No motion NPZ files found under {motion_root}")
    return _write_manifest(variant_root, split, rows)


class MotionDataset(Dataset):
    def __init__(
        self,
        manifest: str,
        text_encoder: str = "llm2vec",
        load_text: bool = True,
        motion_cache_size: int = 0,
        expected_joints: Optional[int] = None,
    ):
        self.manifest = Path(manifest).resolve()
        self.load_text = load_text
        self.motion_cache_size = max(0, int(motion_cache_size))
        self.expected_joints = expected_joints
        variant_root = self.manifest.parent
        if variant_root.parent.name != "skeletons":
            raise ValueError(f"Manifest must be under <dataset-root>/skeletons/<skeleton>: {self.manifest}")
        self.dataset_root = variant_root.parent.parent
        self.text_encoder = text_encoder
        self._text_vectors = None
        self._motion_cache = OrderedDict()
        with self.manifest.open(encoding="utf-8") as stream:
            self.items = [json.loads(line) for line in stream if line.strip()]
        if not self.items:
            raise ValueError(f"Empty manifest: {self.manifest}")
        if self.load_text:
            missing = [index + 1 for index, item in enumerate(self.items) if not item.get("text_vector_keys")]
            if missing:
                raise ValueError(
                    f"Manifest {self.manifest} has no text_vector_keys at row {missing[0]}; rebuild it with kimodo_stats."
                )
            read_text_vector_metadata(self.dataset_root, self.text_encoder)
            read_snapshot_metadata(self.dataset_root, self.text_encoder)

    def __len__(self):
        return len(self.items)

    def _resolve(self, value):
        path = Path(value)
        return path if path.is_absolute() else self.manifest.parent / path

    def _vector_store(self):
        if self._text_vectors is None:
            self._text_vectors = TextVectorStore(self.dataset_root, self.text_encoder)
        return self._text_vectors

    def _load_motion(self, value):
        path = self._resolve(value)
        cached = self._motion_cache.get(path)
        if cached is not None:
            self._motion_cache.move_to_end(path)
            return cached
        with np.load(path, allow_pickle=False) as data:
            motion = (
                np.asarray(data["local_rot_mats"], dtype=np.float32),
                np.asarray(data["root_positions"], dtype=np.float32),
            )
        if self.motion_cache_size:
            self._motion_cache[path] = motion
            self._motion_cache.move_to_end(path)
            while len(self._motion_cache) > self.motion_cache_size:
                self._motion_cache.popitem(last=False)
        return motion

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_text_vectors"] = None
        state["_motion_cache"] = OrderedDict()
        return state

    def close(self):
        store = getattr(self, "_text_vectors", None)
        if store is not None:
            store.close()
            self._text_vectors = None

    def __del__(self):
        self.close()

    def __getitem__(self, index):
        item = self.items[index]
        rotations, roots = self._load_motion(item["motion"])
        if rotations.ndim != 4 or rotations.shape[-2:] != (3, 3):
            raise ValueError(f"Expected local_rot_mats [T,J,3,3], got {rotations.shape}")
        if roots.ndim != 2 or roots.shape[-1] != 3:
            raise ValueError(f"Expected root_positions [T,3], got {roots.shape}")
        if self.expected_joints is not None and rotations.shape[1] != self.expected_joints:
            raise ValueError(
                f"Motion {item['motion']} has {rotations.shape[1]} joints; "
                f"configured skeleton requires {self.expected_joints}"
            )
        if "start_frame" in item:
            start, end = int(item["start_frame"]), int(item["end_frame"])
            rotations, roots = rotations[start:end], roots[start:end]
        if len(rotations) != len(roots):
            raise ValueError(f"Mismatched motion arrays in {item['motion']}")
        sample = {
            "local_rot_mats": torch.from_numpy(rotations),
            "root_positions": torch.from_numpy(roots),
            "name": item.get("name", Path(item["motion"]).stem),
        }
        if self.load_text:
            key = random.choice(item["text_vector_keys"])
            embedded = self._vector_store().get(key)
            sample["text_feat"] = torch.from_numpy(embedded)
        return sample


def collate_motion_batch(samples, max_frames=300):
    """Randomly crop and pad a variable-length batch."""
    lengths = torch.tensor([min(len(x["root_positions"]), max_frames) for x in samples])
    if torch.any(lengths < 2):
        raise ValueError("Kimodo training clips must contain at least two frames")
    batch, max_len = len(samples), int(lengths.max())
    joints = samples[0]["local_rot_mats"].shape[1]
    rotations = torch.zeros(batch, max_len, joints, 3, 3)
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
    return rotations, roots, lengths, frame_mask, text, text_mask
