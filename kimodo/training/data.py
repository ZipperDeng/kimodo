"""Manifest based dataset used by the Kimodo training entry point.

Each JSONL row points at an ``.npz`` containing ``local_rot_mats`` [T,J,3,3],
``root_positions`` [T,3], and either ``text_feat`` [L,4096] or a separate
``text_feat`` path.  Keeping text embeddings on disk avoids loading the 8B
LLM2Vec encoder in every data-loader worker.
"""

import json
import csv
import hashlib
import io
import random
import re
import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


def read_embedding_cache_provenance(path):
    if not path:
        return None
    connection = sqlite3.connect(Path(path).expanduser().resolve())
    try:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "metadata" not in tables:
            return None
        metadata = dict(connection.execute("SELECT key, value FROM metadata"))
        return json.loads(metadata["text_model"]) if "text_model" in metadata else None
    finally:
        connection.close()


def is_seed_dataset_root(root: Path) -> bool:
    """Return whether *root* uses the extracted SEED directory layout."""
    return (root / "soma_uniform").is_dir() and any((root / "metadata").glob("seed_metadata_v*.csv"))


def build_generic_manifest(root: Path, split: str) -> Optional[Path]:
    """Build a manifest by matching motions and embeddings by relative stem."""
    motion_root = root / "motions" / split
    if not motion_root.is_dir():
        return None
    rows = []
    for motion in sorted(motion_root.rglob("*.npz")):
        relative_motion = motion.relative_to(motion_root)
        relative = relative_motion.with_suffix("")
        text = root / "texts" / split / relative_motion.with_suffix(".txt")
        embedding = root / "embeddings" / split / relative_motion.with_suffix(".npy")
        text_dir = root / "texts" / split / relative
        embedding_dir = root / "embeddings" / split / relative
        if text.is_file() and embedding.is_file():
            text_features = [embedding]
        elif text_dir.is_dir() and embedding_dir.is_dir():
            text_files = sorted(text_dir.rglob("*.txt"))
            text_features = [embedding_dir / path.relative_to(text_dir).with_suffix(".npy") for path in text_files]
            missing = [path for path in text_features if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"Missing text embedding paired with {motion}: {missing[0]}")
        else:
            raise FileNotFoundError(f"Missing text or embedding paired with {motion}")
        if not text_features:
            raise ValueError(f"No text descriptions paired with {motion}")
        row = {"motion": motion.relative_to(root).as_posix(), "name": relative.as_posix()}
        relative_features = [path.relative_to(root).as_posix() for path in text_features]
        row["text_feat" if len(relative_features) == 1 else "text_feats"] = (
            relative_features[0] if len(relative_features) == 1 else relative_features
        )
        rows.append(row)
    if not rows:
        raise ValueError(f"No motion NPZ files found under {motion_root}")
    manifest = root / f"{split}.jsonl"
    temporary = manifest.with_suffix(".jsonl.tmp")
    temporary.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    temporary.replace(manifest)
    return manifest


class MotionDataset(Dataset):
    def __init__(self, manifest: str):
        self.manifest = Path(manifest).resolve()
        with self.manifest.open(encoding="utf-8") as stream:
            self.items = [json.loads(line) for line in stream if line.strip()]
        if not self.items:
            raise ValueError(f"Empty manifest: {self.manifest}")

    def __len__(self):
        return len(self.items)

    def _resolve(self, value):
        path = Path(value)
        return path if path.is_absolute() else self.manifest.parent / path

    def __getitem__(self, index):
        item = self.items[index]
        with np.load(self._resolve(item["motion"]), allow_pickle=False) as data:
            rotations = np.asarray(data["local_rot_mats"], dtype=np.float32)
            roots = np.asarray(data["root_positions"], dtype=np.float32)
            embedded = np.asarray(data["text_feat"], dtype=np.float32) if "text_feat" in data else None
        if embedded is None:
            text_feature = random.choice(item["text_feats"]) if "text_feats" in item else item["text_feat"]
            embedded = np.asarray(np.load(self._resolve(text_feature)), dtype=np.float32)
        if embedded.ndim == 1:
            embedded = embedded[None]
        if len(rotations) != len(roots):
            raise ValueError(f"Mismatched motion arrays in {item['motion']}")
        return {
            "local_rot_mats": torch.from_numpy(rotations),
            "root_positions": torch.from_numpy(roots),
            "text_feat": torch.from_numpy(embedded),
            "name": item.get("name", Path(item["motion"]).stem),
        }


class BonesSeedDataset(Dataset):
    """Direct reader for an extracted ``bones-studio/seed`` SOMA dataset.

    Metadata is read with the Python standard library so workers do not need
    pandas/pyarrow. Motions are parsed from SOMA Uniform BVH and reduced from
    120 fps to the requested 30 fps by exact integer decimation.
    """

    text_columns = (
        "content_natural_desc_1", "content_natural_desc_2",
        "content_natural_desc_3", "content_natural_desc_4",
        "content_technical_description", "content_short_description",
        "content_short_description_2",
    )

    def __init__(
        self, root: str, split="train", fps=30, include_mirrored=True,
        temporal_probability=0.5, embedding_cache=None, expected_embedding_dim=None,
    ):
        self.root = resolve_dataset_root(root)
        csv_files = sorted((self.root / "metadata").glob("seed_metadata_v*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No seed_metadata_v*.csv under {self.root / 'metadata'}")
        if not (self.root / "soma_uniform").is_dir():
            archive = self.root / "soma_uniform.tar.gz"
            raise FileNotFoundError(
                f"SOMA motions are not extracted under {self.root}. Extract {archive} into "
                f"{self.root} before training (tar -xzf soma_uniform.tar.gz)."
            )
        with csv_files[-1].open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.temporal_labels = {}
        label_files = sorted((self.root / "metadata").glob("*temporal_labels.jsonl"))
        if label_files:
            with label_files[-1].open(encoding="utf-8") as stream:
                for line in stream:
                    entry = json.loads(line)
                    self.temporal_labels[entry["filename"]] = entry.get("events", [])
        self.rows = []
        for row in rows:
            if not include_mirrored and row.get("is_mirror", "").lower() in ("true", "1"):
                continue
            # Keep an original clip and its _M mirror in the same split.  This
            # prevents a nearly identical motion leaking into validation/test.
            split_key = seed_motion_group(row["filename"])
            bucket = int(hashlib.sha1(split_key.encode()).hexdigest()[:8], 16) % 100
            row_split = "train" if bucket < 98 else ("val" if bucket == 98 else "test")
            if split == row_split:
                self.rows.append(row)
        self.fps = fps
        self.temporal_probability = temporal_probability
        self.embedding_cache = Path(embedding_cache).resolve() if embedding_cache else None
        self._embedding_connection = None
        self._soma30 = None
        self.text_model_provenance = None
        if self.embedding_cache:
            self.text_model_provenance = read_embedding_cache_provenance(self.embedding_cache)
            actual_dim = (self.text_model_provenance or {}).get("embedding_dim")
            if expected_embedding_dim is not None and actual_dim is not None and int(actual_dim) != int(expected_embedding_dim):
                raise ValueError(
                    f"Embedding cache dimension is {actual_dim}, model expects {expected_embedding_dim}: "
                    f"{self.embedding_cache}"
                )

    def __len__(self):
        return len(self.rows)

    @staticmethod
    def _prompt(row):
        choices = [str(row.get(key, "")).strip() for key in BonesSeedDataset.text_columns]
        text = random.choice([value for value in choices if value and value.lower() != "nan"])
        return normalize_seed_prompt(text)

    def __getitem__(self, index):
        from kimodo.skeleton.bvh import parse_bvh_motion

        row = self.rows[index]
        path = self.root / Path(row["move_soma_uniform_path"].replace("\\", "/"))
        rotations, roots, source_fps = parse_bvh_motion(str(path))
        if rotations.shape[1] == 77:
            # Public SOMA Uniform BVH is SOMA77, while Kimodo-SOMA models the
            # named 30-joint subset. Canonicalize on the source hierarchy
            # before dropping joints; doing this with SOMA30 is invalid.
            if self._soma30 is None:
                from kimodo.skeleton import SOMASkeleton30

                self._soma30 = SOMASkeleton30()
            rotations, _ = self._soma30.somaskel77.to_standard_tpose(rotations)
            rotations = self._soma30.from_SOMASkeleton77(rotations)
        elif rotations.shape[1] != 30:
            raise ValueError(f"Expected SOMA77 or SOMA30 motion in {path}, got {rotations.shape[1]} joints")
        events = self.temporal_labels.get(row["filename"], [])
        if events and random.random() < self.temporal_probability:
            event = random.choice(events)
            start = max(0, round(float(event["start_time"]) * source_fps))
            end = min(len(roots), round(float(event["end_time"]) * source_fps))
            if end - start >= 2:
                rotations, roots = rotations[start:end], roots[start:end]
                prompt = normalize_seed_prompt(str(event["description"]))
            else:
                prompt = self._prompt(row)
        else:
            prompt = self._prompt(row)
        stride = round(float(source_fps) / self.fps)
        # BVH stores frame time as a short decimal (commonly 0.008333), so a
        # nominal 120 Hz clip may parse as 120.0048 Hz. Accept that header
        # quantization while still rejecting genuinely non-integral ratios.
        if stride < 1 or abs(float(source_fps) / stride - self.fps) > 1e-2:
            raise ValueError(f"Cannot exactly resample {path} from {source_fps:g} to {self.fps:g} fps")
        sample = {
            "local_rot_mats": rotations[::stride].float(),
            "root_positions": roots[::stride].float(),
            "name": row["filename"],
        }
        if self.embedding_cache:
            if self._embedding_connection is None:
                uri = self.embedding_cache.as_uri() + "?mode=ro"
                self._embedding_connection = sqlite3.connect(uri, uri=True)
            result = self._embedding_connection.execute(
                "SELECT embedding FROM embeddings WHERE prompt = ?", (prompt,)
            ).fetchone()
            if result is None:
                raise KeyError(f"Prompt missing from embedding cache: {prompt}")
            feature = np.load(io.BytesIO(result[0]), allow_pickle=False).astype(np.float32)
            sample["text_feat"] = torch.from_numpy(feature[None] if feature.ndim == 1 else feature)
        else:
            sample["text"] = prompt
        return sample


def resolve_dataset_root(root: str) -> Path:
    """Resolve a compatible dataset directory or the cached official snapshot."""
    if root != "auto":
        return Path(root).expanduser().resolve()
    from huggingface_hub import snapshot_download

    return Path(snapshot_download("bones-studio/seed", repo_type="dataset", local_files_only=True))


def seed_motion_group(filename: str) -> str:
    """Return the shared split key for an original SEED clip and its mirror."""
    return re.sub(r"_M$", "", filename, flags=re.IGNORECASE)


def normalize_seed_prompt(text: str) -> str:
    """Normalize public SEED descriptions to Kimodo's sentence convention."""
    text = " ".join(text.strip().split()).rstrip(".")
    if not text:
        raise ValueError("SEED prompt is empty")
    lower = text.lower()
    while lower.startswith("person person "):
        text = text[len("person ") :]
        lower = text.lower()
    if lower.startswith("person "):
        text = "A " + text
    elif lower.startswith("individual "):
        text = "An " + text
    elif not lower.startswith(("a ", "an ", "the ")):
        text = "A person " + text
    return text[0].upper() + text[1:] + "."


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
        text = [x["text"] for x in samples]
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
