"""Read-only mmap text-vector snapshots published by AssetCMS."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

INDEX_METADATA_NAME = "motion-text-index.json"
SNAPSHOT_FORMAT = "kimodo-text-vectors-mmap-v1"


def read_text_vector_metadata(dataset_root: str | Path, encoder: str) -> tuple[dict, dict]:
    root = Path(dataset_root).expanduser().resolve()
    path = root / INDEX_METADATA_NAME
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing AssetCMS text-vector metadata: {path}. Run the AssetCMS motion embedding migration/indexer."
        )
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 1:
        raise ValueError(f"Unsupported motion text index metadata: {path}")
    provenance = metadata.get("encoders", {}).get(encoder)
    if not provenance:
        raise ValueError(f"Text encoder {encoder!r} is not present in {path}")
    return metadata, provenance


def snapshot_path(dataset_root: str | Path, encoder: str) -> Path:
    return Path(dataset_root).expanduser().resolve() / "training-vectors" / encoder


def read_snapshot_metadata(dataset_root: str | Path, encoder: str) -> tuple[Path, dict]:
    path = snapshot_path(dataset_root, encoder)
    metadata_path = path / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Missing published text-vector snapshot: {metadata_path}. "
            f"Run AssetCMS publish_motion_text_vectors --encoder {encoder}."
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 1 or metadata.get("format") != SNAPSHOT_FORMAT:
        raise ValueError(f"Unsupported text-vector snapshot: {metadata_path}")
    if metadata.get("encoder") != encoder:
        raise ValueError(f"Snapshot encoder mismatch: {metadata.get('encoder')} != {encoder}")
    count, dimension = int(metadata["count"]), int(metadata["dimension"])
    vectors_path = path / metadata["vectors"]
    keys_path = path / metadata["keys"]
    if not vectors_path.is_file() or vectors_path.stat().st_size != count * dimension * 2:
        raise ValueError(f"Incomplete text-vector snapshot: {vectors_path}")
    if not keys_path.is_file():
        raise ValueError(f"Incomplete text-vector key index: {keys_path}")
    return path, metadata


class TextVectorStore:
    """Consume one required AssetCMS float16 mmap snapshot."""

    def __init__(self, dataset_root: str | Path, encoder: str):
        self.path, self.metadata = read_snapshot_metadata(dataset_root, encoder)
        self.dimension = int(self.metadata["dimension"])
        self.count = int(self.metadata["count"])
        vectors_path = self.path / self.metadata["vectors"]
        expected = self.count * self.dimension * np.dtype("<f2").itemsize
        if vectors_path.stat().st_size != expected:
            raise ValueError(f"Incomplete text-vector snapshot: {vectors_path}")
        with (self.path / self.metadata["keys"]).open(encoding="ascii") as stream:
            keys = [line.strip() for line in stream if line.strip()]
        if len(keys) != self.count or len(set(keys)) != self.count:
            raise ValueError(f"Invalid text-vector key index: expected {self.count}, got {len(keys)}")
        self.rows = {key: row for row, key in enumerate(keys)}
        self.vectors = np.memmap(
            vectors_path, dtype="<f2", mode="r", shape=(self.count, self.dimension)
        )

    def get(self, prompt_key: str) -> np.ndarray:
        try:
            row = self.rows[prompt_key]
        except KeyError as error:
            raise KeyError(f"Text vector missing from mmap snapshot: {prompt_key}") from error
        return np.asarray(self.vectors[row], dtype=np.float32)[None]

    def close(self):
        mmap = getattr(self.vectors, "_mmap", None)
        if mmap is not None:
            mmap.close()


__all__ = [
    "TextVectorStore", "read_snapshot_metadata", "read_text_vector_metadata", "snapshot_path",
]
