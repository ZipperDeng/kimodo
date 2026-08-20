"""Precompute text embeddings for a prepared training dataset."""

import argparse
import io
import json
import sqlite3
from pathlib import Path

import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf

from kimodo.training.data import (
    BonesSeedDataset,
    is_seed_dataset_root,
    normalize_seed_prompt,
    resolve_dataset_root,
)
from kimodo.model.text_encoder_registry import (
    TEXT_ENCODER_PRESETS, build_text_encoder_config, text_encoder_provenance_from_config,
)


def _prompts(root):
    import csv

    csv_path = sorted((root / "metadata").glob("seed_metadata_v*.csv"))[-1]
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            for key in BonesSeedDataset.text_columns:
                value = str(row.get(key, "")).strip()
                if value and value.lower() != "nan":
                    yield normalize_seed_prompt(value)
    labels = sorted((root / "metadata").glob("*temporal_labels.jsonl"))
    if labels:
        with labels[-1].open(encoding="utf-8") as stream:
            for line in stream:
                for event in json.loads(line).get("events", []):
                    yield normalize_seed_prompt(str(event["description"]))


def _encode_batches(encoder, records, batch_size):
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        features, lengths = encoder([record[1] for record in batch])
        for (key, _text), feature, length in zip(batch, features, lengths):
            yield key, feature[: int(length)].detach().cpu().float().numpy().astype(np.float16)
        print(f"{min(start + len(batch), len(records))}/{len(records)}", flush=True)


def _cache_seed(root, output, encoder, provenance, batch_size):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    prompts = sorted(set(_prompts(root)))
    connection = sqlite3.connect(output)
    connection.execute("CREATE TABLE embeddings (prompt TEXT PRIMARY KEY, embedding BLOB NOT NULL)")
    connection.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    connection.executemany(
        "INSERT INTO metadata VALUES (?, ?)",
        [("schema_version", "1"), ("text_model", json.dumps(provenance, sort_keys=True))],
    )
    try:
        for prompt, array in _encode_batches(encoder, [(prompt, prompt) for prompt in prompts], batch_size):
            buffer = io.BytesIO()
            np.save(buffer, array, allow_pickle=False)
            connection.execute("INSERT INTO embeddings VALUES (?, ?)", (prompt, buffer.getvalue()))
        connection.commit()
    finally:
        connection.close()


def _cache_generic(root, encoder, provenance, batch_size):
    records = []
    for split in ("train", "val", "test"):
        text_root = root / "texts" / split
        if text_root.is_dir():
            for path in sorted(text_root.rglob("*.txt")):
                records.append((path.relative_to(root / "texts").with_suffix(".npy"), path.read_text(encoding="utf-8").strip()))
    if not records:
        raise ValueError(f"No text files found under {root / 'texts' / 'train'}")
    output_root = root / "embeddings"
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite: {output_root}")
    for relative, array in _encode_batches(encoder, records, batch_size):
        output = output_root / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        np.save(output, array, allow_pickle=False)
    (root / "text-model.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Cache text embeddings for a prepared dataset")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Prepared dataset directory containing texts/{train,val,test}",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--text-encoder", choices=sorted(TEXT_ENCODER_PRESETS))
    args = parser.parse_args()
    root = resolve_dataset_root(args.dataset_root)
    cfg = OmegaConf.load(args.config)
    encoder_config = build_text_encoder_config(args.text_encoder) if args.text_encoder else cfg.training.text_encoder
    encoder = instantiate(encoder_config)
    provenance = getattr(encoder, "provenance", text_encoder_provenance_from_config(encoder_config))
    if is_seed_dataset_root(root):
        _cache_seed(root, root / "text-embeddings.sqlite", encoder, provenance, args.batch_size)
    else:
        _cache_generic(root, encoder, provenance, args.batch_size)


if __name__ == "__main__":
    main()
