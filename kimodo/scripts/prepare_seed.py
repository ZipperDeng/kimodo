"""Prepare the locally cached bones-studio/seed archive for Kimodo training."""

import argparse
import json
import shutil
import tarfile
from pathlib import Path

import numpy as np

from kimodo.training.data import BonesSeedDataset, normalize_seed_prompt, resolve_dataset_root


def arguments():
    parser = argparse.ArgumentParser(description="Convert cached SEED data to the Kimodo dataset layout")
    parser.add_argument("--source", default="auto", help="SEED snapshot directory, or 'auto' for the local HF cache")
    parser.add_argument("--output", required=True, help="New output directory (must not already exist)")
    parser.add_argument("--dry-run", action="store_true", help="Validate paths and report required archive size only")
    return parser.parse_args()


def _safe_target(root: Path, name: str) -> Path:
    target = (root / name).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"Unsafe path in archive: {name}") from error
    return target


def extract_seed(source: Path, output: Path):
    archive = source / "soma_uniform.tar.gz"
    if not archive.is_file():
        raise FileNotFoundError(archive)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    temporary = output.with_name(output.name + ".extracting")
    if temporary.exists():
        raise FileExistsError(f"Remove or rename the incomplete extraction first: {temporary}")
    temporary.mkdir(parents=True)
    try:
        # Streaming mode avoids loading the ~142k-member, 45 GB archive index.
        with tarfile.open(archive, mode="r|gz") as stream:
            for member in stream:
                _safe_target(temporary, member.name)
                if member.issym() or member.islnk() or member.isdev():
                    raise ValueError(f"Unsupported archive member: {member.name}")
                stream.extract(member, temporary)
        shutil.copytree(source / "metadata", temporary / "metadata")
        for name in ("README.md", "LICENSE.md", "LICENSE-DATA.md"):
            path = source / name
            if path.is_file():
                shutil.copy2(path, temporary / name)
        convert_seed_layout(temporary)
        temporary.replace(output)
    except BaseException:
        # Keep partial data for diagnosis; never silently delete tens of GB.
        raise


def _write_text_variants(folder: Path, values) -> None:
    prompts = []
    for value in values:
        value = str(value).strip()
        if value and value.lower() != "nan":
            prompt = normalize_seed_prompt(value)
            if prompt not in prompts:
                prompts.append(prompt)
    if not prompts:
        raise ValueError(f"No usable text descriptions for {folder.name}")
    folder.mkdir(parents=True, exist_ok=False)
    for index, prompt in enumerate(prompts):
        (folder / f"{index:02d}.txt").write_text(prompt + "\n", encoding="utf-8")


def convert_seed_layout(root: Path, fps: int = 30) -> None:
    """Convert extracted SEED BVH and metadata to the generic Kimodo layout."""
    converted = 0
    for split in ("train", "val", "test"):
        dataset = BonesSeedDataset(root, split=split, fps=fps, temporal_probability=0.0)
        for index, row in enumerate(dataset.rows):
            sample = dataset[index]
            name = Path(row["filename"]).name
            motion_path = root / "motions" / split / f"{name}.npz"
            motion_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                motion_path,
                local_rot_mats=sample["local_rot_mats"].numpy(),
                root_positions=sample["root_positions"].numpy(),
            )
            _write_text_variants(
                root / "texts" / split / name,
                [row.get(column, "") for column in BonesSeedDataset.text_columns],
            )
            for event_index, event in enumerate(dataset.temporal_labels.get(row["filename"], [])):
                start = max(0, round(float(event["start_time"]) * fps))
                end = min(len(sample["root_positions"]), round(float(event["end_time"]) * fps))
                if end - start < 2:
                    continue
                event_name = f"{name}__event_{event_index:03d}"
                np.savez_compressed(
                    root / "motions" / split / f"{event_name}.npz",
                    local_rot_mats=sample["local_rot_mats"][start:end].numpy(),
                    root_positions=sample["root_positions"][start:end].numpy(),
                )
                _write_text_variants(
                    root / "texts" / split / event_name,
                    [event["description"]],
                )
            converted += 1
            if converted % 1000 == 0:
                print(f"converted {converted} SEED motions", flush=True)
    metadata = root / "metadata"
    metadata.rename(root / "source_metadata")
    raw_motions = (root / "soma_uniform").resolve()
    raw_motions.relative_to(root.resolve())
    shutil.rmtree(raw_motions)
    (root / "source.json").write_text(
        json.dumps({"dataset": "bones-studio/seed", "fps": fps}, indent=2) + "\n",
        encoding="utf-8",
    )


def main():
    args = arguments()
    source = resolve_dataset_root(args.source)
    output = Path(args.output).expanduser().resolve()
    archive = source / "soma_uniform.tar.gz"
    print(f"source:  {source}")
    print(f"archive: {archive} ({archive.stat().st_size / 2**30:.1f} GiB compressed)")
    print(f"output:  {output}")
    if not args.dry_run:
        extract_seed(source, output)
        print("SEED motion and text data prepared successfully.")


if __name__ == "__main__":
    main()
