# Training Kimodo

The original release contains inference weights but no data loader or training
entry point.  `kimodo.scripts.train` implements the recipe in section 4.3 of
the technical report while reusing the released motion representation and
two-stage denoiser.

## Set up the environment

On a new Linux machine, clone the repository and run the setup script from its
root. It installs `uv` when necessary and creates a Kimodo environment for
both training and inference, installs all dependencies, and verifies CUDA:

```bash
bash scripts/setup_env.sh
source .venv/bin/activate
```

The default environment is `.venv`. To keep it outside the workspace, provide
a user-level path:

```bash
bash scripts/setup_env.sh --venv-dir "$HOME/venvs/kimodo"
source "$HOME/venvs/kimodo/bin/activate"
```

## Prepare data

Data preparation follows one order: convert motion and text, encode the text,
then compute motion statistics and manifests.

### 1. Convert motion and text

#### BONES-SEED (recommended)

The Hugging Face snapshot contains a 42.1 GiB `soma_uniform.tar.gz` plus SEED
metadata. Convert it to Kimodo's standard motion/text directory layout (the
command is offline and never writes to the Hugging Face cache):

```bash
kimodo_prepare_seed --source auto --output /data/dataset-root
```

Use `kimodo_prepare_seed --source auto --output ... --dry-run` to verify the
cached revision and archive size without extracting. The output path must be
new; interrupted extraction is retained as `<output>.extracting` so that a
large partial result is never silently deleted.

The converter applies a stable 98/1/1 split while keeping each original/`_M`
mirror pair together, converts SOMA Uniform BVH from 120 fps to Kimodo's 30
fps NPZ format, preserves all seven overview descriptions as text variants,
and materializes public NVIDIA temporal labels as atomic sub-clips.

#### Custom datasets

Custom datasets use the same input layout. Motion and text files are paired by
their relative path and filename stem:

```text
dataset-root/
├── motions/
│   ├── train/walk/000001.npz
│   └── val/walk/000002.npz       # optional validation split
└── texts/
    ├── train/walk/000001.txt      # one description
    ├── train/run/000003/          # or multiple descriptions for one motion
    │   ├── 00.txt
    │   └── 01.txt
    └── val/walk/000002.txt
```

Each motion NPZ contains `local_rot_mats` (`T,J,3,3`) and `root_positions`
(`T,3`).

### 2. Encode text

Choose one text encoder and build the cache once. The default is LLM2Vec:

```bash
kimodo_cache_text --config configs/train_soma.yaml \
  --dataset-root /data/dataset-root
```

The command mirrors every `.txt` into `embeddings/` as a float16 `.npy` and
writes `text-model.json`. The frozen encoder is kept on CPU by default; set
`training.text_encoder` to `kimodo.model.TextEncoderAPI` with a `url` to use a
separately hosted encoder service.

#### Optional: BGE-M3 multilingual conditioning

To create a new multilingual model family, use BGE-M3 instead of the default
encoder when building the cache:

```bash
kimodo_cache_text --config configs/train_soma.yaml \
  --dataset-root /data/dataset-root --text-encoder bge-m3
```

BGE-M3 uses one normalized 1024-dimensional CLS embedding token and does not
require a query instruction. Add `--text-encoder bge-m3` to the training command
as well. The preset changes `model.denoiser.llm_shape` from `[1,4096]` to
`[1,1024]`; training rejects embeddings whose recorded dimension differs from
the denoiser.

Changing the encoder creates a new model family: released LLM2Vec Kimodo/TMR
weights are not embedding-space compatible with BGE-M3. A BGE-M3 Kimodo model
can still be evaluated with the fixed official LLM2Vec TMR evaluator for a
controlled comparison. Using BGE-M3 inside TMR itself requires retraining TMR.

### 3. Compute motion statistics and manifests

After text embedding is complete, compute normalization statistics from the
training split and generate the dataset manifests:

```bash
kimodo_stats --config configs/train_soma.yaml \
  --dataset-root /data/dataset-root
```

The statistics are tied to the dataset, skeleton, frame rate, crop length, and
feature conversion, so rerun this step when any of them changes. After all
three preparation steps, the directory is:

```text
dataset-root/
├── motions/{train,val,test}/**/*.npz
├── texts/{train,val,test}/**/*.txt
├── embeddings/{train,val,test}/**/*.npy
├── train.jsonl
├── val.jsonl                         # only when a validation split exists
├── dataset.json
├── text-model.json
└── stats/motion/
    ├── global_root/{mean,std}.npy
    ├── local_root/{mean,std}.npy
    └── body/{mean,std}.npy
```

The generated embeddings, manifests, metadata, and statistics do not need to
be edited.

## Run

Train the prepared dataset root:

```bash
kimodo_train --config configs/train_soma.yaml \
  --dataset-root /data/dataset-root --output runs/kimodo-soma

torchrun --nproc_per_node=8 -m kimodo.scripts.train \
  --config configs/train_soma.yaml --dataset-root /data/dataset-root \
  --output runs/kimodo-soma
```

The training loop implements 500k text-only optimizer steps followed by 500k
constraint-curriculum optimizer steps, 30 fps / 300 frame crops, 1000 diffusion
steps, Adam-atan2 at `2e-5`, random initial headings, and the seven weighted
Smooth-L1 terms reported in the paper. During phase two, 10% of samples remain
text-only, 25% mix two constraint families, and the sparse-keyframe limit grows
linearly from 1 to 20 while remaining biased toward fewer keyframes. Transformer
dropout changes from 0.1 to zero at the phase boundary. Text dropout is sampled
independently per item.

The final `model.safetensors` contains the EMA weights used for inference
(decay 0.995, updated every 10 optimizer steps). `training-state.pt` separately
stores raw weights, EMA, optimizer, mixed-precision scaler, RNG state, epoch,
micro-step, and optimizer-step counters for resume. Gradient accumulation and
curriculum progress are defined in optimizer steps, not dataloader micro-steps.
Single-process resume restores the exact saved RNG streams. Distributed resume
re-seeds each rank independently because the rank-zero state file does not
contain every worker's RNG stream.

## Production logging and validation

Every rank-zero run writes `events.jsonl`, `run-metadata.json`, and an exact
`training-config.yaml` under the output directory. JSONL records training loss
components, learning rate, gradient norm, throughput, step time, AMP loss
scale, phase, epoch, and CUDA allocated/reserved memory. Non-finite loss or
gradient norm terminates training immediately. A new run refuses to overwrite
an output containing `training-state.pt`; continuation requires `--resume`.

Install optional dashboards with `pip install -e ".[train]"` and select one or
more fan-out backends:

```yaml
training:
  logging:
    backends: [jsonl, tensorboard, wandb]
    project: kimodo
    run_name: soma-seed-001
    entity: null
    group: seed-retraining
    tags: [soma, seed]
```

For BONES-SEED, validation automatically uses the leakage-safe `val` split.
For a generic dataset, the generated `val.jsonl` is used when
`motions/val` exists. Validation is deterministic, text-only, bounded by
`max_batches`, and evaluates EMA weights without replacing the raw training
parameters:

```yaml
training:
  validation:
    enabled: true
    every: 2000
    batch_size: 8
    max_batches: 32
    monitor: loss
```

`model-best.safetensors` tracks the monitored validation value;
`model.safetensors` is the final EMA model. Periodic models are written
atomically and pruned to `keep_checkpoints`, while `training-state.pt` contains
the resumable optimizer, scaler, EMA, RNG, counters, and best metric.

R@3, FID, foot skate, contact accuracy, and constraint errors require DDIM
generation over a fixed benchmark rather than a denoising validation batch.
Run the documented `benchmark/` pipeline on saved checkpoints at a lower
frequency (for example every 20k--50k steps); do not use training loss as a
replacement for those generation metrics.

`configs/train_soma.yaml` deliberately uses a per-process batch size of one as
a safe smoke-test default. At startup the trainer prints the effective global
batch (`per_device * world_size * accumulation`) and warns when it differs from
the report's 2048. Adjust batch size and accumulation for production runs.

The Bones Rigplay data, transition-generated stitched augmentation, and
NVIDIA's exact data-mixture distribution are not public;
the manifest intentionally permits Rigplay, BONES-SEED, or a private retargeted
dataset without encoding proprietary assumptions. Consequently this entry
point is an open-data approximation of the report recipe, not a claim that the
released 700-hour checkpoint can be reproduced from BONES-SEED alone.
