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

### Optional: prepare LLM2Vec without gated Hugging Face access

If the gated `meta-llama/Meta-Llama-3-8B-Instruct` base model is unavailable
from Hugging Face, download it from ModelScope and map it into the Linux Hugging
Face cache:

```bash
pip install modelscope

export HF_HOME="$HOME/.cache/huggingface"
MODEL_DIR="$HOME/models/Meta-Llama-3-8B-Instruct"
HF_MODEL_DIR="$HF_HOME/hub/models--meta-llama--Meta-Llama-3-8B-Instruct"

modelscope download \
  --model LLM-Research/Meta-Llama-3-8B-Instruct \
  --local_dir "$MODEL_DIR"

mkdir -p "$HF_MODEL_DIR/snapshots" "$HF_MODEL_DIR/refs"
ln -sfn "$MODEL_DIR" "$HF_MODEL_DIR/snapshots/modelscope"
printf '%s\n' modelscope > "$HF_MODEL_DIR/refs/main"

# Enable these after the first successful model load.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

## Prepare data

AssetCMS prepares motion assets and text vectors. Kimodo only computes its
motion statistics/manifests and consumes the completed dataset.

### 1. Convert motion and text

#### Standard dataset layout

One root shares semantic data across skeleton representations, while motion-derived
data is isolated by the skeleton's canonical name:

```text
dataset-root/
├── samples/{train,val,test}.jsonl      # shared annotations and splits
├── texts/{train,val,test}/**/*.txt     # shared alternative text layout
├── motion-text-index.json              # vector schema and model provenance
├── training-vectors/llm2vec/           # published float16 mmap training snapshot
├── bvh/soma/                           # retained source previews
└── skeletons/
    ├── somaskel30/motions/{train,val,test}/**/*.npz
    └── smplx22/motions/{train,val,test}/**/*.npz
```

Each motion NPZ contains `local_rot_mats` (`T,J,3,3`) and `root_positions`
(`T,3`) at 30 FPS. An optional `source.json` can declare `skeleton`,
`joint_count`, `fps`, `coordinate_system`, and `units`; statistics and training
reject incompatible metadata or joint counts.

Dataset-specific download, extraction, and conversion are intentionally outside
this repository. Kimodo consumes any producer's completed dataset root through
the contract above; training code does not import or invoke dataset adapters.

#### SMPL-X

SMPL-X training uses 22 body joints and `configs/train_smplx.yaml`. Convert the
source motion into the standard layout first, then use the SMPL-X config for
both statistics and training. `configs/train_smplx_smoke.yaml` provides the
four-step pipeline check.

### 2. Prepare text vectors in AssetCMS

Kimodo no longer contains a text-cache producer. AssetCMS owns deduplication,
encoding, indexing, provenance, and publishing. Existing LLM2Vec data can be
migrated without running the 8B model again; consult AssetCMS motion processing
documentation. Kimodo accepts only the published float16 mmap snapshot at
`training-vectors/<encoder>/` and never connects to the producer's vector
database or loads a frozen text encoder during training. mmap supports normal
multi-worker DataLoaders and shares the operating-system page cache across
workers and distributed training processes.

#### Optional: BGE-M3 multilingual conditioning

To create a new multilingual model family, incrementally add BGE-M3 to the
AssetCMS collection, then select it with `kimodo_train --text-encoder bge-m3`.

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

After AssetCMS preparation is complete, compute normalization statistics from the
training split and generate the dataset manifests:

```bash
kimodo_stats --config configs/train_soma.yaml \
  --dataset-root /data/dataset-root
```

The command uses CPU with a batch size of 8 and four parallel statistics
workers. Each worker computes local sufficient statistics and the parent
process merges them. Use `--batch-size` or `--num-workers` only when the
machine requires different values.

The statistics are tied to the dataset, skeleton, frame rate, crop length, and
feature conversion, so rerun this step when any of them changes. After all
three preparation steps, the directory is:

```text
dataset-root/
├── texts/{train,val,test}/**/*.txt    # custom directory datasets
├── samples/{train,val,test}.jsonl     # optional catalog and temporal slices
├── motion-text-index.json             # shared vector contract
├── training-vectors/llm2vec/          # mmap snapshot used by training
└── skeletons/somaskel30/
    ├── motions/{train,val,test}/**/*.npz
    ├── train.jsonl
    ├── val.jsonl                      # only when validation exists
    ├── dataset.json
    └── stats/motion/
        ├── global_root/{mean,std}.npy
        ├── local_root/{mean,std}.npy
        └── body/{mean,std}.npy
```

The generated embeddings, manifests, metadata, and statistics do not need to
be edited. Both `kimodo_stats` and `kimodo_train` select the skeleton subdirectory
from the config, so `--dataset-root` always points at the shared root.

## Run

Train the prepared dataset root:

```bash
kimodo_train --config configs/train_soma.yaml \
  --dataset-root /data/dataset-root --output runs/kimodo-soma

torchrun --nproc_per_node=4 -m kimodo.scripts.train \
  --config configs/train_soma.yaml --dataset-root /data/dataset-root \
  --output runs/kimodo-soma
```

Before a long run, exercise the complete pipeline with the small single-device
configuration (four batch steps across both curriculum phases):

```bash
kimodo_train --config configs/train_smoke.yaml \
  --dataset-root /data/dataset-root --output runs/kimodo-smoke
```

This configuration is intentionally not architecture-compatible with released
checkpoints and its metrics are not meaningful; it validates data loading,
SQLite embeddings, forward/backward, validation, EMA, logging, and checkpoint
writing.

The training loop implements 500k text-only global batch steps followed by 500k
constraint-curriculum global batch steps, 30 fps / 300 frame crops, 1000 diffusion
steps, Adam-atan2 at `2e-5`, random initial headings, and the seven weighted
Smooth-L1 terms reported in the paper. During phase two, 10% of samples remain
text-only, 25% mix two constraint families, and the sparse-keyframe limit grows
linearly from 1 to 20 while remaining biased toward fewer keyframes. Transformer
dropout changes from 0.1 to zero at the phase boundary. Text dropout is sampled
independently per item.

The final `model.safetensors` contains the EMA weights used for inference
(decay 0.995, updated every 10 batch steps). `training-state.pt` separately
stores raw weights, EMA, optimizer, mixed-precision scaler, RNG state, epoch,
micro-batch-step and global-batch-step counters for resume. One batch step
consumes `expected_global_batch_size` samples and performs one parameter update.
Single-process resume restores the exact saved RNG streams. Distributed resume
re-seeds each rank independently because the rank-zero state file does not
contain every worker's RNG stream.

## Production logging and validation

Every rank-zero run writes `events.jsonl`, `run-metadata.json`, and an exact
`training-config.yaml` under the output directory. JSONL records training loss
components, learning rate, gradient norm, throughput, batch-step time, AMP loss
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
    every_batch_steps: 2000
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

`configs/train_soma.yaml` targets the report's 4-GPU global batch of 512 using
a conservative 48 GiB starting point of 16 samples per GPU. Users do not set
gradient accumulation: the trainer derives it as
`expected_global_batch_size / (batch_size * WORLD_SIZE)`. This gives 8 on four
GPUs and 32 on one GPU. The target must divide the distributed micro-batch
exactly or startup fails. Profile representative 300-frame batches before a
long run; increasing `batch_size` to 32 on four GPUs automatically reduces
accumulation to 4 while preserving the same global batch.

The Bones Rigplay data, transition-generated stitched augmentation, and
NVIDIA's exact data-mixture distribution are not public;
the manifest intentionally permits Rigplay, BONES-SEED, or a private retargeted
dataset without encoding proprietary assumptions. Consequently this entry
point is an open-data approximation of the report recipe, not a claim that the
released 700-hour checkpoint can be reproduced from BONES-SEED alone.
