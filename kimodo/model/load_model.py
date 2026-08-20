# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Load Kimodo diffusion models from local checkpoints or Hugging Face."""

from pathlib import Path
from typing import Optional

from huggingface_hub import snapshot_download
from omegaconf import OmegaConf, open_dict

from .loading import (
    AVAILABLE_MODELS,
    DEFAULT_MODEL,
    DEFAULT_TEXT_ENCODER_URL,
    MODEL_NAMES,
    TMR_MODELS,
    get_env_var,
    instantiate_from_dict,
)
from .registry import get_model_info, resolve_model_name
from .text_encoder_registry import DEFAULT_TEXT_ENCODER, build_text_encoder_config, read_checkpoint_text_model



def _resolve_hf_model_path(modelname: str) -> Path:
    """Resolve model name to a local path, using Hugging Face cache or CHECKPOINT_DIR."""
    try:
        repo_id = MODEL_NAMES[modelname]
    except KeyError:
        raise ValueError(f"Model '{modelname}' not found. Available models: {MODEL_NAMES.keys()}")

    local_cache = get_env_var("LOCAL_CACHE", "False").lower() == "true"
    if not local_cache:
        snapshot_dir = snapshot_download(repo_id=repo_id)  # will check online no matter what
        return Path(snapshot_dir)

    try:
        snapshot_dir = snapshot_download(repo_id=repo_id, local_files_only=True)  # will check local cache only
        return Path(snapshot_dir)
    except Exception:
        # if local cache is not found, download from online
        try:
            snapshot_dir = snapshot_download(repo_id=repo_id)
            return Path(snapshot_dir)
        except Exception:
            raise RuntimeError(f"Could not resolve model '{modelname}' from Hugging Face (repo: {repo_id}). ") from None


def _build_api_text_encoder_conf(text_encoder_url: str) -> dict:
    return {
        "_target_": "kimodo.model.text_encoder_api.TextEncoderAPI",
        "url": text_encoder_url,
    }


def _build_local_text_encoder_conf(text_encoder_fp32: bool = False, text_encoder_name=None) -> dict:
    text_encoder_name = text_encoder_name or get_env_var("TEXT_ENCODER", DEFAULT_TEXT_ENCODER)
    return build_text_encoder_config(text_encoder_name, fp32=text_encoder_fp32)


def _select_text_encoder_conf(text_encoder_url: str, text_encoder_fp32: bool = False, text_encoder_name=None) -> dict:
    # TEXT_ENCODER_MODE options:
    # - "api": force TextEncoderAPI
    # - "local": force local LLM2VecEncoder
    # - "auto": try API first, fallback to local if unreachable
    mode = get_env_var("TEXT_ENCODER_MODE", "auto").lower()
    # A persisted/explicit preset is part of the checkpoint contract. An
    # unversioned API cannot prove that it serves the same embedding space.
    if mode == "auto" and text_encoder_name is not None:
        return _build_local_text_encoder_conf(text_encoder_fp32, text_encoder_name)
    if mode == "local":
        return _build_local_text_encoder_conf(text_encoder_fp32, text_encoder_name)
    if mode == "api":
        return _build_api_text_encoder_conf(text_encoder_url)

    api_conf = _build_api_text_encoder_conf(text_encoder_url)
    try:
        text_encoder = instantiate_from_dict(api_conf)
        # Probe availability early so inference doesn't fail later.
        text_encoder(["healthcheck"])
        return api_conf
    except Exception as error:
        print(
            "Text encoder service is unreachable, falling back to local LLM2Vec "
            f"encoder. ({type(error).__name__}: {error})"
        )
        return _build_local_text_encoder_conf(text_encoder_fp32, text_encoder_name)


def load_model(
    modelname=None,
    device=None,
    eval_mode: bool = True,
    default_family: Optional[str] = "Kimodo",
    return_resolved_name: bool = False,
    text_encoder=None,
    text_encoder_fp32: bool = False,
    text_encoder_name: Optional[str] = None,
):
    """Load a kimodo model by name (e.g. 'g1', 'soma').

    Resolution of partial/full names (e.g. Kimodo-SOMA-RP-v1, SOMA) is done
    inside this function using default_family when the name is not a known
    short key.

    Args:
        modelname: Model identifier; uses DEFAULT_MODEL if None. Can be a short key,
            a full name (e.g. Kimodo-SOMA-RP-v1), or a partial name; unknown names
            are resolved via resolve_model_name using default_family.
        device: Target device for the model (e.g. 'cuda', 'cpu').
        eval_mode: If True, set model to eval mode.
        default_family: Used when modelname is not in AVAILABLE_MODELS to resolve
            partial names ("Kimodo" for demo/generation, "TMR" for embed script).
            Default "Kimodo".
        return_resolved_name: If True, return (model, resolved_short_key). If False,
            return only the model.
        text_encoder: Pre-built text encoder to reuse. When provided, skips
            text encoder selection/instantiation entirely.
        text_encoder_fp32: If True, uses fp32 for the text encoder rather than default bfloat16.

    Returns:
        Loaded model in eval mode, or (model, resolved short key) if
        return_resolved_name is True.

    Raises:
        ValueError: If modelname is not in AVAILABLE_MODELS and cannot be resolved.
        FileNotFoundError: If config.yaml is missing in the checkpoint folder.
    """
    custom_checkpoint = None
    custom_model_path = None
    if modelname is not None:
        candidate = Path(str(modelname)).expanduser()
        if candidate.is_file() and candidate.suffix == ".safetensors":
            custom_checkpoint = candidate.resolve()
            custom_model_path = custom_checkpoint.parent
        elif candidate.is_dir() and (candidate / "config.yaml").is_file():
            custom_model_path = candidate.resolve()

    if modelname is None:
        modelname = DEFAULT_MODEL
    if custom_model_path is None and modelname not in AVAILABLE_MODELS:
        if default_family is not None:
            modelname = resolve_model_name(modelname, default_family)
        else:
            raise ValueError(
                f"""The model is not recognized.
            Please choose between: {AVAILABLE_MODELS}"""
            )

    resolved_modelname = custom_checkpoint.name if custom_checkpoint else (
        custom_model_path.name if custom_model_path else modelname
    )

    # In case, we specify a custom checkpoint directory
    configured_checkpoint_dir = get_env_var("CHECKPOINT_DIR")
    if custom_model_path is not None:
        model_path = custom_model_path
    elif configured_checkpoint_dir:
        print(f"CHECKPOINT_DIR is set to {configured_checkpoint_dir}, checking the local cache...")
        # Checkpoint folders are named by display name (e.g. Kimodo-SOMA-RP-v1)
        info = get_model_info(modelname)
        checkpoint_folder_name = info.display_name if info is not None else modelname
        model_path = Path(configured_checkpoint_dir) / checkpoint_folder_name
        if not model_path.exists() and modelname != checkpoint_folder_name:
            # Fallback: try short_key for backward compatibility
            model_path = Path(configured_checkpoint_dir) / modelname
        if not model_path.exists():
            print(f"Model folder not found at '{model_path}', downloading it from Hugging Face...")
            model_path = _resolve_hf_model_path(modelname)
    else:
        # Otherwise, we load the model from the local cache or download it from Hugging Face.
        model_path = _resolve_hf_model_path(modelname)

    model_config_path = model_path / "config.yaml"
    if not model_config_path.exists():
        raise FileNotFoundError(f"The model checkpoint folder exists but config.yaml is missing: {model_config_path}")

    model_conf = OmegaConf.load(model_config_path)

    provenance_path = model_path / "text-model.json"
    if text_encoder_name is None and provenance_path.is_file():
        import json

        text_encoder_name = json.loads(provenance_path.read_text(encoding="utf-8")).get("preset")
    if text_encoder_name is None and custom_checkpoint is not None:
        embedded_provenance = read_checkpoint_text_model(custom_checkpoint)
        if embedded_provenance:
            text_encoder_name = embedded_provenance.get("preset")

    if custom_checkpoint is not None:
        with open_dict(model_conf):
            model_conf.denoiser.ckpt_path = str(custom_checkpoint)

    if custom_model_path is None and modelname in TMR_MODELS:
        # Same process at the moment for TMR and Kimodo
        pass

    if text_encoder is not None:
        runtime_conf = OmegaConf.create({"checkpoint_dir": str(model_path)})
    else:
        text_encoder_url = get_env_var("TEXT_ENCODER_URL", DEFAULT_TEXT_ENCODER_URL)
        runtime_conf = OmegaConf.create(
            {
                "checkpoint_dir": str(model_path),
                "text_encoder": _select_text_encoder_conf(
                    text_encoder_url, text_encoder_fp32, text_encoder_name
                ),
            }
        )

    model_cfg = OmegaConf.to_container(OmegaConf.merge(model_conf, runtime_conf), resolve=True)
    model_cfg.pop("checkpoint_dir", None)

    expected_dim = None
    if "denoiser" in model_cfg:
        expected_dim = int(model_cfg["denoiser"]["llm_shape"][-1])
    elif model_cfg.get("llm_shape") is not None:
        expected_dim = int(model_cfg["llm_shape"][-1])
    if text_encoder is None and expected_dim is not None:
        encoder_conf = model_cfg.get("text_encoder", {})
        encoder_target = encoder_conf.get("_target_")
        actual_dim = 1024 if encoder_target == "kimodo.model.BGEM3Encoder" else (
            int(encoder_conf.get("llm_dim", 4096)) if "LLM2VecEncoder" in str(encoder_target) else None
        )
        if actual_dim is not None and actual_dim != expected_dim:
            raise ValueError(
                f"Text encoder outputs {actual_dim} dimensions but checkpoint expects {expected_dim}. "
                "Use the text model recorded in text-model.json."
            )
    elif text_encoder is not None and expected_dim is not None:
        actual_dim = getattr(text_encoder, "embedding_dim", getattr(text_encoder, "llm_dim", None))
        if actual_dim is not None and int(actual_dim) != expected_dim:
            raise ValueError(f"Provided text encoder outputs {actual_dim} dimensions; model expects {expected_dim}")

    if text_encoder is not None:
        # Prevent Hydra from instantiating a new text encoder; pass None so
        # Kimodo.__init__ receives a placeholder we replace immediately after.
        model_cfg["text_encoder"] = None

    model = instantiate_from_dict(model_cfg, overrides={"device": device})

    if text_encoder is not None:
        model.text_encoder = text_encoder

    if eval_mode:
        model = model.eval()
    if return_resolved_name:
        return model, resolved_modelname
    return model
