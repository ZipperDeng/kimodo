"""Central text-encoder presets and provenance helpers."""

import copy
import json
from pathlib import Path

from omegaconf import OmegaConf


DEFAULT_TEXT_ENCODER = "llm2vec"
TEXT_ENCODER_PRESETS = {
    "llm2vec": {
        "target": "kimodo.model.LLM2VecEncoder",
        "embedding_dim": 4096,
        "kwargs": {
            "base_model_name_or_path": "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
            "peft_model_name_or_path": "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised",
            "dtype": "bfloat16", "llm_dim": 4096, "device": "auto",
        },
    },
    "bge-m3": {
        "target": "kimodo.model.BGEM3Encoder",
        "embedding_dim": 1024,
        "kwargs": {
            "model_name_or_path": "BAAI/bge-m3", "revision": None,
            "dtype": "float16", "device": "auto", "max_length": 256,
            "normalize": True,
        },
    },
}


def build_text_encoder_config(name=DEFAULT_TEXT_ENCODER, fp32=False, **overrides):
    if name not in TEXT_ENCODER_PRESETS:
        raise ValueError(f"Unknown text encoder {name!r}; available: {sorted(TEXT_ENCODER_PRESETS)}")
    preset = copy.deepcopy(TEXT_ENCODER_PRESETS[name])
    kwargs = preset["kwargs"]
    if fp32:
        kwargs["dtype"] = "float32"
    kwargs.update({key: value for key, value in overrides.items() if value is not None})
    return {"_target_": preset["target"], **kwargs}


def text_encoder_provenance_from_config(config, preset=None):
    if config is None:
        return None
    values = OmegaConf.to_container(config, resolve=True) if OmegaConf.is_config(config) else dict(config)
    target = values.get("_target_", values.get("target"))
    if preset is None:
        preset = next((name for name, item in TEXT_ENCODER_PRESETS.items() if item["target"] == target), None)
    if preset == "bge-m3":
        return {
            "preset": preset, "implementation": target,
            "model_id": values.get("model_name_or_path"), "revision": values.get("revision"),
            "embedding_dim": 1024, "pooling": "cls", "normalize": values.get("normalize", True),
            "max_length": values.get("max_length", 256), "dtype": values.get("dtype"),
        }
    if preset == "llm2vec":
        return {
            "preset": preset, "implementation": target,
            "base_model_id": values.get("base_model_name_or_path"),
            "adapter_model_id": values.get("peft_model_name_or_path"),
            "revision": values.get("revision"), "embedding_dim": values.get("llm_dim", 4096),
            "pooling": "llm2vec_encode", "normalize": None, "dtype": values.get("dtype"),
        }
    return {"preset": preset, "implementation": target, "embedding_dim": values.get("embedding_dim")}


def read_checkpoint_text_model(path):
    path = Path(path)
    if not path.is_file() or path.suffix != ".safetensors":
        return None
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as stream:
        value = (stream.metadata() or {}).get("text_model")
    return json.loads(value) if value else None
