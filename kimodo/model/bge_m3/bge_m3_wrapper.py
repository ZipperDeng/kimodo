"""BGE-M3 dense sentence encoder with the Kimodo text-encoder interface."""

import os

import numpy as np
import torch

from .bge_m3 import BGEM3


class BGEM3Encoder:
    """Return normalized BGE-M3 CLS embeddings as one conditioning token."""

    def __init__(
        self,
        model_name_or_path="BAAI/bge-m3",
        revision=None,
        dtype="float16",
        device="auto",
        max_length=256,
        normalize=True,
        trust_remote_code=False,
        local_files_only=False,
    ):
        self.model_name_or_path = model_name_or_path
        self.revision = revision
        self.embedding_dim = 1024
        self.max_length = int(max_length)
        self.normalize = bool(normalize)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if str(device).startswith("cpu") and dtype == "float16":
            dtype = "float32"
        self.dtype_name = dtype
        cache_dir = os.environ.get("HUGGINGFACE_CACHE_DIR")
        torch_dtype = getattr(torch, dtype)
        self.model = BGEM3.from_pretrained(
            model_name_or_path,
            revision=revision,
            cache_dir=cache_dir,
            torch_dtype=torch_dtype,
            max_length=self.max_length,
            normalize=self.normalize,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
        config = self.model.model.config
        self.resolved_revision = getattr(config, "_commit_hash", None) or revision
        if int(config.hidden_size) != self.embedding_dim:
            raise ValueError(
                f"BGE-M3 hidden size is {config.hidden_size}, expected {self.embedding_dim}"
            )
        self._device = str(device)
        self.model.to(device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    @property
    def provenance(self):
        return {
            "preset": "bge-m3",
            "implementation": f"{type(self).__module__}.{type(self).__name__}",
            "model_id": self.model_name_or_path,
            "requested_revision": self.revision,
            "revision": self.resolved_revision,
            "embedding_dim": self.embedding_dim,
            "pooling": "cls",
            "normalize": self.normalize,
            "max_length": self.max_length,
            "dtype": self.dtype_name,
        }

    def to(self, device):
        self._device = str(device)
        self.model.to(device)
        return self

    def eval(self):
        self.model.eval()
        return self

    def get_device(self):
        return self.model.device

    @torch.inference_mode()
    def __call__(self, text):
        is_string = isinstance(text, str)
        texts = [text] if is_string else list(text)
        hidden = self.model.encode(texts, device=self._device)
        features = hidden[:, None]
        lengths = np.ones(len(texts), dtype=int).tolist()
        return (features[0], lengths[0]) if is_string else (features, lengths)
