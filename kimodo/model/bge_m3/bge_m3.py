"""Minimal BGE-M3 dense-embedding model used by the Kimodo wrapper."""

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class BGEM3:
    """Own the Hugging Face tokenizer/model and expose dense text encoding."""

    def __init__(self, model, tokenizer, max_length=256, normalize=True):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.normalize = bool(normalize)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path,
        revision=None,
        torch_dtype=None,
        cache_dir=None,
        max_length=256,
        normalize=True,
        trust_remote_code=False,
        local_files_only=False,
    ):
        common = {
            "revision": revision,
            "cache_dir": cache_dir,
            "trust_remote_code": trust_remote_code,
            "local_files_only": local_files_only,
        }
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **common)
        model = AutoModel.from_pretrained(
            model_name_or_path, torch_dtype=torch_dtype, **common
        )
        return cls(model, tokenizer, max_length=max_length, normalize=normalize)

    def to(self, device):
        self.model.to(device)
        return self

    def eval(self):
        self.model.eval()
        return self

    def parameters(self):
        return self.model.parameters()

    @property
    def device(self):
        return self.model.device

    @torch.inference_mode()
    def encode(self, texts, device):
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(device)
        hidden = self.model(**encoded).last_hidden_state[:, 0]
        if self.normalize:
            hidden = F.normalize(hidden.float(), p=2, dim=-1).to(hidden.dtype)
        return hidden
