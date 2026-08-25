"""Training state helpers: EMA, RNG snapshots, and phase-dependent dropout."""

import copy
import random

import numpy as np
import torch
from torch import nn
from contextlib import contextmanager


def compute_gradient_accumulation(
    per_device_batch_size: int,
    world_size: int,
    expected_global_batch_size: int,
) -> int:
    """Derive micro-batch accumulation needed for one global batch step."""
    per_device_batch_size = int(per_device_batch_size)
    world_size = int(world_size)
    expected_global_batch_size = int(expected_global_batch_size)
    if per_device_batch_size < 1:
        raise ValueError("training.batch_size must be at least one")
    if world_size < 1:
        raise ValueError("WORLD_SIZE must be at least one")
    if expected_global_batch_size < 1:
        raise ValueError("training.expected_global_batch_size must be at least one")
    global_micro_batch = per_device_batch_size * world_size
    if expected_global_batch_size < global_micro_batch:
        raise ValueError(
            f"Target global batch {expected_global_batch_size} is smaller than one distributed "
            f"micro-batch ({per_device_batch_size} per GPU x {world_size} processes = "
            f"{global_micro_batch}). Reduce training.batch_size or increase the target."
        )
    if expected_global_batch_size % global_micro_batch:
        raise ValueError(
            f"Target global batch {expected_global_batch_size} is not divisible by one distributed "
            f"micro-batch ({global_micro_batch}). Choose values where "
            "expected_global_batch_size % (batch_size * WORLD_SIZE) == 0."
        )
    return expected_global_batch_size // global_micro_batch


class ExponentialMovingAverage:
    """Maintain an fp32 exponential moving average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.995):
        if not 0.0 < decay < 1.0:
            raise ValueError(f"EMA decay must be in (0, 1), got {decay}")
        self.decay = float(decay)
        self.shadow = {
            name: parameter.detach().float().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, parameter in model.named_parameters():
            if name not in self.shadow:
                continue
            source = parameter.detach().float()
            if self.shadow[name].device != source.device:
                self.shadow[name] = self.shadow[name].to(source.device)
            self.shadow[name].lerp_(source, 1.0 - self.decay)

    def state_dict(self) -> dict:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state: dict, model: nn.Module = None) -> None:
        self.decay = float(state["decay"])
        devices = {name: parameter.device for name, parameter in model.named_parameters()} if model is not None else {}
        self.shadow = {
            name: value.detach().float().to(devices.get(name, value.device)).clone()
            for name, value in state["shadow"].items()
        }

    def averaged_state_dict(self, model: nn.Module) -> dict:
        """Return a complete model state dict with EMA parameters substituted."""
        state = copy.deepcopy(model.state_dict())
        for name, value in self.shadow.items():
            state[name] = value.to(device=state[name].device, dtype=state[name].dtype)
        return state

    @contextmanager
    def apply_to(self, model: nn.Module):
        """Temporarily evaluate a model with EMA parameters, then restore it."""
        backup = {}
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name in self.shadow:
                    backup[name] = parameter.detach().clone()
                    parameter.copy_(self.shadow[name].to(parameter))
        try:
            yield model
        finally:
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if name in backup:
                        parameter.copy_(backup[name])


def set_dropout_probability(model: nn.Module, probability: float) -> None:
    """Set all dropout modules, including Transformer attention dropout."""
    probability = float(probability)
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = probability
        elif isinstance(module, nn.MultiheadAttention):
            module.dropout = probability


def capture_rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
