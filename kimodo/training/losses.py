"""Component-wise Kimodo reconstruction objective."""

import torch
from torch import nn


class KimodoLoss(nn.Module):
    weights = {
        "smooth_root_pos": 10.0,
        "global_root_heading": 2.0,
        "local_joints_positions": 10.0,
        "velocities": 3.0,
        "global_rot_data": 10.0,
        "foot_contacts": 4.0,
    }

    def __init__(self, motion_rep, fk_weight=5.0):
        super().__init__()
        self.motion_rep = motion_rep
        self.fk_weight = fk_weight

    @staticmethod
    def _masked_smooth_l1(pred, target, valid):
        loss = torch.nn.functional.smooth_l1_loss(pred, target, reduction="none")
        valid = valid.to(loss.dtype)
        while valid.ndim < loss.ndim:
            valid = valid.unsqueeze(-1)
        return (loss * valid).sum() / (valid.sum() * loss.shape[-1]).clamp_min(1)

    def forward(self, pred, target, valid):
        losses = {}
        for name, weight in self.weights.items():
            feature = self.motion_rep.slice_dict[name]
            losses[name] = self._masked_smooth_l1(pred[..., feature], target[..., feature], valid)
        pred_world = self.motion_rep.inverse(pred, is_normalized=True, posed_joints_from="rotations")["posed_joints"]
        target_world = self.motion_rep.inverse(target, is_normalized=True, posed_joints_from="positions")["posed_joints"]
        losses["fk"] = self._masked_smooth_l1(pred_world.flatten(-2), target_world.flatten(-2), valid)
        total = sum(self.weights[name] * losses[name] for name in self.weights) + self.fk_weight * losses["fk"]
        return total, losses
