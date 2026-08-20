"""Random constraint patterns for phase two of Kimodo training."""

import random

import torch


def _sparse_frame_count(length, max_keyframes):
    """Sample a count biased toward sparse constraints, as in the report."""
    upper = max(1, min(int(max_keyframes), int(length)))
    # Squaring a uniform sample puts more mass near one while retaining the
    # full [1, upper] support.
    return 1 + int((upper - 1) * random.random() ** 2)


def _frames(length, dense=False, max_keyframes=20):
    if dense:
        start = random.randrange(length)
        end = random.randrange(start + 1, length + 1)
        return torch.arange(start, end)
    count = _sparse_frame_count(length, max_keyframes)
    return torch.randperm(length)[:count]


def sample_constraints(
    motion,
    lengths,
    motion_rep,
    curriculum_progress=1.0,
    max_sparse_keyframes=20,
    no_constraint_probability=0.10,
    mixed_constraint_probability=0.25,
):
    """Sample the phase-two constraint curriculum from report section 4.3.

    Sparse-keyframe capacity grows linearly from one to
    ``max_sparse_keyframes``.  Per sample, no constraint is used with 10%
    probability and two independently chosen families are mixed with 25%
    probability by default.
    """
    mask = torch.zeros_like(motion, dtype=torch.bool)
    pos = motion_rep.slice_dict["local_joints_positions"]
    rot = motion_rep.slice_dict["global_rot_data"]
    root_pos = motion_rep.slice_dict["smooth_root_pos"]
    heading = motion_rep.slice_dict["global_root_heading"]
    contacts = motion_rep.slice_dict["foot_contacts"]
    joints = motion_rep.nbjoints
    end_effectors = list(dict.fromkeys(
        motion_rep.skeleton.left_foot_joint_idx + motion_rep.skeleton.right_foot_joint_idx +
        [motion_rep.skeleton.bone_index[x] for x in
         motion_rep.skeleton.left_hand_joint_names[:1] + motion_rep.skeleton.right_hand_joint_names[:1]]
    ))
    families = ("full_body", "end_effectors", "root_sparse", "root_path", "contacts")
    curriculum_progress = max(0.0, min(1.0, float(curriculum_progress)))
    current_max_keyframes = 1 + int((max_sparse_keyframes - 1) * curriculum_progress)
    for b, length_tensor in enumerate(lengths):
        length = int(length_tensor)
        if random.random() < no_constraint_probability:
            continue
        num_patterns = 2 if random.random() < mixed_constraint_probability else 1
        for family in random.sample(families, k=num_patterns):
            frames = _frames(
                length,
                dense=family == "root_path",
                max_keyframes=current_max_keyframes,
            ).to(motion.device)
            if family == "full_body":
                mask[b, frames, pos] = True
            elif family == "end_effectors":
                chosen = random.sample(end_effectors, k=random.randint(1, len(end_effectors)))
                for joint in chosen:
                    mask[b, frames, pos.start + 3 * joint : pos.start + 3 * (joint + 1)] = True
                    mask[b, frames, rot.start + 6 * joint : rot.start + 6 * (joint + 1)] = True
            elif family in ("root_sparse", "root_path"):
                mask[b, frames, root_pos.start + 0] = True
                mask[b, frames, root_pos.start + 2] = True
                mask[b, frames, heading] = True
            else:
                mask[b, frames, contacts] = True
    return motion, mask
