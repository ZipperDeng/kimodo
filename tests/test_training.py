import math
import random
import unittest
import tempfile
import json
from pathlib import Path

import numpy as np
import torch

from kimodo.motion_rep.feature_utils import RotateFeatures
from kimodo.training.constraints import sample_constraints
from kimodo.training.data import (
    MotionDataset,
    collate_motion_batch,
    read_release_manifest,
    validate_dataset_contract,
)
from kimodo.training.optimizer import AdamAtan2
from kimodo.training.state import (
    ExponentialMovingAverage,
    compute_gradient_accumulation,
    set_dropout_probability,
)
from kimodo.training.logging import ExperimentLogger
from omegaconf import OmegaConf
from kimodo.training.stats import RunningMoments, compute_motion_stats, save_motion_stats
from kimodo.motion_rep.smooth_root import get_smooth_root_pos, smooth_signal, smooth_signals
from kimodo.training.text_vectors import TextVectorStore
from kimodo.geometry import matrix_to_quaternion, quaternion_to_matrix


def _write_store(variant, motions):
    """Write an AssetCMS-style mmap motion store from ``[(local_rot_mats, root_positions)]``."""
    folder = variant / "motions"
    folder.mkdir(parents=True)
    records, offset = [], 0
    with (folder / "rotations.f32").open("wb") as rotations, (folder / "roots.f32").open("wb") as roots:
        for matrices, positions in motions:
            matrix_to_quaternion(torch.as_tensor(matrices)).numpy().astype("<f4").tofile(rotations)
            np.asarray(positions, dtype="<f4").tofile(roots)
            records.append({"source": f"test:{len(records)}", "offset": offset, "frames": len(positions)})
            offset += len(positions)
    (folder / "index.json").write_text(json.dumps({
        "schema_version": 1, "format": "kimodo-motion-mmap-v1", "skeleton": variant.name,
        "joints": motions[0][0].shape[1], "count": len(records), "frames": offset,
        "dtype": "float32", "byte_order": "little", "rotation": "quaternion_wxyz",
        "rotations": "rotations.f32", "roots": "roots.f32", "motions": records,
    }), encoding="utf-8")


def _random_rotations(frames, joints, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return quaternion_to_matrix(torch.randn(frames, joints, 4, generator=generator)).numpy()


def _make_release(cms, rows, skeleton="somaskel30"):
    """Create ``library/releases/test`` with one train manifest; returns the manifest path."""
    release = cms / "library" / "releases" / "test"
    variant = release / "skeletons" / skeleton
    variant.mkdir(parents=True)
    (release / "manifest.json").write_text(json.dumps({
        "schema_version": 2, "release": "test",
        "selection": {"datasets": None, "splits": None, "skeleton": skeleton},
    }), encoding="utf-8")
    manifest = variant / "train.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return manifest


class _Skeleton:
    left_foot_joint_idx = [1]
    right_foot_joint_idx = [2]
    left_hand_joint_names = ["LeftHand"]
    right_hand_joint_names = ["RightHand"]
    bone_index = {"LeftHand": 3, "RightHand": 4}


class _MotionRep:
    nbjoints = 5
    skeleton = _Skeleton()
    slice_dict = {
        "smooth_root_pos": slice(0, 3),
        "global_root_heading": slice(3, 5),
        "local_joints_positions": slice(5, 20),
        "global_rot_data": slice(20, 50),
        "foot_contacts": slice(50, 54),
    }


class RotateFeaturesTests(unittest.TestCase):
    def test_batch_angles_broadcast_over_time_and_joints(self):
        rotations = RotateFeatures(torch.tensor([0.0, math.pi / 2]))
        positions = torch.tensor([1.0, 0.0, 0.0]).expand(2, 3, 4, 3).clone()
        rotated = rotations.rotate_positions(positions)

        self.assertEqual(rotated.shape, positions.shape)
        torch.testing.assert_close(rotated[0], positions[0])
        torch.testing.assert_close(
            rotated[1],
            torch.tensor([0.0, 0.0, -1.0]).expand(3, 4, 3),
            atol=1e-6,
            rtol=0,
        )

    def test_batch_angles_broadcast_over_2d_time(self):
        rotations = RotateFeatures(torch.tensor([0.0, math.pi / 2]))
        positions = torch.tensor([1.0, 0.0]).expand(2, 3, 2).clone()
        rotated = rotations.rotate_2d_positions(positions)

        self.assertEqual(rotated.shape, positions.shape)
        torch.testing.assert_close(rotated[0], positions[0])
        torch.testing.assert_close(
            rotated[1],
            torch.tensor([0.0, 1.0]).expand(3, 2),
            atol=1e-6,
            rtol=0,
        )


class AdamAtan2Tests(unittest.TestCase):
    def test_first_step_matches_reference_formula(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
        optimizer = AdamAtan2([parameter], lr=2e-5)
        parameter.grad = torch.tensor([1.0], dtype=torch.float64)
        optimizer.step()
        expected_update = (2e-5 / 0.1) * math.atan2(0.1, 1.0)
        self.assertAlmostEqual(parameter.item(), 1.0 - expected_update, places=12)

    def test_first_step_is_gradient_scale_invariant(self):
        values = []
        for gradient in (1.0, 1e-12):
            parameter = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
            optimizer = AdamAtan2([parameter], lr=2e-5)
            parameter.grad = torch.tensor([gradient], dtype=torch.float64)
            optimizer.step()
            values.append(parameter.item())
        self.assertAlmostEqual(values[0], values[1], places=12)


class CurriculumTests(unittest.TestCase):
    def setUp(self):
        random.seed(0)
        torch.manual_seed(0)
        self.motion = torch.zeros(8, 30, 54)
        self.lengths = torch.full((8,), 30)

    def test_no_constraint_probability_can_disable_all_constraints(self):
        _, mask = sample_constraints(
            self.motion,
            self.lengths,
            _MotionRep(),
            no_constraint_probability=1.0,
        )
        self.assertFalse(mask.any())

    def test_curriculum_produces_valid_mask(self):
        observed, mask = sample_constraints(
            self.motion,
            self.lengths,
            _MotionRep(),
            curriculum_progress=0.5,
            no_constraint_probability=0.0,
            mixed_constraint_probability=1.0,
        )
        self.assertEqual(observed.shape, self.motion.shape)
        self.assertEqual(mask.shape, self.motion.shape)
        self.assertTrue(mask.any())


class TrainingStateTests(unittest.TestCase):
    def test_gradient_accumulation_is_derived_from_global_batch(self):
        self.assertEqual(compute_gradient_accumulation(16, 4, 512), 8)
        self.assertEqual(compute_gradient_accumulation(16, 1, 512), 32)

    def test_gradient_accumulation_requires_exact_global_batch(self):
        with self.assertRaisesRegex(ValueError, "not divisible"):
            compute_gradient_accumulation(24, 4, 512)
        with self.assertRaisesRegex(ValueError, "smaller"):
            compute_gradient_accumulation(32, 4, 64)

    def test_dropout_and_ema(self):
        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout(0.1))
        ema = ExponentialMovingAverage(model, decay=0.5)
        with torch.no_grad():
            model[0].weight.add_(2.0)
        ema.update(model)
        averaged = ema.averaged_state_dict(model)
        self.assertTrue(torch.allclose(averaged["0.weight"], model[0].weight.detach().cpu() - 1.0))
        set_dropout_probability(model, 0.0)
        self.assertEqual(model[1].p, 0.0)

    def test_ema_context_restores_raw_parameters(self):
        model = torch.nn.Linear(2, 2)
        ema = ExponentialMovingAverage(model, decay=0.5)
        raw = model.weight.detach().clone()
        with torch.no_grad():
            model.weight.add_(2.0)
        changed = model.weight.detach().clone()
        with ema.apply_to(model):
            self.assertTrue(torch.allclose(model.weight, raw))
        self.assertTrue(torch.allclose(model.weight, changed))

    def test_running_moments_respects_valid_mask(self):
        moments = RunningMoments(2)
        values = torch.tensor([[[1.0, 2.0], [100.0, 100.0]], [[3.0, 4.0], [5.0, 6.0]]])
        valid = torch.tensor([[True, False], [True, True]])
        moments.update(values, valid)
        mean, std = moments.compute()
        expected = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        self.assertTrue(torch.allclose(mean, expected.mean(0)))
        self.assertTrue(torch.allclose(std, expected.std(0, correction=0)))

    def test_running_moments_merges_worker_partials(self):
        values = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float64)
        moments = RunningMoments(2)
        for chunk in values.split((1, 2)):
            moments.merge(len(chunk), chunk.sum(0), chunk.square().sum(0))
        mean, std = moments.compute()
        torch.testing.assert_close(mean, values.float().mean(0))
        torch.testing.assert_close(std, values.float().std(0, correction=0))


class ReleaseDatasetTests(unittest.TestCase):
    def test_published_mmap_text_vectors_are_loaded(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            snapshot = root / "training-vectors" / "llm2vec"
            snapshot.mkdir(parents=True)
            key = "a" * 64
            values = np.arange(8, dtype=np.float16).reshape(1, 8)
            values.tofile(snapshot / "vectors.f16")
            (snapshot / "keys.txt").write_text(key + "\n", encoding="ascii")
            (snapshot / "metadata.json").write_text(json.dumps({
                "schema_version": 1,
                "format": "kimodo-text-vectors-mmap-v1",
                "encoder": "llm2vec",
                "count": 1,
                "dimension": 8,
                "dtype": "float16",
                "vectors": "vectors.f16",
                "keys": "keys.txt",
            }), encoding="utf-8")

            store = TextVectorStore(root, "llm2vec")
            np.testing.assert_array_equal(store.get(key), values.astype(np.float32))
            store.close()

    def test_dataset_contract_rejects_wrong_skeleton(self):
        class Skeleton:
            name = "smplx22"
            nbjoints = 22

        with tempfile.TemporaryDirectory() as folder:
            release = _make_release(Path(folder), [], skeleton="smplx22").parents[2]
            (release / "skeletons" / "smplx22" / "source.json").write_text(
                json.dumps({"skeleton": "somaskel30", "joint_count": 30, "fps": 30}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "skeleton"):
                validate_dataset_contract(release, Skeleton(), 30)

    def test_dataset_contract_rejects_non_release_root(self):
        class Skeleton:
            name = "somaskel30"
            nbjoints = 30

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "skeletons" / "somaskel30").mkdir(parents=True)
            with self.assertRaisesRegex(FileNotFoundError, "not an AssetCMS release"):
                validate_dataset_contract(root, Skeleton(), 30)
            with self.assertRaisesRegex(FileNotFoundError, "not an AssetCMS release"):
                read_release_manifest(root)

    def test_smplx_training_config_uses_22_joint_representation(self):
        from hydra.utils import instantiate

        cfg = OmegaConf.load("configs/train_smplx_smoke.yaml")
        motion_rep = instantiate(cfg.model.denoiser.motion_rep)
        self.assertEqual(motion_rep.skeleton.name, "smplx22")
        self.assertEqual(motion_rep.nbjoints, 22)
        self.assertEqual(motion_rep.motion_rep_dim, 273)

    def test_self_contained_release_reads_motion_store_slices(self):
        with tempfile.TemporaryDirectory() as folder:
            first = (_random_rotations(4, 3, seed=1), np.arange(12, dtype=np.float32).reshape(4, 3))
            second = (_random_rotations(5, 3, seed=2), np.arange(15, dtype=np.float32).reshape(5, 3) + 100)
            manifest = _make_release(Path(folder), [
                {"sample_uid": "u", "name": "a", "motion_index": 0, "text_vector_keys": ["k"]},
                {"name": "b_event", "motion_index": 1, "start_frame": 1, "end_frame": 3, "text_vector_keys": ["k"]},
            ])
            _write_store(manifest.parent, [first, second])
            release = manifest.parents[2]

            self.assertEqual(read_release_manifest(release)["release"], "test")
            dataset = MotionDataset(manifest, load_text=False, expected_joints=3)
            whole, event = dataset[0], dataset[1]
            torch.testing.assert_close(whole["local_rot_mats"], torch.from_numpy(first[0]), atol=1e-6, rtol=0)
            self.assertTrue(torch.equal(whole["root_positions"], torch.from_numpy(first[1])))
            torch.testing.assert_close(event["local_rot_mats"], torch.from_numpy(second[0][1:3]), atol=1e-6, rtol=0)
            self.assertTrue(torch.equal(event["root_positions"], torch.from_numpy(second[1][1:3])))
            self.assertNotIn("text_feat", event)
            batch = collate_motion_batch([event], max_frames=300)
            self.assertIsNone(batch[4])
            self.assertIsNone(batch[5])
            dataset.close()

            class Skeleton:
                name = "smplx22"
                nbjoints = 22

            with self.assertRaisesRegex(ValueError, "published for skeleton"):
                validate_dataset_contract(release, Skeleton(), 30)

    def test_training_dataset_rejects_manifest_without_embedding_keys(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest = _make_release(Path(folder), [{"motion_index": 0, "name": "sample"}])
            _write_store(manifest.parent, [(_random_rotations(2, 1), np.zeros((2, 3)))])
            (manifest.parents[2] / "motion-text-index.json").write_text(json.dumps({
                "schema_version": 1,
                "encoders": {"llm2vec": {"embedding_dim": 4096, "vector_name": "text_llm2vec"}},
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no text_vector_keys"):
                MotionDataset(manifest)

    def test_dataset_rejects_invalid_rows_before_training(self):
        cases = [
            ({"motion": "motions/seed/train/a.npz"}, "no valid motion_index"),
            ({"motion_index": 1}, "no valid motion_index"),
            ({"motion_index": 0, "start_frame": 1, "end_frame": 5}, "invalid frame range"),
        ]
        for row, message in cases:
            with self.subTest(row=row), tempfile.TemporaryDirectory() as folder:
                manifest = _make_release(Path(folder), [{**row, "name": "a", "text_vector_keys": ["k"]}])
                _write_store(manifest.parent, [(_random_rotations(4, 2), np.zeros((4, 3)))])
                with self.assertRaisesRegex(ValueError, message):
                    MotionDataset(manifest, load_text=False)
        with tempfile.TemporaryDirectory() as folder:
            manifest = _make_release(Path(folder), [{"motion_index": 0, "name": "a"}])
            _write_store(manifest.parent, [(_random_rotations(4, 2), np.zeros((4, 3)))])
            with self.assertRaisesRegex(ValueError, "configured skeleton requires 30"):
                MotionDataset(manifest, load_text=False, expected_joints=30)
            (manifest.parent / "motions" / "roots.f32").write_bytes(b"\0" * 8)
            with self.assertRaisesRegex(ValueError, "Incomplete motion store"):
                MotionDataset(manifest, load_text=False)

class SmoothRootTests(unittest.TestCase):
    def test_batched_smoother_matches_reference_per_signal(self):
        rng = np.random.default_rng(0)
        lengths = (6, 16, 31, 33, 64, 100, 257, 300)
        signals = [np.cumsum(rng.normal(0, 0.03, (n, 2)), axis=0).astype(np.float32) for n in lengths]
        batched = smooth_signals(signals)
        for signal, result in zip(signals, batched):
            expected = smooth_signal(signal, np.full(len(signal), 0.06))
            np.testing.assert_allclose(result, expected, atol=1e-5, rtol=0)

    def test_padding_does_not_change_smooth_root(self):
        rng = np.random.default_rng(1)
        short = torch.from_numpy(np.cumsum(rng.normal(0, 0.03, (40, 3)), axis=0).astype(np.float32))
        padded = torch.zeros(2, 90, 3)
        padded[0, :40] = short
        padded[1] = torch.from_numpy(np.cumsum(rng.normal(0, 0.03, (90, 3)), axis=0).astype(np.float32))
        batched = get_smooth_root_pos(padded, torch.tensor([40, 90]))
        torch.testing.assert_close(batched[0, :40], get_smooth_root_pos(short[None])[0])
        torch.testing.assert_close(batched[0, 40:], padded[0, 40:])
        torch.testing.assert_close(batched[1], get_smooth_root_pos(padded[1:])[0])

    def test_short_clips_skip_admm_but_keep_reference_final_pass(self):
        # Two points have no acceleration term, so the level keeps its targets;
        # smooth_signal's final extrapolate-and-average pass still applies.
        signal = np.array([[0.0, 0.0], [0.5, 0.1]], dtype=np.float32)
        np.testing.assert_allclose(smooth_signals([signal])[0], [[0.375, 0.075], [0.75, 0.15]], atol=1e-6)


class PaddedBatchTests(unittest.TestCase):
    def test_padded_batch_stays_finite_through_training_transform(self):
        from hydra.utils import instantiate

        cfg = OmegaConf.load("configs/train_smoke.yaml")
        rep_cfg = OmegaConf.to_container(cfg.model.denoiser.motion_rep, resolve=False)
        rep_cfg["stats_path"] = None
        motion_rep = instantiate(rep_cfg)
        joints = motion_rep.nbjoints
        samples = [
            {
                "local_rot_mats": torch.from_numpy(_random_rotations(n, joints, seed=n)),
                "root_positions": torch.cumsum(torch.full((n, 3), 0.01), dim=0),
            }
            for n in (8, 20)
        ]
        rotations, roots, lengths, valid, _text, _text_mask, smooth = collate_motion_batch(samples, max_frames=300)
        self.assertTrue(torch.equal(rotations[0, 8:], torch.eye(3).expand(12, joints, 3, 3)))
        with torch.no_grad():
            features = motion_rep(rotations, roots, to_normalize=False, lengths=lengths, smooth_root_pos=smooth)
            features = motion_rep.translate_2d_to_zero(features)
            features = motion_rep.rotate_to(features, torch.tensor([0.3, 2.0]))
        self.assertTrue(torch.isfinite(features).all())


class MotionStatsTests(unittest.TestCase):
    def test_stats_are_saved_in_motion_rep_layout_and_reproducible(self):
        from hydra.utils import instantiate

        cfg = OmegaConf.load("configs/train_smoke.yaml")
        rep_cfg = OmegaConf.to_container(cfg.model.denoiser.motion_rep, resolve=False)
        rep_cfg["stats_path"] = None
        motion_rep = instantiate(rep_cfg)
        joints = motion_rep.nbjoints
        with tempfile.TemporaryDirectory() as folder:
            motions = [
                (_random_rotations(n, joints, seed=n), np.cumsum(np.full((n, 3), 0.01, np.float32), axis=0))
                for n in (12, 20, 33)
            ]
            manifest = _make_release(Path(folder), [
                {"name": f"m{i}", "motion_index": i, "text_vector_keys": ["k"]} for i in range(3)
            ])
            _write_store(manifest.parent, motions)
            dataset = MotionDataset(manifest, load_text=False, expected_joints=joints)
            try:
                kwargs = dict(max_samples=3, batch_size=2, num_workers=0, max_frames=16, seed=7)
                rng_before = torch.get_rng_state()
                first = compute_motion_stats(dataset, motion_rep, torch.device("cpu"), **kwargs)
                self.assertTrue(torch.equal(torch.get_rng_state(), rng_before))
                second = compute_motion_stats(dataset, motion_rep, torch.device("cpu"), **kwargs)
                for name in ("global_root", "local_root", "body"):
                    torch.testing.assert_close(first[name].compute(), second[name].compute())
                self.assertEqual(first["body"].count, 12 + 16 + 16)
                stats_dir = Path(folder) / "out" / "stats" / "motion"
                save_motion_stats(stats_dir, first)
                rep_cfg["stats_path"] = str(stats_dir)
                loaded = instantiate(rep_cfg)
                self.assertEqual(loaded.body_stats.mean.shape[-1], motion_rep.body_dim)
            finally:
                dataset.close()


class ProgressReporterTests(unittest.TestCase):
    def test_format_duration(self):
        from kimodo.training.logging import format_duration

        self.assertEqual(format_duration(3725), "01:02:05")
        self.assertEqual(format_duration(90061), "1d 01:01:01")

    def test_eta_skips_startup_interval_and_lines_are_readable(self):
        import io
        from unittest import mock

        from kimodo.training.logging import ProgressReporter

        clock = iter([0.0, 30.0, 32.0, 34.0])  # 30 s startup, then 2 s per 10 steps
        with mock.patch("kimodo.training.logging.time.perf_counter", lambda: next(clock)):
            stream = io.StringIO()
            reporter = ProgressReporter(total_steps=100, start_step=0, global_batch_size=8, stream=stream)
            first = reporter.timing(10)
            second = reporter.timing(20)
            third = reporter.timing(30)
        self.assertIsNone(first["eta_seconds"])
        self.assertAlmostEqual(second["eta_seconds"], 80 * 0.2)
        self.assertAlmostEqual(third["eta_seconds"], 70 * 0.2)
        self.assertAlmostEqual(third["samples_per_second"], 10 * 8 / 2)
        for step, timing in ((10, first), (30, third)):
            metrics = {
                **timing, "loss": 0.5, "gradient_norm": 1.2, "learning_rate": 2e-5, "phase": 1,
                "data_wait_fraction": 0.05, "components": {"smooth_root_pos": 0.1, "fk": 0.02},
            }
            reporter.step(step, metrics)
            json.dumps({k: v for k, v in metrics.items() if k != "components"}, allow_nan=False)
        lines = stream.getvalue().splitlines()
        self.assertIn("step  10/100  10.0% P1", lines[0])
        self.assertIn("ETA estimating", lines[0])
        self.assertIn("ETA 00:00:14", lines[2])
        self.assertIn("root 0.1000  fk 0.0200", lines[3])


class LoggingTests(unittest.TestCase):
    def test_jsonl_logger_appends_resume_events(self):
        cfg = OmegaConf.create({"training": {"logging": {"backends": ["jsonl"]}}})
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            logger = ExperimentLogger(output, cfg)
            logger.log({"loss": torch.tensor(1.25)}, 1)
            logger.close()
            logger = ExperimentLogger(output, cfg, resume=True)
            logger.log({"loss": 1.0}, 2, event="validation")
            logger.close()
            records = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
            self.assertEqual([record["batch_step"] for record in records], [1, 2])
            self.assertEqual(records[1]["event"], "validation")


if __name__ == "__main__":
    unittest.main()
