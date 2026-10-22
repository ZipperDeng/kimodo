import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import torch
from safetensors.torch import save_file

from kimodo.benchmark_job import JobLock, build_commands, create_manifest, run_job


def _fixture(folder):
    root = Path(folder)
    suite = root / "suite"
    (suite / "case").mkdir(parents=True)
    (suite / "case/meta.json").write_text('{"text": "walk"}', encoding="utf-8")
    checkpoint = root / "model-0000001.safetensors"
    save_file({"weight": torch.ones(1)}, str(checkpoint))
    config = {
        "job_dir": str(root / "job"),
        "suite": str(suite),
        "model": {"checkpoint": str(checkpoint)},
        "generation": {}, "embedding": {}, "evaluation": {}, "summary": {},
        "publish": {"wandb": {"enabled": False}},
    }
    commands = {stage: ["python", stage] for stage in ("generate", "embed", "evaluate", "summarize")}
    return config, commands


class BenchmarkJobTests(unittest.TestCase):
    def test_generator_and_evaluator_text_models_are_tracked_separately(self):
        with tempfile.TemporaryDirectory() as folder:
            config, _ = _fixture(folder)
            config["generation"]["text_model"] = "bge-m3"
            config["embedding"].update({"text_model": "llm2vec", "model": "tmr-soma-rp"})
            checkpoint = Path(config["model"]["checkpoint"])
            (checkpoint.parent / "text-model.json").write_text(
                json.dumps({"preset": "bge-m3", "revision": "commit", "embedding_dim": 1024}),
                encoding="utf-8",
            )
            commands = build_commands(config, Path.cwd())
            self.assertIn("bge-m3", commands["generate"])
            self.assertIn("llm2vec", commands["embed"])
            manifest = create_manifest(config, commands)
            self.assertEqual(manifest["generation_text_model"]["revision"], "commit")
            self.assertEqual(manifest["evaluator_text_model"]["preset"], "llm2vec")

    def test_lock_is_exclusive_and_released(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / ".lock"
            with JobLock(path):
                with self.assertRaises(RuntimeError):
                    with JobLock(path):
                        pass
            self.assertFalse(path.exists())

    def test_completed_job_is_idempotent_on_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            config, commands = _fixture(folder)
            with mock.patch("kimodo.benchmark_job.build_commands", return_value=commands), mock.patch(
                "kimodo.benchmark_job.run_command", return_value=Path(folder) / "stage.log"
            ) as runner, mock.patch("kimodo.benchmark_job.validate_stage", return_value={"ok": 1}):
                state = run_job(config)
                self.assertEqual(state["status"], "COMPLETED")
                self.assertEqual(runner.call_count, 4)
                runner.reset_mock()
                state = run_job(config, resume=True)
                self.assertEqual(state["status"], "COMPLETED")
                runner.assert_not_called()

    def test_failed_stage_resumes_without_repeating_completed_stage(self):
        with tempfile.TemporaryDirectory() as folder:
            config, commands = _fixture(folder)

            def fail_embed(stage, *_args, **_kwargs):
                if stage == "embed":
                    raise subprocess.CalledProcessError(7, ["embed"])
                return Path(folder) / f"{stage}.log"

            with mock.patch("kimodo.benchmark_job.build_commands", return_value=commands), mock.patch(
                "kimodo.benchmark_job.run_command", side_effect=fail_embed
            ), mock.patch("kimodo.benchmark_job.validate_stage", return_value={}):
                with self.assertRaises(subprocess.CalledProcessError):
                    run_job(config)
            state_path = Path(config["job_dir"]) / "job-state.json"
            failed = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(failed["status"], "FAILED")
            self.assertEqual(failed["stages"]["generate"]["status"], "COMPLETED")
            self.assertEqual(failed["stages"]["embed"]["status"], "FAILED")

            with mock.patch("kimodo.benchmark_job.build_commands", return_value=commands), mock.patch(
                "kimodo.benchmark_job.run_command", return_value=Path(folder) / "ok.log"
            ) as runner, mock.patch("kimodo.benchmark_job.validate_stage", return_value={}):
                completed = run_job(config, resume=True)
            self.assertEqual(completed["status"], "COMPLETED")
            self.assertEqual([call.args[0] for call in runner.call_args_list], ["embed", "evaluate", "summarize"])
            self.assertEqual(completed["stages"]["embed"]["attempt"], 2)


if __name__ == "__main__":
    unittest.main()
