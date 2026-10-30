"""Behavior checks for training records; all temporary artifacts stay on D:."""
import csv
import io
import json
import pickle
import sys
import subprocess
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import SimpleITK as sitk
import torch
from torch import nn

import train as training
import utils
import test as evaluation
from checkpoints import load_checkpoint, save_checkpoint
from option import parse_args


TEST_TMP = Path(__file__).resolve().parents[1] / "tmp" / "training_logging"
TEST_TMP.mkdir(parents=True, exist_ok=True)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(.5))

    def forward(self, value, return_aux=False):
        prediction = value * self.weight
        return (prediction, self.weight.square()) if return_aux else prediction


class OneOverflowScaler:
    """Exercise the overflow branch on CPU without requiring CUDA hardware."""
    def __init__(self, *args, **kwargs):
        self.value = 8.
        self.skip = True

    def is_enabled(self):
        return False

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        pass

    def get_scale(self):
        return self.value

    def step(self, optimizer):
        if not self.skip:
            optimizer.step()

    def update(self):
        if self.skip:
            self.value = 4.
            self.skip = False

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        self.skip = False


class TrainingLoggingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=TEST_TMP)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        registry = patch.object(utils, "EXPERIMENT_INDEX", self.root / "experiments.csv", create=True)
        registry.start()
        self.addCleanup(registry.stop)
        self.data = self.root / "data"
        low = np.full((16, 16), 819, dtype=np.float32)  # normalized 0.2
        high = np.tile(np.array([2457, 3276], dtype=np.float32), (16, 8))
        for quality, image in (("LQ", low), ("HQ", high)):
            folder = self.data / "MRI" / "train" / quality
            folder.mkdir(parents=True)
            for index in range(3):
                with (folder / f"Patient_1_{index}.bin").open("wb") as handle:
                    pickle.dump(image, handle)
            folder = self.data / "MRI" / "validation" / quality
            folder.mkdir(parents=True)
            sitk.WriteImage(sitk.GetImageFromArray(image), str(folder / "Patient_2_0.nii"))

    def options(self, directory, total=3, interval=0, resume=""):
        argv = ["--data_path", str(self.data), "--save_dir", str(directory),
                "--modalities", "MRI", "--no_cuda", "--patch_size", "16",
                "--total_iterations", str(total), "--val_interval", str(interval),
                "--lr", "0.01", "--min_lr", "0"]
        if resume:
            argv += ["--resume", str(resume)]
        return parse_args("train", argv)

    def run_training(self, options):
        with patch.object(training, "get_model", TinyModel), redirect_stdout(io.StringIO()):
            return training.train(options)

    def read_rows(self, directory):
        path = directory / "training.csv"
        self.assertTrue(path.is_file(), "training.csv must be saved automatically")
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    def interrupt_after_first_checkpoint(self, directory):
        def interrupt(path, payload):
            save_checkpoint(path, payload)
            if Path(path).name == "last.pth":
                raise RuntimeError("planned interruption")

        with patch.object(training, "save_checkpoint", interrupt):
            with self.assertRaisesRegex(RuntimeError, "planned interruption"):
                self.run_training(self.options(directory, interval=1))

    def test_records_loss_components_used_lr_and_first_periodic_final_steps(self):
        directory = self.root / "run"
        self.run_training(self.options(directory, total=103, interval=1000))
        rows = self.read_rows(directory)
        self.assertEqual([int(row["step"]) for row in rows], [1, 100, 103])
        first = rows[0]
        for key, value in {"epoch": 1, "l1": .6, "balance": .25,
                           "weighted_balance": .00025, "total_loss": .60025,
                           "lr": .01, "grad_norm": .199, "amp_scale": 1,
                           "skipped_batches": 0}.items():
            self.assertAlmostEqual(float(first[key]), value, places=6, msg=key)
        console = directory / "console.log"
        self.assertTrue(console.is_file(), "console.log must be saved automatically")
        self.assertIn("[Training]", console.read_text(encoding="utf-8"))
        self.assertIn("[Done]", console.read_text(encoding="utf-8"))

    def test_resume_discards_rows_ahead_of_checkpoint_and_accepts_legacy_counter(self):
        directory = self.root / "run"
        self.interrupt_after_first_checkpoint(directory)
        rows = self.read_rows(directory)
        with (directory / "training.csv").open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=list(rows[0])).writerow({**rows[0], "step": 100})
        checkpoint = load_checkpoint(directory / "last.pth")
        checkpoint.pop("skipped_batches", None)
        save_checkpoint(directory / "last.pth", checkpoint)
        self.run_training(self.options(directory, interval=1, resume="auto"))
        rows = self.read_rows(directory)
        self.assertEqual([int(row["step"]) for row in rows], [1, 3])
        console = (directory / "console.log").read_text(encoding="utf-8")
        self.assertIn("planned interruption", console)
        self.assertIn("[Done]", console)

    def test_migration_keeps_source_logs_and_copies_history_and_skipped_counter(self):
        source, target = self.root / "source", self.root / "target"
        self.interrupt_after_first_checkpoint(source)
        source_csv = (source / "training.csv").read_bytes() if (source / "training.csv").exists() else b""
        checkpoint = load_checkpoint(source / "last.pth")
        checkpoint["skipped_batches"] = 7
        save_checkpoint(source / "last.pth", checkpoint)
        self.run_training(self.options(target, interval=1, resume=source / "last.pth"))
        rows = self.read_rows(target)
        self.assertEqual([int(row["step"]) for row in rows], [1, 3])
        self.assertEqual(int(rows[-1]["skipped_batches"]), 7)
        self.assertEqual((source / "training.csv").read_bytes(), source_csv)
        self.assertEqual(load_checkpoint(target / "last.pth")["skipped_batches"], 7)
        self.assertIn("planned interruption", (target / "console.log").read_text(encoding="utf-8"))

    def test_overflow_does_not_advance_successful_step_or_scheduler(self):
        directory = self.root / "run"
        with patch.object(training.torch.amp, "GradScaler", OneOverflowScaler):
            self.run_training(self.options(directory, total=2))
        rows = self.read_rows(directory)
        self.assertEqual([int(row["step"]) for row in rows], [1, 2])
        self.assertTrue(all(int(row["skipped_batches"]) == 1 for row in rows))
        self.assertEqual(float(rows[0]["lr"]), .01)
        checkpoint = load_checkpoint(directory / "last.pth")
        self.assertEqual(checkpoint["scheduler"]["last_epoch"], 2)
        self.assertEqual(checkpoint["skipped_batches"], 1)

    def test_rejected_directory_is_not_given_logs(self):
        directory = self.root / "foreign"
        directory.mkdir()
        (directory / "run.json").write_text(json.dumps({"run_id": "another-run"}))
        with self.assertRaises(FileExistsError):
            self.run_training(self.options(directory))
        self.assertFalse((directory / "console.log").exists())
        self.assertFalse((directory / "training.csv").exists())

    def test_existing_unowned_training_log_is_not_appended(self):
        directory = self.root / "unowned"
        directory.mkdir()
        log = directory / "console.log"
        log.write_text("old run\n", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            self.run_training(self.options(directory))
        self.assertEqual(log.read_text(encoding="utf-8"), "old run\n")

    def test_console_captures_buffered_output_stderr_and_exception_and_restores_streams(self):
        capture = getattr(utils, "capture_console", None)
        self.assertTrue(callable(capture), "console capture is not implemented")
        output, errors = io.StringIO(), io.StringIO()
        path = self.root / "console.log"
        with redirect_stdout(output), redirect_stderr(errors):
            with self.assertRaisesRegex(RuntimeError, "captured failure"):
                with capture() as console:
                    print("准备训练")
                    self.assertFalse(path.exists())
                    console.attach(path)
                    print("diagnostic", file=sys.stderr)
                    raise RuntimeError("captured failure")
            self.assertIs(sys.stdout, output)
            self.assertIs(sys.stderr, errors)
        text = path.read_text(encoding="utf-8")
        self.assertIn("准备训练", text)
        self.assertIn("diagnostic", text)
        self.assertIn("RuntimeError: captured failure", text)
        self.assertNotIn("Traceback", errors.getvalue())

    def test_run_tracks_provenance_protocol_lifecycle_resources_and_index(self):
        directory = self.root / "run"
        options = self.options(directory)
        options.experiment_name = "E001"
        options.purpose = "check balance loss"
        options.comparison = "baseline"
        options.notes = "small fixture"
        self.run_training(options)
        run = json.loads((directory / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(run.get("status"), "completed", "run lifecycle must be recorded")
        self.assertEqual(run["experiment"]["purpose"], "check balance loss")
        self.assertEqual(run["sessions"][-1]["status"], "completed")
        self.assertIn("commit", run["provenance"]["git"])
        self.assertIn("train.py", run["provenance"]["core_sources"])
        self.assertEqual(run["provenance"]["environment"]["device"], "cpu")
        self.assertFalse(run["provenance"]["environment"]["amp"])
        self.assertEqual(run["protocol"]["evaluation_ranges"]["CT"], [-160, 240])
        self.assertGreaterEqual(run["resources"]["training_seconds"], 0)
        self.assertGreater(run["resources"]["validation_seconds"], 0)
        self.assertIsNone(run["resources"]["peak_allocated_bytes"])
        event_types = {event["type"] for event in run["events"]}
        self.assertTrue({"started", "validation", "checkpoint_saved", "completed"}.issubset(event_types))
        with (self.root / "experiments.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["run_id"], run["run_id"])
        self.assertEqual(rows[0]["purpose"], "check balance loss")

    def test_validation_replay_is_atomic_complete_and_preserves_session_history(self):
        directory = self.root / "run"
        def fail_before_second_snapshot(path, payload):
            if Path(path).name == "last.pth" and payload["step"] == 2:
                raise RuntimeError("snapshot not written")
            save_checkpoint(path, payload)
        with patch.object(training, "save_checkpoint", fail_before_second_snapshot):
            with self.assertRaisesRegex(RuntimeError, "snapshot not written"):
                self.run_training(self.options(directory, interval=1))
        failed = json.loads((directory / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(failed.get("status"), "failed", "failures must update run metadata")
        self.run_training(self.options(directory, interval=1, resume="auto"))
        run = json.loads((directory / "run.json").read_text(encoding="utf-8"))
        self.assertEqual([session["status"] for session in run["sessions"]], ["failed", "completed"])
        with (directory / "validation.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([int(row["step"]) for row in rows], [1, 2, 3])
        self.assertTrue(all(row["event_id"] and row["session_id"] for row in rows))
        self.assertEqual(load_checkpoint(directory / "last.pth")["validation_event_id"], rows[-1]["event_id"])

    def test_validation_reconciliation_removes_partial_and_replaces_duplicate_events(self):
        reconcile = getattr(utils, "reconcile_validation_csv", None)
        self.assertTrue(callable(reconcile), "validation reconciliation is missing")
        fields = ["step", "modality", "psnr", "ssim", "rmse", "run_id", "session_id", "event_id"]
        path = self.root / "validation.csv"
        old_fields = fields[:5]
        old_rows = [{"step": step, "modality": modality, "psnr": score, "ssim": .9, "rmse": 1}
                    for step, modality, score in ((1, "CT", 20), (1, "MRI", 21),
                                                  (2, "CT", 22), (3, "CT", 23), (3, "MRI", 24))]
        utils.save_summary_csv(path, old_rows, old_fields)
        report = reconcile(path, fields, ["CT", "MRI"], 1)
        self.assertGreaterEqual(report["discarded_groups"], 1)
        self.assertIn(3, report["ahead_steps"])
        new_rows = [{"step": 3, "modality": modality, "psnr": 30, "ssim": .95, "rmse": .5,
                     "run_id": "r", "session_id": "s", "event_id": "new"} for modality in ("CT", "MRI")]
        utils.commit_validation_csv(path, fields, new_rows)
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([int(row["step"]) for row in rows], [1, 1, 3, 3])
        self.assertEqual([row["event_id"] for row in rows[-2:]], ["new", "new"])

    def test_evaluation_binds_checkpoint_protocol_samples_resources_and_index(self):
        directory = self.root / "run"
        self.run_training(self.options(directory))
        args = parse_args("test", ["--model", str(directory / "best.pth"), "--test_path", str(self.data),
                                   "--split", "validation", "--result_dir", str(self.root / "results"),
                                   "--modalities", "MRI", "--ensemble", "both", "--no_cuda", "--no_save_images"])
        with patch.object(evaluation, "get_model", TinyModel), redirect_stdout(io.StringIO()):
            output = evaluation.test(args)
        summary = json.loads((output / "medical_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary.get("status"), "completed", "evaluation status must be persisted")
        checkpoint = load_checkpoint(directory / "best.pth")
        identity = summary["checkpoint_identity"]
        self.assertEqual(identity["run_id"], checkpoint["run_id"])
        self.assertEqual(identity["step"], checkpoint["step"])
        self.assertEqual(len(identity["sha256"]), 64)
        self.assertEqual(summary["protocol"]["metrics"]["ssim"]["window_size"], 11)
        self.assertEqual(summary["samples"]["counts"]["MRI"], 1)
        self.assertEqual(summary["provenance"]["environment"]["device"], "cpu")
        self.assertEqual(set(summary["metrics"]), {"no_self_ensemble", "self_ensemble"})
        self.assertIn("no_self_ensemble", summary["resources"])
        self.assertIn("self_ensemble", summary["resources"])
        with (self.root / "experiments.csv").open(newline="", encoding="utf-8") as handle:
            index = list(csv.DictReader(handle))
        self.assertEqual(len(index), 1)
        self.assertEqual(index[0]["status"], "completed")

    def test_evaluation_retains_first_mode_when_second_mode_fails(self):
        directory = self.root / "run"
        self.run_training(self.options(directory))
        args = parse_args("test", ["--model", str(directory / "best.pth"), "--test_path", str(self.data),
                                   "--split", "validation", "--result_dir", str(self.root / "results"),
                                   "--modalities", "MRI", "--ensemble", "both", "--no_cuda", "--no_save_images"])
        actual = evaluation.evaluate_medical
        def evaluate(*args, **kwargs):
            if args[3]:
                raise RuntimeError("self ensemble failed")
            return actual(*args, **kwargs)
        with patch.object(evaluation, "get_model", TinyModel), patch.object(evaluation, "evaluate_medical", evaluate):
            with redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "self ensemble failed"):
                evaluation.test(args)
        output = next((self.root / "results").iterdir())
        summary = json.loads((output / "medical_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary.get("status"), "failed")
        self.assertEqual(set(summary["metrics"]), {"no_self_ensemble"})
        self.assertTrue((output / "medical_metrics.csv").is_file())

    def test_atomic_json_failure_preserves_previous_record(self):
        path = self.root / "run.json"
        utils.save_summary_json(path, {"status": "running"})
        with patch.object(utils.os, "replace", side_effect=OSError("replace failed")):
            with self.assertRaisesRegex(OSError, "replace failed"):
                utils.save_summary_json(path, {"status": "completed"})
        self.assertEqual(json.loads(path.read_text()), {"status": "running"})

    def test_index_preserves_manual_notes_and_links_test_results(self):
        update = getattr(utils, "update_experiment_index", None)
        self.assertTrue(callable(update), "experiment index is missing")
        run = {"run_id": "r", "save_dir": str(self.root), "status": "completed", "config": {"seed": 42},
               "experiment": {"purpose": "original"}, "resources": {}, "best": {}, "provenance": {}}
        update(run)
        path = self.root / "experiments.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields, rows = reader.fieldnames, list(reader)
        rows[0]["notes"] = "manual note"
        utils.save_summary_csv(path, rows, fields)
        run["experiment"] = {"purpose": "original", "notes": "automated original note"}
        update(run)
        update(run, {"evaluation_id": "eval", "checkpoint_identity": {"step": 7},
                     "config": {"split": "test", "max_images": 1}, "output_dir": str(self.root),
                     "metrics": {"no_self_ensemble": {"macro": {"psnr": 30},
                                  "modalities": {"MRI": {"psnr": 30, "ssim": .9, "rmse": 1}}}},
                     "resources": {}, "samples": {}})
        with path.open(newline="", encoding="utf-8") as handle:
            row = next(csv.DictReader(handle))
        self.assertEqual(row["notes"], "manual note")
        self.assertEqual(row["purpose"], "original")
        self.assertEqual(row["test_evaluation_id"], "eval")
        self.assertEqual(float(row["test_MRI_psnr"]), 30)

    def test_locked_index_does_not_stop_training_or_overwrite_run_status(self):
        actual = utils.save_summary_csv
        def locked(path, rows, fields):
            if Path(path) == self.root / "experiments.csv":
                raise PermissionError("index open in Excel")
            return actual(path, rows, fields)
        directory = self.root / "run"
        with patch.object(utils, "save_summary_csv", locked), redirect_stderr(io.StringIO()):
            try:
                self.run_training(self.options(directory))
            except PermissionError as error:
                self.fail(f"comparison index availability must not stop model training: {error}")
        self.assertEqual(load_checkpoint(directory / "last.pth")["step"], 3)
        self.assertEqual(json.loads((directory / "run.json").read_text())["status"], "completed")

    def test_model_initialization_failure_records_failed_session(self):
        directory = self.root / "run"
        with patch.object(training, "get_model", side_effect=RuntimeError("model allocation failed")):
            with redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "model allocation failed"):
                training.train(self.options(directory))
        path = directory / "run.json"
        self.assertTrue(path.is_file(), "a validated run must record initialization failures")
        run = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(run["status"], "failed")
        self.assertEqual(run["sessions"][-1]["error"]["message"], "model allocation failed")
        self.assertIn("model allocation failed", (directory / "console.log").read_text(encoding="utf-8"))

    def test_initialization_rejects_checkpoint_replaced_during_loading(self):
        source = self.root / "init.pth"
        save_checkpoint(source, {"model": TinyModel().state_dict(), "run_id": "source-a", "step": 1})
        options = self.options(self.root / "run")
        options.init_model = str(source)
        def replaced(path):
            payload = load_checkpoint(path)
            save_checkpoint(source, {"model": TinyModel().state_dict(), "run_id": "source-b", "step": 2})
            return payload
        with patch.object(training, "load_checkpoint", replaced):
            with self.assertRaisesRegex(RuntimeError, "changed while loading"):
                self.run_training(options)

    def test_concurrent_index_updates_keep_all_experiments(self):
        path = self.root / "parallel.csv"
        code = ("import sys, utils; from pathlib import Path; "
                "utils.EXPERIMENT_INDEX=Path(sys.argv[1]); "
                "[utils.update_experiment_index({'run_id':sys.argv[2], 'status':'running', "
                "'config':{}, 'experiment':{}}) for _ in range(12)]")
        processes = [subprocess.Popen([sys.executable, "-B", "-c", code, str(path), str(index)],
                                      cwd=Path(__file__).resolve().parents[1],
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                     for index in range(6)]
        results = []
        try:
            for process in processes:
                stdout, stderr = process.communicate(timeout=30)
                results.append((process.returncode, stderr))
        finally:
            for process in processes:
                if process.poll() is None:
                    process.terminate()
                process.communicate()
        for code, stderr in results:
            self.assertEqual(code, 0, stderr.decode(errors="replace"))
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual({row["run_id"] for row in rows}, {str(index) for index in range(6)})

    def test_migration_can_retry_interrupted_history_copy(self):
        source, target = self.root / "source", self.root / "target"
        self.interrupt_after_first_checkpoint(source)
        copy = training.copy_checkpoint
        def fail_copy(origin, destination):
            if Path(destination).name == "training.csv":
                raise OSError("copy interrupted")
            return copy(origin, destination)
        with patch.object(training, "copy_checkpoint", fail_copy):
            with self.assertRaisesRegex(OSError, "copy interrupted"):
                self.run_training(self.options(target, interval=1, resume=source / "last.pth"))
        self.assertEqual(json.loads((target / "run.json").read_text())["status"], "failed")
        self.run_training(self.options(target, interval=1, resume=source / "last.pth"))
        self.assertEqual(load_checkpoint(target / "last.pth")["step"], 3)

    def test_effective_cpu_precision_is_distinct_from_requested_options(self):
        directory = self.root / "run"
        options = self.options(directory)
        options.amp = options.tf32 = True
        self.run_training(options)
        run = json.loads((directory / "run.json").read_text())
        environment = run["provenance"]["environment"]
        self.assertFalse(environment["amp"])
        self.assertFalse(environment["tf32"]["matmul"])
        self.assertFalse(environment["tf32"]["cudnn"])
        with (self.root / "experiments.csv").open(newline="", encoding="utf-8") as handle:
            row = next(csv.DictReader(handle))
        self.assertEqual(row["amp"], "False")
        self.assertEqual(row["amp_requested"], "True")

    def test_index_accepts_excel_utf8_bom_and_keeps_manual_notes(self):
        run = {"run_id": "r", "experiment": {"notes": "original"}, "config": {}}
        utils.update_experiment_index(run)
        path = self.root / "experiments.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields, rows = reader.fieldnames, list(reader)
        rows[0]["notes"] = "Excel 中填写的结论"
        with path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        utils.update_experiment_index(run)
        with path.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["notes"], "Excel 中填写的结论")


    def test_second_process_on_same_directory_is_refused(self):
        directory = self.root / "locked"
        directory.mkdir()
        code = ("import sys, time\n"
                "from pathlib import Path\n"
                "from utils import training_lock\n"
                "with training_lock(Path(sys.argv[1])):\n"
                "    print('locked', flush=True)\n"
                "    time.sleep(15)\n")
        process = subprocess.Popen([sys.executable, "-B", "-c", code, str(directory)],
                                   cwd=Path(__file__).resolve().parents[1],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            line = process.stdout.readline().decode().strip()
            if line != "locked":
                self.fail(f"lock helper did not start: {process.stderr.read().decode(errors='replace')}")
            with self.assertRaisesRegex(RuntimeError, "already used by another training process"):
                self.run_training(self.options(directory))
        finally:
            process.terminate()
            process.communicate()
        self.assertFalse((directory / "last.pth").exists(), "a locked directory must stay untouched")

    def test_legacy_checkpoint_reuses_identity_when_migrating(self):
        source, target = self.root / "source", self.root / "target"
        self.run_training(self.options(source, interval=1))
        legacy = load_checkpoint(source / "last.pth")
        identity = legacy.pop("run_id")
        save_checkpoint(source / "last.pth", legacy)
        (source / "run.json").unlink()
        self.run_training(self.options(target, interval=1, resume=source / "last.pth"))
        marker = json.loads((target / "run.json").read_text())
        self.assertEqual(marker["run_id"], identity)
        self.assertEqual(load_checkpoint(target / "best.pth")["run_id"], identity)
        self.assertEqual(load_checkpoint(target / "last.pth")["run_id"], identity)

    def test_missing_run_json_restores_resource_totals_from_checkpoint(self):
        directory = self.root / "run"
        self.run_training(self.options(directory, interval=1))
        before = load_checkpoint(directory / "last.pth")["resources"]
        self.assertGreater(before["successful_updates"], 0)
        (directory / "run.json").unlink()
        self.run_training(self.options(directory, interval=1, resume="auto"))
        run = json.loads((directory / "run.json").read_text())
        self.assertEqual(run["resources_source"], "checkpoint")
        self.assertGreaterEqual(run["resources"]["successful_updates"], before["successful_updates"])
        self.assertGreaterEqual(run["resources"]["wall_seconds"], before["wall_seconds"])


if __name__ == "__main__":
    unittest.main()
