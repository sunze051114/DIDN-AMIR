"""Independently evaluate a DIDN+AMIR checkpoint on MRI/CT/PET slices."""
import json
import os
import traceback
from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4

import torch

from checkpoints import load_checkpoint
from dataset import MedicalDataset
from model import get_model
from option import parse_args
from utils import (ResourceTracker, collect_provenance, file_identity, save_summary_csv,
                   save_summary_json, update_experiment_index)
from validation import evaluate_medical, metric_protocol


def load_model(path, device, return_metadata=False):
    """Load weights and bind their reported identity to the same stable file."""
    path = Path(path)
    before = path.stat()
    checkpoint = load_checkpoint(path)
    with torch.device("meta"):
        model = get_model()
    model.load_state_dict(checkpoint["model"], strict=True, assign=True)
    model = model.to(device).eval()
    metadata = {key: checkpoint[key] for key in
                ("run_id", "step", "arch", "metrics", "validation_event_id") if key in checkpoint}
    identity = file_identity(path, metadata=metadata)
    after = path.stat()
    signature = lambda stat: (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    if signature(before) != signature(after):
        raise RuntimeError(f"Checkpoint changed while loading: {path}; retry with a stable checkpoint")
    result = model, checkpoint.get("config")
    return (*result, {**metadata, "checkpoint_identity": identity}) if return_metadata else result


def recorded_samples(path):
    """Validation samples recorded next to a checkpoint, when that run saved them."""
    candidate = Path(path).parent / "validation_samples.json"
    if not candidate.is_file():
        return None
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    samples = payload.get("samples")
    return samples if isinstance(samples, dict) and samples else None


def select_samples(samples, modalities):
    """Recorded samples narrowed to the evaluated modalities, or None if incomplete."""
    if samples is None:
        return None
    selected = {modality: names for modality, names in samples.items() if modality in modalities}
    return selected if set(selected) == set(modalities) else None


def _index_run(path, identity, training_config):
    """Only link evaluation to a run whose ID is recorded in these weights."""
    run_id = (identity or {}).get("run_id")
    if not run_id:
        return None
    candidate = path.parent / "run.json"
    if candidate.is_file():
        try:
            run = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            print(f"[Warning] Cannot read neighboring run metadata: {error}", flush=True)
        else:
            if isinstance(run, dict) and run.get("run_id") == run_id:
                return run
    # Legacy checkpoints can identify their run without inventing its provenance.
    return {"run_id": run_id, "save_dir": str(path.parent), "config": training_config}


def _timestamp():
    return datetime.now(timezone.utc).isoformat()


def test(opt):
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
    device = torch.device("cuda" if not opt.no_cuda and torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
    path = Path(opt.model or Path(opt.save_dir) / "best.pth").expanduser().resolve()
    evaluation_id = uuid4().hex
    output = Path(opt.result_dir).expanduser().resolve() / (
        datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + evaluation_id[:8])
    output.mkdir(parents=True)
    report = {
        "evaluation_id": evaluation_id, "status": "running", "started_at": _timestamp(),
        "ended_at": None, "checkpoint": str(path), "checkpoint_identity": None,
        "checkpoint_metadata": {}, "config": vars(opt), "training_config": None,
        "protocol": metric_protocol(), "samples": {}, "provenance": {},
        "resources": {}, "metrics": {}, "modes": {}, "output_dir": str(output),
    }
    fields = ["modality", "image", "psnr", "ssim", "rmse", "self_ensemble", "checkpoint",
              "run_id", "checkpoint_step", "checkpoint_sha256", "evaluation_id"]
    all_rows, run = [], None

    def persist():
        save_summary_csv(output / "medical_metrics.csv", all_rows, fields=fields)
        save_summary_json(output / "medical_summary.json", report)
        if run is not None:
            update_experiment_index(run, evaluation=report)

    persist()
    group, tracker = None, None
    try:
        report["provenance"] = collect_provenance(opt, device, amp=False)
        if not path.is_file():
            raise FileNotFoundError(
                f"No DIDN+AMIR checkpoint at {path}; finish training or pass --model. "
                "Resuming into a new --save_dir migrates the matching best.pth from the original directory")
        print(f"[Checkpoint] {path}", flush=True)
        model, training_config, metadata = load_model(path, device, return_metadata=True)
        identity = metadata.pop("checkpoint_identity")
        report.update(checkpoint_identity=identity, checkpoint_metadata=metadata,
                      training_config=training_config)
        report["provenance"] = collect_provenance(opt, device, amp=False, model=model)
        run = _index_run(path, identity, training_config)
        # Keep the existing fallback: recorded samples first, otherwise seed + max_images.
        recorded = recorded_samples(path) if opt.split == "validation" else None
        samples = select_samples(recorded, opt.modalities)
        seed = (training_config or {}).get("seed", 42)
        warning = None
        if opt.split == "validation" and samples is None:
            reason = ("Recorded samples do not cover every requested modality" if recorded is not None
                      else "No usable validation_samples.json was found beside the checkpoint")
            warning = (f"{reason}; deriving validation samples with seed={seed} and "
                       f"max_images={opt.max_images}. This may differ from the checkpoint's original validation subset")
            print(f"[Warning] {warning}", flush=True)
        dataset = MedicalDataset(opt.test_path, opt.modalities, opt.split, seed=seed,
                                 limit=opt.max_images, samples=samples)
        report["samples"] = {
            "selection": dict(dataset.selection), "selected": dataset.selected,
            "counts": dataset.counts, "patient_counts": dataset.patient_counts,
            "patient_id_source": "Filename-inferred using dataset.patient_key; not verified patient identities",
            "selection_source": ("recorded_validation_samples" if samples is not None else
                                 "derived_seed_and_max_images" if opt.split == "validation" else
                                 "test_split_options"),
            "warning": warning,
        }
        print(f"[Dataset] split={opt.split} seed={seed} samples={'recorded' if samples else 'derived'}",
              flush=True)
        persist()
        modes = (False, True) if opt.ensemble == "both" else (opt.ensemble == "self",)
        for use_se in modes:
            group = "self_ensemble" if use_se else "no_self_ensemble"
            report["modes"][group] = {"status": "running", "started_at": _timestamp(), "ended_at": None}
            tracker = ResourceTracker(device)
            persist()
            try:
                with tracker.validation():
                    summary, rows = evaluate_medical(model, dataset, device, use_se,
                                                     None if opt.no_save_images else output / group, opt.threads)
            finally:
                report["resources"][group] = tracker.snapshot(0)
            report["metrics"][group] = summary
            report["modes"][group].update(status="completed", ended_at=_timestamp())
            all_rows.extend({**row, "self_ensemble": int(use_se), "checkpoint": str(path),
                             "run_id": identity.get("run_id"), "checkpoint_step": identity.get("step"),
                             "checkpoint_sha256": identity["sha256"], "evaluation_id": evaluation_id}
                            for row in rows)
            persist()
        report.update(status="completed", ended_at=_timestamp())
        persist()
    except BaseException as error:
        status = "interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed"
        report.update(status=status, ended_at=_timestamp(), error={
            "type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()})
        if group is not None and report["modes"][group]["status"] == "running":
            report["modes"][group].update(status=status, ended_at=report["ended_at"])
        try:
            persist()
        except Exception as recording_error:
            print(f"[Warning] Cannot persist evaluation failure: {recording_error}", flush=True)
        print(f"[{status.capitalize()}] Results: {output}", flush=True)
        raise
    print(f"[Done] {report['metrics']}; results: {output}", flush=True)
    return output


if __name__ == "__main__":
    test(parse_args("test"))
