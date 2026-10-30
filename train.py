"""Iteration-based training of DIDN+AMIR on existing paired medical data."""
import hashlib
import json
import os
import sys
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from checkpoints import (checkpoint_summary, copy_checkpoint, ensure_best_checkpoint, load_checkpoint, restore_training,
                         save_checkpoint, training_state)
from dataset import MedicalDataset
from validation import evaluate_medical, metric_protocol
from model import get_model
from option import parse_args
from utils import (ResourceTracker, capture_console, collect_provenance, commit_validation_csv,
                   file_identity, log_result_csv, prepare_training_csv, reconcile_validation_csv,
                   save_summary_json, training_lock, update_experiment_index)


TRAINING_FIELDS = ["step", "epoch", "l1", "balance", "weighted_balance", "total_loss",
                   "lr", "grad_norm", "amp_scale", "skipped_batches"]
VALIDATION_FIELDS = ["step", "modality", "psnr", "ssim", "rmse", "run_id", "session_id", "event_id"]


def _timestamp():
    return datetime.now(timezone.utc).isoformat()


def _load_training_checkpoint(path):
    """Bind initialization/resume provenance to the bytes actually loaded."""
    path = Path(path).expanduser().resolve()
    before = path.stat()
    checkpoint = load_checkpoint(path)
    identity = file_identity(path, checkpoint)
    after = path.stat()
    signature = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
    if signature(before) != signature(after):
        raise RuntimeError(f"Checkpoint changed while loading: {path}; retry with a stable checkpoint")
    return checkpoint, identity


class _TrainingRecord:
    """Keep provenance, lifecycle and resource records in the existing run.json."""
    def __init__(self, directory, run_id, config, opt, device, model, resume, step,
                 fallback_resources=None):
        self.directory, self.step = directory, step
        previous = read_marker(directory)
        if previous is None and resume is not None:
            candidate = read_marker(resume.parent)
            if candidate and candidate.get("run_id") == run_id:
                previous = candidate
        self.data = dict(previous or {})
        self.data.update(run_id=run_id, save_dir=str(directory), config=config,
                         protocol=metric_protocol(), status="running")
        if resume is not None and resume.parent != directory:
            self.data["migrated_from"] = str(resume.parent)
        # run.json is authoritative; a missing marker must not reset cumulative cost.
        resources = self.data.get("resources") or fallback_resources
        if resources:
            self.data["resources"] = resources
            if not (previous or {}).get("resources"):
                self.data["resources_source"] = "checkpoint"
        self.tracker = ResourceTracker(device, initial_step=step, previous=resources)
        provenance = collect_provenance(opt, device, amp=opt.amp, model=model)
        # Source text/diffs occur once per distinct code snapshot, not once per resume.
        sources = {"git": provenance["git"], "core_sources": provenance["core_sources"]}
        source_id = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
        self.data.setdefault("source_snapshots", {}).setdefault(source_id, sources)
        self.data["provenance"] = {
            **provenance, "source_id": source_id,
            "git": {key: value for key, value in provenance["git"].items() if key != "diff"},
            "core_sources": {name: {key: value for key, value in source.items() if key != "untracked_source"}
                             for name, source in provenance["core_sources"].items()}}
        self.session_id = uuid4().hex
        sessions = self.data.setdefault("sessions", [])
        for session in sessions:
            if session.get("status") == "running":
                session["status"] = "unclosed"
        source = resume or (Path(opt.init_model).expanduser().resolve() if opt.init_model else None)
        lineage = {"mode": "resume" if resume else "init_model" if source else "fresh"}
        if source is not None:
            lineage["checkpoint"] = {"path": str(source), "loaded": False}
        self.session = {"session_id": self.session_id, "started_at": _timestamp(),
                        "status": "running", "start_step": step, "source": lineage,
                        "command": provenance["command"], "options": provenance["options"],
                        "environment": provenance["environment"], "source_id": source_id}
        sessions.append(self.session)
        experiment = self.data.setdefault("experiment", {})
        self.annotation_overrides = {}
        for key, attribute in (("name", "experiment_name"), ("purpose", "purpose"),
                               ("comparison", "comparison"), ("notes", "notes")):
            value = getattr(opt, attribute, "")
            if value or key not in experiment:
                experiment[key] = value
            if value:
                self.annotation_overrides[key] = value
        self.event("resumed" if resume else "started", source=lineage)

    def event(self, kind, **values):
        event = {"event_id": uuid4().hex, "session_id": self.session_id,
                 "timestamp": _timestamp(), "type": kind, "step": self.step, **values}
        self.data.setdefault("events", []).append(event)
        return event

    def save(self):
        self.data["resources"] = self.tracker.snapshot(self.step)
        self.session.update(last_step=self.step, resources=self.data["resources"])
        self.data["updated_at"] = _timestamp()
        save_summary_json(self.directory / "run.json", self.data)
        if update_experiment_index(self.data, annotation_overrides=self.annotation_overrides) is not None:
            self.annotation_overrides.clear()

    def finish(self, status, error=None):
        self.data["status"] = self.session["status"] = status
        self.session.update(ended_at=_timestamp(), end_step=self.step)
        if error is not None:
            self.session["error"] = {"type": type(error).__name__, "message": str(error)}
        self.event(status, error=self.session.get("error"))
        self.save()


def epoch_indices(length, batch_size, seed, epoch):
    generator = torch.Generator().manual_seed(seed + epoch)
    indices = torch.randperm(length, generator=generator).tolist()
    return indices[:length // batch_size * batch_size]


def check_medical_data(opt):
    report = {}
    for split in ("train", "validation", "test"):
        dataset = MedicalDataset(opt.data_path, opt.modalities, split, opt.patch_size, opt.seed)
        samples = {}
        for modality in opt.modalities:
            indices = [i for i, record in enumerate(dataset.records) if record[2] == modality]
            shapes = set()
            for index in (indices[0], indices[len(indices) // 2], indices[-1]):
                low, high, _, _ = dataset[index]
                shapes.add(tuple(low.shape))
            samples[modality] = {"pairs": len(indices), "sample_shapes": sorted(shapes)}
        report[split] = samples
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def data_signature(*datasets):
    digest = hashlib.sha256()
    for dataset in datasets:
        for low, high, modality in dataset.records:
            digest.update(f"{low}\0{high}\0{modality}\n".encode())
    return digest.hexdigest()


TRAINING_ARTIFACTS = ("last.pth", "best.pth", "config.json", "validation.csv", "training.csv", "console.log",
                      "validation_samples.json", "run.json")


def read_marker(directory):
    """run.json payload of a training directory, or None when it has no marker."""
    marker = Path(directory) / "run.json"
    if not marker.is_file():
        return None
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise FileExistsError(f"{marker} is unreadable; choose a new --save_dir") from None
    return payload if isinstance(payload, dict) else None


def _adopt_run_id(recorded, origin):
    print(f"[Checkpoints] Legacy checkpoint has no run id; reusing {recorded} recorded in {origin}",
          flush=True)
    return recorded


def resolve_run_id(resume, directory, config=None):
    """Run id of this invocation.

    Checkpoints saved before run ids existed cannot identify themselves, so an
    existing identity of the same lineage is reused instead of minting a new id:
    the directory marker, then the source directory marker, then a sibling
    best.pth. Otherwise every resume of such a checkpoint would pick a different
    id and reject its own, still valid, best checkpoint.
    """
    checkpoint_run = checkpoint_summary(resume)["run_id"] if resume is not None else None
    if checkpoint_run:
        return checkpoint_run
    if resume is None:
        return uuid4().hex
    marker = read_marker(directory)
    if marker and marker.get("run_id"):
        if resume.parent == directory or marker.get("migrated_from") == str(resume.parent):
            return _adopt_run_id(marker["run_id"], directory / "run.json")
    source_marker = read_marker(resume.parent)
    if source_marker and source_marker.get("run_id"):
        return _adopt_run_id(source_marker["run_id"], resume.parent / "run.json")
    sibling = resume.parent / "best.pth"
    if sibling.is_file():
        summary = checkpoint_summary(sibling)
        if summary["run_id"] and (config is None or summary["config"] in (None, config)):
            return _adopt_run_id(summary["run_id"], sibling)
    print(f"[Checkpoints] {resume.name} predates run ids; adopting a new one for this directory",
          flush=True)
    return uuid4().hex


def resume_location(opt, directory):
    """Checkpoint to resume from, or None for a fresh run."""
    if not opt.resume:
        return None
    path = directory / "last.pth" if opt.resume == "auto" else Path(opt.resume).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {path}")
    return path


def check_output_directory(directory, resume):
    """Refuse to write into the results of another experiment."""
    if resume is None:
        if any((directory / name).exists() for name in ("last.pth", "best.pth")):
            raise FileExistsError(f"Training outputs already exist in {directory}; use --resume auto or another --save_dir")
        return
    if resume.parent == directory:
        return
    existing = [name for name in TRAINING_ARTIFACTS if (directory / name).exists()]
    if "last.pth" in existing:
        raise FileExistsError(
            f"{directory} already holds a complete run ({existing}); use --resume auto there "
            "or choose a new --save_dir")
    if existing:
        if not (directory / "run.json").is_file():
            raise FileExistsError(
                f"{directory} contains {existing} but no run.json, so the experiment cannot be "
                "identified as an interrupted migration; choose a new --save_dir")


def check_partial_migration(directory, resume, config, selection, run_id):
    """Only the experiment that owns the directory may write into it.

    Identical configurations and sample lists are not enough: two separate runs can
    share them completely. The directory marker (`run.json`) records which run owns
    the artifacts, so weights, best scores and logs can never be mixed across runs.
    """
    marker = read_marker(directory)
    existing = ([name for name in TRAINING_ARTIFACTS if (directory / name).exists()]
                if directory.is_dir() else [])
    if marker is not None:
        recorded = marker.get("run_id")
        if resume is None:
            raise FileExistsError(
                f"{directory} already belongs to experiment {recorded}; choose a new --save_dir")
        if not run_id or recorded != run_id:
            raise FileExistsError(
                f"{directory} belongs to experiment {recorded}, not {run_id or 'this checkpoint'}; "
                "choose a new --save_dir")
    elif resume is None and any((directory / name).exists() for name in ("training.csv", "console.log")):
        raise FileExistsError(
            f"{directory} holds training logs but no run.json; choose a new --save_dir")
    elif resume is not None and resume.parent != directory and existing:
        raise FileExistsError(
            f"{directory} holds training artifacts but no run.json, so the experiment cannot be "
            "identified; choose a new --save_dir")
    if resume is None or resume.parent == directory or not existing:
        return
    recorded_config = directory / "config.json"
    if recorded_config.is_file():
        try:
            previous = json.loads(recorded_config.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise FileExistsError(f"{recorded_config} is unreadable; choose a new --save_dir") from None
        if previous != config:
            raise FileExistsError(f"{recorded_config} belongs to another configuration; choose a new --save_dir")
    samples = directory / "validation_samples.json"
    if samples.is_file():
        previous = json.loads(samples.read_text(encoding="utf-8"))
        if previous.get("samples") != selection["samples"]:
            raise FileExistsError(f"{samples} belongs to another validation selection; choose a new --save_dir")
    print(f"[Checkpoints] Completing an interrupted migration into {directory}", flush=True)


def validation_selection(dataset):
    return {"strategy": dataset.selection["strategy"], "seed": dataset.selection["seed"],
            "limit": dataset.selection["limit"], "counts": dataset.counts,
            "patients": dataset.patient_counts, "samples": dataset.selected}


def check_validation_samples(selection, source):
    """A resumed history must be scored on exactly the samples recorded for it."""
    path = Path(source) / "validation_samples.json"
    if not path.is_file():
        print(f"[Validation] No {path.name} in {source}; using the current deterministic selection", flush=True)
        return
    previous = json.loads(path.read_text(encoding="utf-8"))
    if previous.get("strategy") != selection["strategy"] or previous.get("samples") != selection["samples"]:
        raise ValueError(f"Validation samples in {path} differ from the current selection; "
                         "resume with the original --val_samples, --seed and --data_path")


def train(opt):
    """Serialize one save_dir across processes for the whole run, then train."""
    directory = Path(opt.save_dir).expanduser().resolve()
    guard = nullcontext() if opt.check_data else training_lock(directory)
    with guard, capture_console() as console:
        records = {}
        try:
            result = _train(opt, console, records)
        except BaseException as error:
            if "run" in records:
                try:
                    records["run"].finish(
                        "interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed", error)
                except Exception as record_error:
                    print(f"[Recording] Could not finalize run metadata: {record_error}", file=sys.stderr)
            raise
        if "run" in records:
            records["run"].finish("completed")
        return result


def _train(opt, console, records):
    if opt.check_data:
        return check_medical_data(opt)
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
    if opt.deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    device = torch.device("cuda" if not opt.no_cuda and torch.cuda.is_available() else "cpu")
    torch.manual_seed(opt.seed)
    torch.use_deterministic_algorithms(opt.deterministic)
    torch.backends.cudnn.deterministic = opt.deterministic
    torch.backends.cudnn.benchmark = device.type == "cuda" and not opt.deterministic
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = opt.tf32
    directory = Path(opt.save_dir).expanduser().resolve()
    resume = resume_location(opt, directory)
    check_output_directory(directory, resume)
    dataset = MedicalDataset(opt.data_path, opt.modalities, "train", opt.patch_size, opt.seed)
    validation = MedicalDataset(opt.data_path, opt.modalities, "validation",
                                opt.patch_size, opt.seed, limit=opt.val_samples)
    selection = validation_selection(validation)
    if resume is not None:
        check_validation_samples(selection, resume.parent)
    batches = len(dataset) // opt.batchSize
    if not batches:
        raise ValueError("Dataset is smaller than one complete training batch")
    keys = ("modalities", "batchSize", "patch_size", "seed",
            "lr", "min_lr", "total_iterations", "balance_weight", "val_interval", "val_samples",
            "amp", "tf32", "grad_clip", "deterministic")
    config = {key: getattr(opt, key) for key in keys}
    config.update(data_path=str(Path(opt.data_path).resolve()), train_counts=dataset.counts,
                  validation_counts=validation.counts, data_signature=data_signature(dataset, validation))
    # Identity needs the computed configuration to validate a sibling best.pth.
    run_id = resolve_run_id(resume, directory, config)
    check_partial_migration(directory, resume, config, selection, run_id)
    resume_metadata = checkpoint_summary(resume) if resume is not None else None
    if resume_metadata is not None and resume_metadata.get("config") != config:
        raise ValueError("Resume configuration differs; use the original training parameters")
    initial_step = (resume_metadata or {}).get("step") or 0
    directory.mkdir(parents=True, exist_ok=True)
    marker = read_marker(directory)
    if marker is None and resume is not None:
        source_marker = read_marker(resume.parent)
        marker = source_marker if source_marker and source_marker.get("run_id") == run_id else None
    marker = dict(marker or {})
    marker["run_id"] = run_id
    if resume is not None and resume.parent != directory:
        marker["migrated_from"] = str(resume.parent)
    # Register ownership before copying any migration files or allocating the model.
    save_summary_json(directory / "run.json", marker)
    record = _TrainingRecord(directory, run_id, config, opt, device, None, resume, initial_step,
                             fallback_resources=(resume_metadata or {}).get("resources"))
    records["run"] = record
    record.save()
    if resume is not None and resume.parent != directory:
        for name in ("validation.csv", "training.csv", "console.log"):
            history = resume.parent / name
            if history.is_file() and not (directory / name).is_file():
                copy_checkpoint(history, directory / name)
                print(f"[Checkpoints] Copied {name} from {history}", flush=True)
    console.attach(directory / "console.log")
    save_summary_json(directory / "validation_samples.json", selection)
    save_summary_json(directory / "config.json", config)
    print(f"[Training] DIDN+AMIR; {device}; paired data={dataset.counts}; "
          f"batch={opt.batchSize}; patch={opt.patch_size}; updates={opt.total_iterations}", flush=True)
    print(f"[Validation] strategy={selection['strategy']} limit={selection['limit']} "
          f"samples={selection['counts']} patients={selection['patients']}", flush=True)
    model = get_model().to(device)
    dtypes = sorted({str(value.dtype) for value in model.parameters()})
    record.data["provenance"]["environment"].update(
        model_dtype=dtypes[0] if dtypes else None, model_dtypes=dtypes)
    record.data["model"] = {"arch": "didn_amir", "class": type(model).__name__,
                            "parameters": sum(value.numel() for value in model.parameters())}
    if opt.init_model:
        checkpoint, identity = _load_training_checkpoint(opt.init_model)
        record.session["source"]["checkpoint"] = {**identity, "loaded": True}
        model.load_state_dict(checkpoint["model"], strict=True)
        del checkpoint
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    use_amp = opt.amp and device.type == "cuda"
    if opt.amp and not use_amp:
        print("[AMP] --amp requested but CUDA is unavailable; training in FP32", flush=True)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=opt.total_iterations, eta_min=opt.min_lr)
    start_epoch, offset, best_psnr, step = 1, 0, float("-inf"), 0
    skipped_batches = 0
    validation_event_id = None
    if resume is not None:
        checkpoint, identity = _load_training_checkpoint(resume)
        record.session["source"]["checkpoint"] = {**identity, "loaded": True}
        if checkpoint.get("step") != initial_step or checkpoint.get("run_id") not in (None, run_id):
            raise RuntimeError(f"Resume checkpoint changed since preflight: {resume}")
        if "scheduler" not in checkpoint or "step" not in checkpoint:
            raise ValueError("AMIR resume requires a complete medical training checkpoint")
        start_epoch, offset, best_psnr = restore_training(checkpoint, model, optimizer, scaler, config)
        scheduler.load_state_dict(checkpoint["scheduler"])
        step = checkpoint["step"]
        skipped_batches = checkpoint.get("skipped_batches", 0)
        validation_event_id = checkpoint.get("validation_event_id")
        if "skipped_batches" not in checkpoint:
            print("[Training] Legacy checkpoint has no overflow count; starting skipped_batches at 0",
                  flush=True)
        del checkpoint
        best_psnr = ensure_best_checkpoint(directory, resume.parent, best_psnr, config, run_id)
    print(f"[Checkpoints] {directory}; best.pth for testing, last.pth for resume", flush=True)
    prepare_training_csv(directory / "training.csv", TRAINING_FIELDS, step)
    record.step = step
    report = reconcile_validation_csv(directory / "validation.csv", VALIDATION_FIELDS, opt.modalities, step)
    if report.get("discarded_groups") or report.get("ahead_steps"):
        record.event("validation_reconciled", **report)
    best_path = directory / "best.pth"
    if best_path.is_file():
        best = checkpoint_summary(best_path)
        record.data["best"] = {"step": best["step"], "metrics": best.get("metrics"),
                               "validation_event_id": best.get("validation_event_id")}
        # best.pth can be newer than last.pth and can precede its CSV commit.
        if best.get("metrics") and best.get("step") not in report.get("retained_steps", []):
            recovered_id = best.get("validation_event_id") or uuid4().hex
            rows = [{"step": best["step"], "modality": modality, **metrics, "run_id": run_id,
                     "session_id": record.session_id, "event_id": recovered_id}
                    for modality, metrics in best["metrics"]["modalities"].items()]
            commit_validation_csv(directory / "validation.csv", VALIDATION_FIELDS, rows)
            record.event("validation_recovered_from_best", checkpoint_step=best["step"],
                         validation_event_id=recovered_id)
    record.save()

    def snapshot(epoch, next_iteration):
        state = training_state(model, optimizer, scaler, config, epoch, next_iteration, best_psnr, run_id)
        state.update(step=step, scheduler=scheduler.state_dict(), skipped_batches=skipped_batches,
                     validation_event_id=validation_event_id, resources=record.tracker.snapshot(step))
        save_checkpoint(directory / "last.pth", state)
        record.event("checkpoint_saved", checkpoint="last.pth", validation_event_id=validation_event_id)
        record.save()

    def validate():
        nonlocal best_psnr, validation_event_id
        event_id = uuid4().hex
        # Validation must not alter the next stochastic SRM training gates.
        rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
        old_conv, old_matmul = torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32
        torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
        try:
            with record.tracker.validation():
                summary, _ = evaluate_medical(model, validation, device, num_workers=opt.threads)
        finally:
            torch.set_rng_state(rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
            torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = old_conv, old_matmul
        score = summary["macro"]["psnr"]
        if score > best_psnr:
            best_psnr = score
            save_checkpoint(directory / "best.pth", {"model": model.state_dict(), "arch": "didn_amir",
                            "config": config, "step": step, "metrics": summary, "run_id": run_id,
                            "validation_event_id": event_id})
            record.data["best"] = {"step": step, "metrics": summary, "validation_event_id": event_id}
            record.event("best_updated", checkpoint="best.pth", validation_event_id=event_id)
        rows = [{"step": step, "modality": modality, **metrics, "run_id": run_id,
                 "session_id": record.session_id, "event_id": event_id}
                for modality, metrics in summary["modalities"].items()]
        replaced = commit_validation_csv(directory / "validation.csv", VALIDATION_FIELDS, rows)
        validation_event_id = event_id
        record.event("validation", validation_event_id=event_id, metrics=summary,
                     replaced_event_ids=replaced)
        record.save()
        print(f"[Validation] update={step} macro PSNR={score:.4f}; {summary['modalities']}", flush=True)

    if resume is not None and resume.parent != directory:
        # A migrated directory must stay resumable even if this process stops next.
        snapshot(start_epoch, offset)
        print(f"[Checkpoints] Resumable last.pth written in {directory}", flush=True)

    epoch = start_epoch
    record.tracker.start_training()
    while step < opt.total_iterations:
        if not 0 <= offset <= batches:
            raise ValueError("Resume batch position is outside the dataset")
        dataset.epoch = epoch
        indices = epoch_indices(len(dataset), opt.batchSize, opt.seed, epoch)
        loader = DataLoader(Subset(dataset, indices[offset * opt.batchSize:]), batch_size=opt.batchSize,
                            num_workers=opt.threads, pin_memory=device.type == "cuda",
                            generator=torch.Generator().manual_seed(opt.seed + epoch))
        model.train()
        for iteration, (low, high, _, _) in enumerate(loader, offset):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                prediction, balance = model(low.to(device), return_aux=True)
                reconstruction = nn.functional.l1_loss(prediction, high.to(device))
                loss = reconstruction + opt.balance_weight * balance
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at update {step + 1}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm = nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip or float("inf"),
                                           error_if_nonfinite=not use_amp)
            old_scale = scaler.get_scale()
            lr_used = optimizer.param_groups[0]["lr"]
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < old_scale:
                skipped_batches += 1
                print(f"[AMP] Skipped overflowed batch; scale={scaler.get_scale():.1f} "
                      f"skipped_batches={skipped_batches}", flush=True)
                continue
            scheduler.step()
            step += 1
            record.step = step
            final = step >= opt.total_iterations
            if step == 1 or step % 100 == 0 or final:
                row = {"step": step, "epoch": epoch, "l1": reconstruction.item(),
                       "balance": balance.item(), "weighted_balance": opt.balance_weight * balance.item(),
                       "total_loss": loss.item(), "lr": lr_used, "grad_norm": norm.item(),
                       "amp_scale": old_scale, "skipped_batches": skipped_batches}
                log_result_csv(directory / "training.csv", TRAINING_FIELDS, row)
                print(f"[Train] update={step} L1={row['l1']:.6f} balance={row['balance']:.6f} "
                      f"weighted_balance={row['weighted_balance']:.6f} loss={row['total_loss']:.6f} "
                      f"lr={lr_used:.3g} grad={row['grad_norm']:.4f} "
                      f"amp_scale={old_scale:.1f} skipped_batches={skipped_batches}", flush=True)
            should_validate = final or (opt.val_interval > 0 and step % opt.val_interval == 0)
            if should_validate:
                validate()
                snapshot(epoch, iteration + 1)
            if final:
                print(f"[Done] Checkpoints: {directory}", flush=True)
                return directory
        if opt.val_interval == 0:
            validate()
        epoch, offset = epoch + 1, 0
        snapshot(epoch, 0)
    print(f"[Done] Already completed {step} updates: {directory}", flush=True)
    return directory


if __name__ == "__main__":
    train(parse_args("train"))
