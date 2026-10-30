"""CSV training logs and evaluation reports."""
import csv
import hashlib
import io
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path


def log_result_csv(path, fields, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def save_summary_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_summary_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _json_value(value):
    """Keep record metadata portable, including pathlib and package versions."""
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _json_value(value.item())
        except (ValueError, RuntimeError):
            pass
    return str(value)


def file_identity(path, metadata=None):
    """Hash a file in one streaming pass without loading a checkpoint."""
    path = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        size = os.fstat(handle.fileno()).st_size
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    result = {"path": str(path), "size_bytes": size, "sha256": digest.hexdigest()}
    if metadata:
        for key in ("run_id", "step", "arch", "metrics", "validation_event_id"):
            if key in metadata:
                result[key] = _json_value(metadata[key])
    return result


CORE_SOURCES = ("train.py", "test.py", "validation.py", "dataset.py",
                "model.py", "option.py", "checkpoints.py", "utils.py")


def collect_provenance(options, device, amp=False, model=None):
    """Record effective runtime state separately from strict resume configuration.

    Source hashes and the scoped Git diff describe only runtime Python sources;
    unrelated output/recovery deletions never become part of the snapshot.
    """
    import numpy as np
    import SimpleITK as sitk
    import torch

    root = Path(__file__).resolve().parent

    def git(*arguments):
        try:
            result = subprocess.run(["git", *arguments], cwd=root, capture_output=True,
                                    text=True, encoding="utf-8", errors="replace", timeout=15)
        except (OSError, subprocess.SubprocessError):
            return None
        return result.stdout if result.returncode == 0 else None

    status = git("status", "--porcelain", "--untracked-files=normal")
    commit, branch = git("rev-parse", "HEAD"), git("branch", "--show-current")
    tracked = set((git("ls-files", "--", *CORE_SOURCES) or "").splitlines())
    sources = {}
    for name in CORE_SOURCES:
        path = root / name
        if path.is_file():
            content = path.read_bytes()
            sources[name] = {"sha256": hashlib.sha256(content).hexdigest(),
                             "size_bytes": len(content)}
            if name not in tracked:
                sources[name]["untracked_source"] = content.decode("utf-8", errors="replace")
    device = torch.device(device)
    cuda = device.type == "cuda" and torch.cuda.is_available()
    gpu = None
    if cuda:
        index = device.index if device.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        gpu = {"index": index, "name": properties.name, "total_memory_bytes": properties.total_memory,
               "capability": [properties.major, properties.minor]}
    dtypes = sorted({str(value.dtype) for value in model.parameters()}) if model is not None else []
    if model is not None and not dtypes:
        dtypes = sorted({str(value.dtype) for value in model.buffers()})
    environment = {
        "python": platform.python_version(), "python_implementation": platform.python_implementation(),
        "platform": platform.platform(), "machine": platform.machine(),
        "torch": str(torch.__version__), "numpy": str(np.__version__),
        "SimpleITK": str(sitk.Version_VersionString()), "cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()), "cudnn": torch.backends.cudnn.version(),
        "device": str(device), "gpu": gpu, "model_dtype": dtypes[0] if dtypes else None,
        "model_dtypes": dtypes, "amp": bool(amp and cuda),
        "tf32": {"matmul": bool(cuda and torch.backends.cuda.matmul.allow_tf32),
                 "cudnn": bool(cuda and torch.backends.cudnn.allow_tf32)},
        "deterministic": {"algorithms": torch.are_deterministic_algorithms_enabled(),
                          "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
                          "cudnn": bool(torch.backends.cudnn.deterministic),
                          "cudnn_benchmark": bool(torch.backends.cudnn.benchmark)},
        "threads": {"torch": torch.get_num_threads(), "interop": torch.get_num_interop_threads(),
                    "data_loader": getattr(options, "threads", None)},
    }
    return {"command": {"argv": list(sys.argv), "executable": sys.executable},
            "options": _json_value(options if isinstance(options, dict) else vars(options)),
            "git": {"commit": commit.strip() if commit else None,
                    "branch": branch.strip() if branch else None,
                    "dirty": bool(status.strip()) if status is not None else None,
                    "diff": git("diff", "HEAD", "--no-ext-diff", "--", *CORE_SOURCES)},
            "core_sources": sources, "environment": environment}


class ResourceTracker:
    """Cumulative process timings and allocator peaks across resumed sessions.

    The loop clock includes data loading, checkpointing and logging after
    start_training(), and excludes validation contexts. It is not kernel-only
    time. CUDA synchronizes at phase boundaries, never at each update.
    """
    def __init__(self, device, initial_step=0, previous=None):
        import torch
        self.device = torch.device(device)
        self.initial_step = int(initial_step)
        self.previous = dict(previous or {})
        self.wall_start = time.perf_counter()
        self.training_start = None
        self.validation_seconds = 0.0
        self.loop_validation_seconds = 0.0
        self.validation_start = None
        self.cuda = self.device.type == "cuda" and torch.cuda.is_available()
        if self.cuda:
            torch.cuda.reset_peak_memory_stats(self.device)

    def _sync(self):
        if self.cuda:
            import torch
            torch.cuda.synchronize(self.device)

    def start_training(self):
        if self.training_start is None:
            self._sync()
            self.training_start = time.perf_counter()

    @contextmanager
    def validation(self):
        if self.validation_start is not None:
            raise RuntimeError("Validation timing contexts cannot overlap")
        self._sync()
        self.validation_start = time.perf_counter()
        try:
            yield
        finally:
            try:
                self._sync()
            finally:
                elapsed = time.perf_counter() - self.validation_start
                self.validation_seconds += elapsed
                if self.training_start is not None:
                    self.loop_validation_seconds += elapsed
                self.validation_start = None

    def snapshot(self, step):
        now = time.perf_counter()
        active_validation = now - self.validation_start if self.validation_start is not None else 0.0
        loop_elapsed = now - self.training_start if self.training_start is not None else 0.0
        excluded = self.loop_validation_seconds + (active_validation if self.training_start is not None else 0.0)
        training = float(self.previous.get("training_seconds") or 0) + max(0.0, loop_elapsed - excluded)
        updates = int(self.previous.get("successful_updates") or 0) + max(0, int(step) - self.initial_step)
        result = {"wall_seconds": float(self.previous.get("wall_seconds") or 0) + now - self.wall_start,
                  "training_seconds": training,
                  "validation_seconds": float(self.previous.get("validation_seconds") or 0)
                                        + self.validation_seconds + active_validation,
                  "successful_updates": updates,
                  "updates_per_second": updates / training if training > 0 else 0.0,
                  "peak_allocated_bytes": None, "peak_reserved_bytes": None}
        if self.cuda:
            import torch
            for key, getter in (("peak_allocated_bytes", torch.cuda.max_memory_allocated),
                                ("peak_reserved_bytes", torch.cuda.max_memory_reserved)):
                result[key] = max(int(self.previous.get(key) or 0), int(getter(self.device)))
        return result


def _read_csv(path):
    path = Path(path)
    if not path.is_file():
        return [], []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def reconcile_validation_csv(path, fields, modalities, checkpoint_step):
    """Retain the latest complete validation event at each step, including ahead events.

    Legacy events have no event id; repeated modalities delimit successive events
    at the same step. A partial later event never replaces a complete earlier one.
    """
    columns, rows = _read_csv(path)
    required = {"step", "modality", "psnr", "ssim", "rmse"}
    if columns and not required.issubset(columns):
        raise ValueError(f"Unexpected validation CSV columns in {path}")
    modalities = list(modalities)
    if len(set(modalities)) != len(modalities) or not modalities:
        raise ValueError("Validation modalities must be nonempty and distinct")
    groups, identified = [], {}
    legacy = None
    for position, row in enumerate(rows):
        try:
            step = int(row.get("step", ""))
            if step < 0:
                raise ValueError("Negative validation step")
        except (TypeError, ValueError):
            groups.append({"step": None, "rows": [row], "last": position})
            legacy = None
            continue
        event_id = row.get("event_id") or ""
        if event_id:
            legacy = None
            key = (step, event_id)
            if key not in identified:
                identified[key] = {"step": step, "rows": [], "last": position}
                groups.append(identified[key])
            group = identified[key]
        else:
            if (legacy is None or legacy["step"] != step
                    or row.get("modality") in {item.get("modality") for item in legacy["rows"]}):
                legacy = {"step": step, "rows": [], "last": position}
                groups.append(legacy)
            group = legacy
        group["rows"].append(row)
        group["last"] = position

    def complete(group):
        values = group["rows"]
        if (group["step"] is None or len(values) != len(modalities)
                or {row.get("modality") for row in values} != set(modalities)):
            return False
        for row in values:
            if None in row or any(row.get(key) is None for key in required):
                return False
            try:
                scores = {key: float(row[key]) for key in ("psnr", "ssim", "rmse")}
            except (ValueError, TypeError):
                return False
            if (math.isnan(scores["psnr"]) or scores["psnr"] == float("-inf")
                    or not math.isfinite(scores["ssim"]) or not math.isfinite(scores["rmse"])
                    or scores["rmse"] < 0):
                return False
        # Mixed run/session identities cannot form a single event.
        return all(len({row.get(key) or "" for row in values}) == 1
                   for key in ("run_id", "session_id", "event_id"))

    retained = {}
    for group in groups:
        if complete(group) and (group["step"] not in retained
                                or group["last"] > retained[group["step"]]["last"]):
            retained[group["step"]] = group
    output = []
    for step, group in sorted(retained.items()):
        by_modality = {row["modality"]: row for row in group["rows"]}
        output.extend({key: by_modality[modality].get(key) or "" for key in fields}
                      for modality in modalities)
    save_summary_csv(path, output, fields)
    return {"discarded_groups": len(groups) - len(retained), "kept_groups": len(retained),
            "total_groups": len(groups), "retained_steps": sorted(retained),
            "ahead_steps": sorted(step for step in retained if step > checkpoint_step),
            "upgraded_legacy": bool(columns and "event_id" not in columns)}


def commit_validation_csv(path, fields, rows):
    """Commit a complete validation group in one replace, superseding its step."""
    rows = list(rows)
    if not rows:
        raise ValueError("A validation event cannot be empty")
    step = int(rows[0]["step"])
    if any(int(row["step"]) != step for row in rows):
        raise ValueError("A validation event must contain a single step")
    _, existing = _read_csv(path)
    retained, replaced = [], []
    for row in existing:
        try:
            same_step = int(row.get("step", "")) == step
        except (TypeError, ValueError):
            same_step = False
        if same_step:
            event_id = row.get("event_id")
            if event_id and event_id not in replaced:
                replaced.append(event_id)
        else:
            retained.append(row)
    combined = retained + rows
    combined.sort(key=lambda row: int(row["step"]))
    save_summary_csv(path, combined, fields)
    return replaced


EXPERIMENT_INDEX = Path(__file__).resolve().parent / "output" / "experiments.csv"
EXPERIMENT_FIELDS = [
    "run_id", "name", "purpose", "comparison", "notes", "status", "save_dir", "seed",
    "modalities", "batchSize", "patch_size", "total_iterations", "lr", "min_lr", "balance_weight",
    "val_interval", "val_samples", "amp", "amp_requested", "tf32", "tf32_requested",
    "tf32_matmul", "tf32_cudnn", "deterministic", "deterministic_requested",
    "deterministic_cudnn", "data_path", "data_signature",
    "git_commit", "git_branch", "git_dirty", "best_step", "val_macro_psnr", "val_macro_ssim",
    "val_macro_rmse", "train_seconds", "validation_seconds", "wall_seconds", "successful_updates",
    "updates_per_second", "peak_allocated_bytes", "peak_reserved_bytes", "test_evaluation_id",
    "test_checkpoint_step", "test_checkpoint_sha256", "test_subset", "test_mode", "test_summary_path",
    "test_macro_psnr", "test_macro_ssim", "test_macro_rmse", "test_wall_seconds",
    "test_validation_seconds", "test_peak_allocated_bytes", "test_peak_reserved_bytes",
]


def _index_metrics(prefix, summary):
    result = {}
    for key in ("psnr", "ssim", "rmse"):
        if key in summary.get("macro", {}):
            result[f"{prefix}_macro_{key}"] = summary["macro"][key]
    for modality, metrics in summary.get("modalities", {}).items():
        for key in ("psnr", "ssim", "rmse"):
            if key in metrics:
                result[f"{prefix}_{modality}_{key}"] = metrics[key]
    return result


def _index_cell(value):
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_json_value(value), ensure_ascii=False, sort_keys=True)
    return "" if value is None else _json_value(value)


@contextmanager
def _experiment_index_lock(path):
    """Serialize a complete index transaction without a persistent lock file."""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateMutexW.argtypes = (ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR)
        kernel.CreateMutexW.restype = wintypes.HANDLE
        kernel.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
        kernel.WaitForSingleObject.restype = wintypes.DWORD
        kernel.ReleaseMutex.argtypes = (wintypes.HANDLE,)
        kernel.ReleaseMutex.restype = wintypes.BOOL
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel.CloseHandle.restype = wintypes.BOOL
        normalized = os.path.normcase(os.path.normpath(str(path)))
        name = "Local\\DIDNExperimentIndex_" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        handle = kernel.CreateMutexW(None, False, name)
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        acquired = False
        try:
            result = kernel.WaitForSingleObject(handle, 5000)
            if result == 0x102:
                raise TimeoutError(f"Timed out waiting for experiment index {path}")
            if result not in (0, 0x80):  # WAIT_OBJECT_0 or WAIT_ABANDONED
                raise ctypes.WinError(ctypes.get_last_error())
            acquired = True
            yield
        finally:
            try:
                if acquired and not kernel.ReleaseMutex(handle):
                    raise ctypes.WinError(ctypes.get_last_error())
            finally:
                kernel.CloseHandle(handle)
    else:
        import fcntl

        descriptor = os.open(path.parent, os.O_RDONLY)
        acquired = False
        deadline = time.monotonic() + 5.0
        try:
            while not acquired:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"Timed out waiting for experiment index {path}")
                    time.sleep(min(.05, max(0.0, deadline - time.monotonic())))
            yield
        finally:
            try:
                if acquired:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)


@contextmanager
def training_lock(directory, timeout=0.0):
    """Hold an exclusive cross-process lock on one training directory.

    The lock lives for the whole run, so a second process using the same
    --save_dir cannot pass the directory checks and afterwards overwrite
    run.json, checkpoints, CSV logs or the fixed *.tmp files. Windows uses a
    named mutex (released by the kernel if the process dies); POSIX uses an
    advisory lock on .train.lock.
    """
    directory = Path(directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    message = (f"{directory} is already used by another training process; "
               "use a separate --save_dir for each parallel experiment")
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateMutexW.argtypes = (ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR)
        kernel.CreateMutexW.restype = wintypes.HANDLE
        kernel.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
        kernel.WaitForSingleObject.restype = wintypes.DWORD
        kernel.ReleaseMutex.argtypes = (wintypes.HANDLE,)
        kernel.ReleaseMutex.restype = wintypes.BOOL
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel.CloseHandle.restype = wintypes.BOOL
        normalized = os.path.normcase(os.path.normpath(str(directory)))
        name = "Local\\DIDNTraining_" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        handle = kernel.CreateMutexW(None, False, name)
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        acquired = False
        try:
            waited = kernel.WaitForSingleObject(handle, int(max(0.0, timeout) * 1000))
            if waited == 0x102:  # WAIT_TIMEOUT
                raise RuntimeError(message)
            if waited not in (0, 0x80):  # WAIT_OBJECT_0 or WAIT_ABANDONED
                raise ctypes.WinError(ctypes.get_last_error())
            acquired = True
            yield
        finally:
            try:
                if acquired and not kernel.ReleaseMutex(handle):
                    raise ctypes.WinError(ctypes.get_last_error())
            finally:
                kernel.CloseHandle(handle)
    else:
        import fcntl

        descriptor = os.open(directory / ".train.lock", os.O_RDWR | os.O_CREAT, 0o644)
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError(message) from None
            yield
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)


def update_experiment_index(run, evaluation=None, annotation_overrides=None):
    """Refresh the comparison index without interrupting authoritative outputs.

    A native process lock covers reading, merging and atomic replacement. Excel
    sharing violations or a lock timeout only issue a warning; run metadata,
    checkpoints and evaluation reports remain usable independently of this index.
    """
    if not run.get("run_id"):
        return None
    try:
        path = Path(EXPERIMENT_INDEX).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        with _experiment_index_lock(path):
            return _update_experiment_index(path, run, evaluation, annotation_overrides)
    except (OSError, TimeoutError) as error:
        print(f"[Warning] Could not update experiment index {EXPERIMENT_INDEX}: {error}. "
              "Run metadata, checkpoints and evaluation reports remain authoritative.", file=sys.stderr, flush=True)
        return None


def _update_experiment_index(path, run, evaluation=None, annotation_overrides=None):
    """Atomically upsert one run while preserving annotations and prior test fields.

    Only independent test-split evaluations populate test columns. When both
    inference modes exist, no_self_ensemble is the comparable primary result.
    """
    run_id = run.get("run_id")
    if not run_id:
        return None
    columns, rows = _read_csv(path)
    matching = [row for row in rows if row.get("run_id") == run_id]
    previous = matching[-1] if matching else {}
    target = dict(previous)
    config = run.get("config") or {}
    experiment = run.get("experiment") or {}
    resources = run.get("resources") or {}
    provenance = run.get("provenance") or {}
    git = provenance.get("git") or {}
    environment = provenance.get("environment") or {}
    best = run.get("best") or {}
    values = {"run_id": run_id}
    for key in ("status", "save_dir"):
        if run.get(key) is not None:
            values[key] = run[key]
    for key in ("name", "purpose", "comparison", "notes"):
        if not previous.get(key) and experiment.get(key) not in (None, ""):
            values[key] = experiment[key]
    for key in ("seed", "modalities", "batchSize", "patch_size", "total_iterations", "lr", "min_lr",
                "balance_weight", "val_interval", "val_samples",
                "data_path", "data_signature"):
        if key in config:
            values[key] = config[key]
    for key in ("amp", "tf32", "deterministic"):
        if key in config:
            values[f"{key}_requested"] = config[key]
        effective = environment.get(key)
        if isinstance(effective, dict):
            primary = "matmul" if key == "tf32" else "algorithms"
            values[key] = effective.get(primary)
            for backend in (("matmul", "cudnn") if key == "tf32" else ("cudnn",)):
                if backend in effective:
                    values[f"{key}_{backend}"] = effective[backend]
        else:
            values[key] = effective
    for key in ("commit", "branch", "dirty"):
        if key in git:
            values[f"git_{key}"] = git[key]
    if best.get("step") is not None:
        values["best_step"] = best["step"]
    values.update(_index_metrics("val", best.get("metrics") or {}))
    for key in ("training_seconds", "validation_seconds", "wall_seconds", "successful_updates",
                "updates_per_second", "peak_allocated_bytes", "peak_reserved_bytes"):
        if key in resources:
            values["train_seconds" if key == "training_seconds" else key] = resources[key]
    for key, value in values.items():
        # An evaluation may carry an older neighboring run.json. Fill missing
        # training metadata, but keep the registry's latest lifecycle and best.
        if evaluation is None or not previous.get(key):
            target[key] = _index_cell(value)
    for key in ("name", "purpose", "comparison", "notes"):
        if annotation_overrides is not None and key in annotation_overrides:
            target[key] = _index_cell(annotation_overrides[key])
    if evaluation is not None:
        evaluation_config = evaluation.get("config") or {}
        split = evaluation_config.get("split", evaluation.get("split"))
        metrics = evaluation.get("metrics") or {}
        mode = "no_self_ensemble" if "no_self_ensemble" in metrics else "self_ensemble"
        if split == "test" and mode in metrics:
            identity = evaluation.get("checkpoint_identity") or {}
            # Clear primary metric cells when replacing a result from another
            # subset/modalities; old task scores must not look like this test.
            for key in list(target):
                if key.startswith("test_") and key.endswith(("_psnr", "_ssim", "_rmse")):
                    target[key] = ""
            target.update(_index_metrics("test", metrics[mode]))
            output = evaluation.get("output_dir")
            target.update({
                "test_evaluation_id": evaluation.get("evaluation_id", ""),
                "test_checkpoint_step": identity.get("step", ""),
                "test_checkpoint_sha256": identity.get("sha256", ""),
                "test_subset": int(evaluation_config.get("max_images") or 0) > 0,
                "test_mode": mode,
                "test_summary_path": str(Path(output).resolve() / "medical_summary.json") if output else "",
            })
            evaluation_resources = evaluation.get("resources") or {}
            mode_resources = evaluation_resources.get(mode, evaluation_resources)
            for key in ("wall_seconds", "validation_seconds", "peak_allocated_bytes", "peak_reserved_bytes"):
                target[f"test_{key}"] = mode_resources.get(key, "")
    fields = list(EXPERIMENT_FIELDS)
    for key in [*columns, *target]:
        if key not in fields:
            fields.append(key)
    target = {key: _index_cell(value) for key, value in target.items()}
    result, written = [], False
    for row in rows:
        if row.get("run_id") == run_id:
            if not written:
                result.append(target)
                written = True
        else:
            result.append(row)
    if not written:
        result.append(target)
    save_summary_csv(path, result, fields)
    return path


class _ConsoleLog:
    """Buffer setup messages until the training directory's ownership is checked."""
    def __init__(self):
        self.stream = io.StringIO()
        self.attached = False

    def attach(self, path):
        if self.attached:
            raise RuntimeError("Console log is already attached")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        stream = path.open("a", encoding="utf-8")
        try:
            stream.write(self.stream.getvalue())
            stream.flush()
        except BaseException:
            stream.close()
            raise
        self.stream.close()
        self.stream, self.attached = stream, True

    def write(self, value):
        self.stream.write(value)
        self.stream.flush()


class _Tee:
    def __init__(self, original, log):
        self.original, self.log = original, log

    def write(self, value):
        result = self.original.write(value)
        self.log.write(value)
        return result

    def flush(self):
        self.original.flush()
        self.log.stream.flush()

    def __getattr__(self, name):
        return getattr(self.original, name)


@contextmanager
def capture_console():
    """Tee Python stdout/stderr; attach only after run validation succeeds."""
    log = _ConsoleLog()
    try:
        with redirect_stdout(_Tee(sys.stdout, log)), redirect_stderr(_Tee(sys.stderr, log)):
            try:
                yield log
            except BaseException:
                if log.attached:
                    # The interpreter prints the re-raised exception to the terminal.
                    # Write directly to the file here to avoid printing it twice.
                    traceback.print_exc(file=log.stream)
                    log.stream.flush()
                raise
    finally:
        log.stream.close()


def prepare_training_csv(path, fields, checkpoint_step):
    """Keep only training records supported by the checkpoint being resumed."""
    path = Path(path)
    if not path.is_file():
        save_summary_csv(path, [], fields)
        return
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != fields:
            raise ValueError(f"Unexpected training CSV columns in {path}")
        rows = list(reader)
    retained = [row for row in rows if int(row["step"]) <= checkpoint_step]
    if len(retained) != len(rows):
        temporary = path.with_suffix(path.suffix + ".tmp")
        try:
            save_summary_csv(temporary, retained, fields)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
