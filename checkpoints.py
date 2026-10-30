"""DIDN+AMIR state dictionaries, atomic saves and complete training recovery."""
import math
import os
import shutil
from pathlib import Path
from collections.abc import Mapping

import torch


BEST_TOLERANCE = 1e-9


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"Invalid checkpoint: {path}")
    if checkpoint and all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
        checkpoint = {"model": checkpoint}
    if checkpoint.get("arch", "didn_amir") != "didn_amir":
        raise ValueError(f"Expected a DIDN+AMIR checkpoint: {path}")
    state = checkpoint.get("model")
    if not isinstance(state, Mapping) or not state or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError(f"Checkpoint has no valid model state: {path}")
    return dict(checkpoint)


def save_checkpoint(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def checkpoint_psnr(checkpoint):
    """Macro PSNR recorded when a best checkpoint was selected, or None."""
    metrics = checkpoint.get("metrics")
    if not isinstance(metrics, Mapping):
        return None
    macro = metrics.get("macro")
    if not isinstance(macro, Mapping) or macro.get("psnr") is None:
        return None
    return float(macro["psnr"])


def model_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def cuda_rng_states(states, available):
    """Saved CUDA generator states that exist on the current machine."""
    states = list(states or [])
    return states[:max(min(len(states), available), 0)]


def _restore_scaler(scaler, state, saved_enabled=None):
    # A disabled GradScaler stores an empty state dict, so availability may differ
    # between the saved run and this one; never fail or silently reset the scale.
    if not scaler.is_enabled():
        if state:
            print("[AMP] AMP is disabled on this device; the saved scaler state is kept "
                  "in the checkpoint but not restored", flush=True)
        return
    if not state:
        origin = "disabled AMP" if saved_enabled is False else "no scaler state"
        print(f"[AMP] Checkpoint saved with {origin}; starting a fresh scaler on this device", flush=True)
        return
    scaler.load_state_dict(state)


def _restore_rng(rng, device):
    torch.set_rng_state(rng["cpu"])
    states = list(rng.get("cuda") or [])
    if not states:
        if device.type == "cuda":
            print("[RNG] Checkpoint has no CUDA generator state; CUDA sampling starts fresh", flush=True)
        return
    if device.type != "cuda":
        print("[RNG] Checkpoint holds CUDA generator state; this CPU run leaves it unused", flush=True)
        return
    available = torch.cuda.device_count()
    kept = cuda_rng_states(states, available)
    if kept:
        torch.cuda.set_rng_state_all(kept)
    if len(kept) != available or len(states) != available:
        print(f"[RNG] Restored {len(kept)} of {available} visible CUDA generator states "
              f"({len(states)} saved); unmatched devices start fresh", flush=True)


def restore_training(checkpoint, model, optimizer, scaler, config):
    required = {"optimizer", "scaler", "next_epoch", "next_iteration", "config", "rng", "best_psnr"}
    if not required.issubset(checkpoint):
        raise ValueError("Weights-only checkpoint cannot resume Adam/data position; use --init_model instead")
    if checkpoint["config"] != config:
        changed = [key for key in set(config) | set(checkpoint["config"])
                   if checkpoint["config"].get(key) != config.get(key)]
        raise ValueError(f"Resume configuration differs: {changed}")
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    _restore_scaler(scaler, checkpoint["scaler"], checkpoint.get("scaler_enabled"))
    _restore_rng(checkpoint["rng"], model_device(model))
    return checkpoint["next_epoch"], checkpoint["next_iteration"], checkpoint["best_psnr"]


def checkpoint_summary(path):
    """Metadata of a checkpoint without materializing its model weights."""
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except (TypeError, RuntimeError, ValueError):
        payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Invalid checkpoint: {path}")
    if payload and all(isinstance(value, torch.Tensor) for value in payload.values()):
        payload = {"model": payload}
    if payload.get("arch", "didn_amir") != "didn_amir":
        raise ValueError(f"Expected a DIDN+AMIR checkpoint: {path}")
    state = payload.get("model")
    if not isinstance(state, Mapping) or not state:
        raise ValueError(f"Checkpoint has no valid model state: {path}")
    return {"psnr": checkpoint_psnr(payload), "config": payload.get("config"),
            "step": payload.get("step"), "run_id": payload.get("run_id"),
            "arch": payload.get("arch", "didn_amir"), "metrics": payload.get("metrics"),
            "validation_event_id": payload.get("validation_event_id"),
            "resources": payload.get("resources")}


def copy_checkpoint(source, target):
    """File-level atomic copy, so migrating weights never reloads the model."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def ensure_best_checkpoint(directory, source, best, config=None, run_id=None):
    """Return the best PSNR that <directory>/best.pth really represents.

    best.pth and last.pth are written separately, so an interrupted run can leave
    a newer best.pth behind an older snapshot. Resuming must keep that better
    checkpoint instead of letting a later, worse validation overwrite it.
    """
    target = Path(directory) / "best.pth"
    origin = Path(source) / "best.pth"
    if target.is_file():
        candidate, migrate = target, False
    elif origin.is_file():
        candidate, migrate = origin, True
    elif math.isfinite(best):
        raise FileNotFoundError(
            f"Resumed history records best PSNR {best:.4f} but neither {target} nor {origin} exists; "
            "keep the original best.pth, use --init_model, or start a new --save_dir")
    else:
        return best
    summary = checkpoint_summary(candidate)
    if config is not None and summary["config"] not in (None, config):
        raise ValueError(f"{candidate} was saved with a different training configuration")
    if run_id and summary["run_id"] and summary["run_id"] != run_id:
        raise ValueError(f"{candidate} belongs to experiment {summary['run_id']}, not {run_id}")
    recorded = summary["psnr"]
    if recorded is None:
        raise ValueError(f"{candidate} has no recorded macro PSNR, so it cannot be matched to this history")
    if recorded + BEST_TOLERANCE < best:
        raise ValueError(
            f"{candidate} records macro PSNR {recorded:.6f} but the resumed history claims {best:.6f}; "
            "the weights behind the recorded best are missing, so continuing would lower the best")
    if migrate:
        copy_checkpoint(origin, target)
        print(f"[Checkpoints] Migrated best.pth from {origin}", flush=True)
    elif recorded > best + BEST_TOLERANCE:
        print(f"[Checkpoints] best.pth ({recorded:.4f} dB, update {summary['step']}) is ahead of the "
              f"snapshot ({best:.4f} dB); keeping it", flush=True)
    return max(recorded, best)


def training_state(model, optimizer, scaler, config, next_epoch, next_iteration, best_psnr, run_id=None):
    return {"model": model.state_dict(), "arch": "didn_amir", "epoch": next_epoch,
            "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
            "scaler_enabled": scaler.is_enabled(), "run_id": run_id,
            "next_epoch": next_epoch, "next_iteration": next_iteration,
            "best_psnr": best_psnr, "config": config,
            "rng": {"cpu": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if next(model.parameters()).is_cuda else []}}
