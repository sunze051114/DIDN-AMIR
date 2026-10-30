"""Options for DIDN+AMIR training and independent medical evaluation."""
import argparse
import math
from pathlib import Path

from dataset import default_data_root, MODALITIES


DEFAULT_SAVE_DIR = Path(__file__).resolve().parent / "checkpoint" / "didn_amir"


def parse_args(mode="train", argv=None):
    if mode not in ("train", "test"):
        raise ValueError(f"Unknown entry point: {mode}")
    parser = argparse.ArgumentParser(description=f"DIDN+AMIR {mode}")
    parser.add_argument("--modalities", nargs="+", choices=MODALITIES, default=list(MODALITIES))
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--no_cuda", action="store_true", help="Run on CPU")
    parser.add_argument("--threads", type=int, default=0, help="DataLoader workers; 0 avoids Windows spawn overhead")
    parser.add_argument("--save_dir", default=str(DEFAULT_SAVE_DIR), help="Directory for training checkpoints and default test weights")
    if mode == "train":
        parser.add_argument("--experiment_name", default="", help="Human-readable experiment name")
        parser.add_argument("--purpose", default="", help="Experiment question or purpose")
        parser.add_argument("--comparison", default="", help="Reference experiment ID or name")
        parser.add_argument("--notes", default="", help="Experiment observations or notes")
        parser.add_argument("--data_path", default=default_data_root(), help="AMIR All-in-One root")
        parser.add_argument("--batchSize", type=int, default=1)
        parser.add_argument("--patch_size", type=int, default=128)
        parser.add_argument("--total_iterations", type=int, default=200000)
        parser.add_argument("--lr", type=float, default=2e-4)
        parser.add_argument("--min_lr", type=float, default=1e-6)
        parser.add_argument("--balance_weight", type=float, default=.001)
        parser.add_argument("--val_interval", type=int, default=1000, help="Optimizer updates; 0 validates at epoch end")
        parser.add_argument("--val_samples", type=int, default=32, help="Validation slices per modality, spread over patients; 0 uses all")
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--amp", action="store_true")
        parser.add_argument("--tf32", action="store_true")
        parser.add_argument("--grad_clip", type=float, default=0, help="0 disables clipping")
        parser.add_argument("--deterministic", action="store_true")
        parser.add_argument("--check_data", action="store_true", help="Check pairing and sample shapes in all splits, then exit")
        group = parser.add_mutually_exclusive_group()
        group.add_argument("--resume", default="", help="Full training snapshot or 'auto' for <save_dir>/last.pth; another --save_dir must be empty")
        group.add_argument("--init_model", default="", help="Matching DIDN+AMIR weights; start fresh optimizer and schedule")
    else:
        parser.add_argument("--test_path", default=default_data_root(), help="AMIR All-in-One root")
        parser.add_argument("--split", choices=("validation", "test"), default="test")
        parser.add_argument("--model", default="", help="DIDN+AMIR checkpoint; default <save_dir>/best.pth")
        parser.add_argument("--result_dir", default="results")
        parser.add_argument("--max_images", type=int, default=0, help="Limit per modality; 0 uses all")
        parser.add_argument("--ensemble", choices=("none", "self", "both"), default="none")
        parser.add_argument("--no_save_images", action="store_true", help="Write metrics without NIfTI predictions")
    args = parser.parse_args(argv)
    if len(set(args.modalities)) != len(args.modalities) or args.threads < 0:
        parser.error("modalities must be distinct and threads must be nonnegative")
    if mode == "train":
        if args.batchSize <= 0 or args.total_iterations <= 0 or args.patch_size < 16 or args.patch_size % 8:
            parser.error("batchSize/total_iterations must be positive; patch_size must be >=16 divisible by 8")
        if not all(math.isfinite(v) for v in (args.lr, args.min_lr, args.balance_weight, args.grad_clip)):
            parser.error("Learning rates, balance_weight and grad_clip must be finite")
        if not 0 <= args.min_lr <= args.lr or args.lr <= 0 or min(args.balance_weight, args.grad_clip) < 0:
            parser.error("Invalid learning rate, balance_weight or grad_clip")
        if min(args.seed, args.val_interval, args.val_samples) < 0:
            parser.error("seed/val_interval/val_samples must be nonnegative")
    elif args.max_images < 0:
        parser.error("max_images must be nonnegative")
    return args
