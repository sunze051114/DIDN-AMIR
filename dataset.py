"""AMIR's preprocessed paired MRI/CT/PET slices, with original split boundaries.

Validation subsets requested with `limit` are drawn per patient so that the
slices driving best.pth selection are not confined to one subject.
"""
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


MODALITIES = ("PET", "CT", "MRI")
RANGES = {"PET": (0., 20.), "CT": (-1024., 3071.), "MRI": (0., 4095.)}
EVAL_RANGES = {**RANGES, "CT": (-160., 240.)}


def default_data_root():
    return str(Path(__file__).resolve().parent / "data/All-in-One")


def normalize(image, modality):
    low, high = RANGES[modality]
    return (image.clamp(low, high) - low) / (high - low)


def denormalize(image, modality, evaluation=True):
    low, high = RANGES[modality]
    image = image * (high - low) + low
    return image.clamp(*(EVAL_RANGES if evaluation else RANGES)[modality])


def patient_key(name):
    """Patient prefix of an AMIR slice name: tokens up to the first one with a digit.

    'Patient_19_0' -> 'Patient_19', 'IXI015-HH-1258-T2_7' -> 'IXI015-HH-1258-T2',
    'L109_42_0' -> 'L109'.
    """
    key = []
    for token in name.split("_"):
        key.append(token)
        if any(character.isdigit() for character in token):
            break
    return "_".join(key)


def stratified_names(names, limit, seed):
    """Deterministic subset that spreads `limit` slices over as many patients as possible."""
    patients = {}
    for name in names:
        patients.setdefault(patient_key(name), []).append(name)
    keys = sorted(patients)
    if len(keys) < 2 or limit >= len(names):
        return names[:limit]
    order = np.random.default_rng(np.random.SeedSequence([seed, limit])).permutation(len(keys)).tolist()
    queues = []
    for position, index in enumerate(order):
        slices = patients[keys[index]]
        shuffle = np.random.default_rng(np.random.SeedSequence([seed, limit, position]))
        queues.append([slices[i] for i in shuffle.permutation(len(slices)).tolist()])
    selected, depth = [], 0
    while len(selected) < limit and any(depth < len(queue) for queue in queues):
        for queue in queues:
            if len(selected) == limit:
                break
            if depth < len(queue):
                selected.append(queue[depth])
        depth += 1
    return selected


def read_slice(path):
    path = Path(path)
    if path.suffix == ".bin":
        # Author-provided .bin files are pickled arrays; use trusted data only.
        with path.open("rb") as handle:
            image = pickle.load(handle)
    else:
        import SimpleITK as sitk
        image = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
    value = torch.as_tensor(np.asarray(image) if not isinstance(image, torch.Tensor) else image).float()
    if value.ndim == 2:
        value = value.unsqueeze(0)
    if value.ndim != 3 or value.shape[0] != 1 or min(value.shape[-2:]) < 1 or not torch.isfinite(value).all():
        raise ValueError(f"Expected finite [H,W] or [1,H,W] medical slice: {path}, {value.shape}")
    return value


class MedicalDataset(Dataset):
    def __init__(self, root, modalities=MODALITIES, split="train", patch_size=128,
                 seed=42, limit=0, samples=None):
        if split not in ("train", "validation", "test"):
            raise ValueError("split must be train, validation or test")
        if not modalities or len(set(modalities)) != len(modalities) or any(m not in MODALITIES for m in modalities):
            raise ValueError("Select distinct modalities from PET, CT, MRI")
        if patch_size < 16 or patch_size % 8 or seed < 0 or limit < 0:
            raise ValueError("patch_size must be >=16 divisible by 8; seed/limit must be nonnegative")
        if samples is not None:
            if split == "train":
                raise ValueError("samples only apply to validation or test splits")
            if any(modality not in modalities for modality in samples):
                raise ValueError("samples contain modalities that were not selected")
            if any(not names for names in samples.values()):
                raise ValueError("samples must not contain empty modalities")
        self.root, self.split = Path(root).resolve(), split
        self.patch_size, self.seed, self.epoch = patch_size, seed, 0
        self.records, self.counts = [], {}
        self.selected, self.patient_counts = {}, {}
        if samples is not None:
            strategy = "recorded"
        elif not limit:
            strategy = "all"
        elif split == "validation":
            strategy = "patient_stratified"
        else:
            strategy = "prefix"
        self.selection = {"strategy": strategy, "seed": seed, "limit": limit}
        suffixes = (".bin",) if split == "train" else (".nii", ".nii.gz")
        for modality in modalities:
            folder = self.root / modality / split
            low = {p.name: p for p in (folder / "LQ").glob("*") if p.is_file() and p.name.endswith(suffixes)}
            high = {p.name: p for p in (folder / "HQ").glob("*") if p.is_file() and p.name.endswith(suffixes)}
            if not low or low.keys() != high.keys():
                raise ValueError(f"Missing/mismatched {modality}/{split} pairs in {folder}: "
                                 f"LQ={len(low)}, HQ={len(high)}, "
                                 f"missing HQ={len(low.keys()-high.keys())}, missing LQ={len(high.keys()-low.keys())}")
            names = sorted(low)
            if samples is not None:
                requested = samples.get(modality)
                if not requested:
                    raise ValueError(f"Recorded samples have no {modality} entries for {split}")
                missing = [name for name in requested if name not in low]
                if missing:
                    raise ValueError(f"{modality}/{split} recorded samples are missing: {missing[:3]}")
                names = list(requested)[:limit] if limit else list(requested)
            elif limit:
                # Validation subsets feed best.pth selection, so spread them over patients.
                names = stratified_names(names, limit, seed) if split == "validation" else names[:limit]
            self.counts[modality] = len(names)
            self.selected[modality] = names
            self.patient_counts[modality] = len({patient_key(name) for name in names})
            self.records.extend((low[name], high[name], modality) for name in names)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        low_path, high_path, modality = self.records[index]
        low, high = read_slice(low_path), read_slice(high_path)
        if low.shape != high.shape:
            raise ValueError(f"Paired shapes differ: {low_path}, {high_path}")
        low, high = normalize(low, modality), normalize(high, modality)
        if self.split == "train":
            height, width = low.shape[-2:]
            if min(height, width) < self.patch_size:
                raise ValueError(f"Slice smaller than patch_size: {low_path}")
            # Crop varies by epoch, reproducibly across workers and resume.
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index]))
            y = int(rng.integers(height - self.patch_size + 1))
            x = int(rng.integers(width - self.patch_size + 1))
            low = low[:, y:y + self.patch_size, x:x + self.patch_size]
            high = high[:, y:y + self.patch_size, x:x + self.patch_size]
        name = low_path.name.removesuffix(".gz").removesuffix(".nii").removesuffix(".bin")
        return low.contiguous(), high.contiguous(), modality, name
