"""Full-slice inference and AMIR's physical-range PSNR/SSIM/RMSE protocol.

Metrics use the evaluation window; exported NIfTI volumes keep the full
training range so clipping of the metric window never reaches disk.
"""
from pathlib import Path
from math import exp

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from dataset import EVAL_RANGES, RANGES, denormalize


def metric_protocol():
    """Describe the current numerical protocol without changing evaluation."""
    return {
        "version": 1,
        "physical_ranges": {key: list(value) for key, value in RANGES.items()},
        "evaluation_ranges": {key: list(value) for key, value in EVAL_RANGES.items()},
        "normalization": "Clamp to the modality physical range, then scale to [0, 1]",
        "metric_clamping": "Convert reference and prediction to physical units; clamp both to evaluation_ranges",
        "export_clamping": "Convert prediction to physical units; clamp to physical_ranges",
        "metrics": {
            "data_range": "Per-HQ-slice maximum minus minimum after evaluation clamping",
            "constant_reference": "Raise ValueError when data_range is zero",
            "psnr": "10 * log10(data_range ** 2 / mean((reference - prediction) ** 2))",
            "rmse": "sqrt(mean((reference - prediction) ** 2)) in physical units",
            "ssim": {
                "window_size": 11,
                "sigma": 1.5,
                "window": "Normalized separable discrete Gaussian",
                "padding": "zero",
                "padding_pixels": 5,
                "constants": {"k1": 0.01, "k2": 0.03},
                "reduction": "Arithmetic mean over all original pixels, including borders",
            },
        },
        "aggregation": {
            "slice": "Compute metrics independently for each full HQ/LQ slice pair",
            "modality": "Arithmetic mean of slice metrics within each modality",
            "macro": "Arithmetic mean of modality means; equal weight per modality",
        },
        "inference": {
            "factor": 8,
            "minimum_size": 16,
            "padding": "replicate",
            "padding_sides": ["right", "bottom"],
            "crop_to_original": True,
            "self_ensemble": {
                "count": 8,
                "rotations_degrees": [0, 90, 180, 270],
                "horizontal_flip": [False, True],
                "order": "Rotate, optionally flip; predict; undo flip and rotation",
                "reduction": "Arithmetic mean of the eight inverse-transformed normalized predictions",
            },
        },
    }


def predict_slice(model, image, self_ensemble=False):
    """Pad to DIDN's factor of eight; retain every original pixel."""
    def once(value):
        height, width = value.shape[-2:]
        pad_h = max(16, (height + 7) // 8 * 8) - height
        pad_w = max(16, (width + 7) // 8 * 8) - width
        value = F.pad(value, (0, pad_w, 0, pad_h), mode="replicate")
        return model(value)[..., :height, :width]
    if not self_ensemble:
        return once(image)
    outputs = []
    for flip in (False, True):
        for rotation in range(4):
            value = torch.rot90(image, rotation, (-2, -1))
            if flip:
                value = value.flip(-1)
            value = once(value)
            if flip:
                value = value.flip(-1)
            outputs.append(torch.rot90(value, -rotation, (-2, -1)))
    return torch.stack(outputs).mean(0)


def medical_metrics(reference, prediction):
    """AMIR release: per-slice HQ max-min; Gaussian SSIM with zero padding."""
    data_range = reference.max() - reference.min()
    if data_range <= 0:
        raise ValueError("Constant HQ slice has zero metric data_range")
    mse = (reference - prediction).square().mean()
    psnr = 10 * torch.log10(data_range.square() / mse)
    gaussian = torch.tensor([exp(-(i - 5) ** 2 / (2 * 1.5 ** 2)) for i in range(11)],
                            device=reference.device, dtype=reference.dtype)
    gaussian = gaussian / gaussian.sum()
    window = (gaussian[:, None] * gaussian[None, :])[None, None]
    mu_a, mu_b = F.conv2d(reference, window, padding=5), F.conv2d(prediction, window, padding=5)
    var_a = F.conv2d(reference.square(), window, padding=5) - mu_a.square()
    var_b = F.conv2d(prediction.square(), window, padding=5) - mu_b.square()
    cov = F.conv2d(reference * prediction, window, padding=5) - mu_a * mu_b
    c1, c2 = (.01 * data_range).square(), (.03 * data_range).square()
    ssim = (((2 * mu_a * mu_b + c1) * (2 * cov + c2)) /
            ((mu_a.square() + mu_b.square() + c1) * (var_a + var_b + c2))).mean()
    return {"psnr": psnr.item(), "ssim": ssim.item(), "rmse": mse.sqrt().item()}


@torch.inference_mode()
def evaluate_medical(model, dataset, device, self_ensemble=False, output_dir=None, num_workers=0):
    was_training = model.training
    model.eval()
    rows = []
    try:
        for low, high, modalities, names in DataLoader(dataset, batch_size=1, num_workers=num_workers,
                                                      generator=torch.Generator().manual_seed(0)):
            modality, name = modalities[0], names[0]
            normalized = predict_slice(model, low.to(device), self_ensemble).float().cpu()
            if not torch.isfinite(normalized).all():
                raise FloatingPointError(f"Nonfinite prediction: {modality}/{name}")
            # Metrics keep AMIR's evaluation window; the exported slice keeps the
            # full training range so CT HU outside [-160, 240] survive.
            prediction = denormalize(normalized, modality)
            reference = denormalize(high, modality)
            scores = medical_metrics(reference, prediction)
            rows.append({"modality": modality, "image": name, **scores})
            if output_dir is not None:
                import SimpleITK as sitk
                folder = Path(output_dir) / modality
                folder.mkdir(parents=True, exist_ok=True)
                source = sitk.ReadImage(str(dataset.records[len(rows) - 1][0]))
                array = denormalize(normalized, modality, evaluation=False)[0, 0].numpy()
                if source.GetDimension() == 3:
                    array = array[None]
                output = sitk.GetImageFromArray(array)
                output.CopyInformation(source)
                sitk.WriteImage(output, str(folder / f"{name}.nii.gz"))
    finally:
        model.train(was_training)
    if not rows:
        raise ValueError("No medical evaluation samples")
    metrics = ("psnr", "ssim", "rmse")
    by_modality = {}
    for modality in dataset.counts:
        selected = [row for row in rows if row["modality"] == modality]
        by_modality[modality] = {"count": len(selected), **{
            key: float(np.mean([row[key] for row in selected])) for key in metrics}}
    # Equal weight per task for selecting the best all-in-one checkpoint.
    macro = {key: float(np.mean([value[key] for value in by_modality.values()])) for key in metrics}
    return {"macro": macro, "modalities": by_modality}, rows
