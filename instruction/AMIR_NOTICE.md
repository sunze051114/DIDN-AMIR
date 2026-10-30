# AMIR attribution

The RIN, SRM/MoE and CRM algorithms in `model.py`, medical normalization
and metric definitions are adapted from Zhiwen Yang et al.'s implementation:

- [All-In-One Medical Image Restoration via Task-Adaptive Routing](https://github.com/Yaziwel/All-In-One-Medical-Image-Restoration-via-Task-Adaptive-Routing), MICCAI 2024.
- Local source: `D:/projects/AMIR`, commit `3fe6fd800e48bc05bde2c624274faaed86cfa65a`.
- Source files: `model/Model_AMIR.py`, `data/common.py`,
  `data/MedicalDataUniform.py`, `evaluation/evaluation_metric.py`.
- License: Apache 2.0; full text in [AMIR_LICENSE](AMIR_LICENSE).

Changes in this project: the routing classes are consolidated with the DIDN
backbone in `model.py`; native PyTorch reshaping replaces einops; sparse expert
fusion uses mathematically equivalent logsumexp for numerical stability;
gating/CDF computations use FP32 during AMP; the dictionary has 16 entries of
length 256 as in the local AMIR configuration. Routing is inserted into DIDN's
six DUBs rather than AMIR's Transformer encoder/decoder. Data loading is
self-contained, crops are reproducible by epoch/index, full-slice evaluation
preserves NIfTI geometry and pads/unpads to DIDN's spatial factor. No data
files or patient splits are regenerated or moved.

```bibtex
@inproceedings{yang2024amir,
  title={All-In-One Medical Image Restoration via Task-Adaptive Routing},
  author={Yang, Zhiwen and Chen, Haowei and Qian, Ziniu and Yi, Yang and Zhang, Hui and Zhao, Dan and Wei, Bingzheng and Xu, Yan},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={67--77},
  year={2024},
  organization={Springer}
}
```
