## HIPNO: Symmetry-Aware Physics-Informed Neural Operators for Noninvasive Hemodynamic Inference

*Yunbei Pan\*, Jiahang Sha\*, Simon A. Lee, Maxime Cannesson, Wei Wang, Jeffrey N. Chiang*
*UCLA*


[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/) [![PyTorch](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/) [![Status](https://img.shields.io/badge/status-under_review-orange.svg)](#citation)

> [!NOTE]
> **Partial release.** We are preparing the codebase for public release and will add the remaining components shortly. Trained weights and data partitions will follow upon publication.

### Overview

From noninvasive ECG and PPG signals, HIPNO recovers arterial pressure together with the vascular mechanics that generate it. Structural non-identifiability impedes the recovery of these mechanics, because distinct physiological states can produce indistinguishable pressure observations. HIPNO eliminates this indeterminacy through a symmetry-aware formulation with identifiable coordinates for vascular tone and cardiac flow drive, which allows the two to be perturbed independently. The paper presents the full construction.

### Citation

```bibtex
@misc{pan2027hipno,
  title  = {HIPNO: Symmetry-Aware Physics-Informed Neural Operators
            for Noninvasive Hemodynamic Inference},
  author = {Pan, Yunbei and Sha, Jiahang and Lee, Simon A. and
            Cannesson, Maxime and Wang, Wei and Chiang, Jeffrey N.},
  note   = {Under review},
  year   = {2026}
}
```

### License and intended use

A software license for this repository is pending institutional review and will be added before public release. VitalDB is distributed under CC-BY-4.0 by its independent source.

**Research use only.** HIPNO is not a medical device and must not be used to guide clinical decisions.
