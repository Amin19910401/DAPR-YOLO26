# DAPR: Dynamic Adaptive Prototype-based Refinement

Official implementation of the paper:

> **DAPR: Dynamic Adaptive Prototype-based Refinement for Efficient Small Object Detection in Aerial Images**
>
> Amin Doustali, Mahdi Hasanlou, Alireza Sharifi

---

## Overview

DAPR is a lightweight feature-refinement module integrated into YOLO26 for efficient small-object detection in aerial imagery. The module decomposes spatial features into a compact set of soft-assigned parts and dynamically refines class-part-conditioned prototypes per spatial configuration, matching parts via temperature-scaled cross-attention and selectively fusing the refined representation through a learned confidence gate.

**Key results on the EAGLE benchmark (2-class: Car, Truck):**

| Model | mAP@0.5 | mAP@0.5:0.95 | Params (M) | Latency (ms) |
|-------|---------|--------------|------------|--------------|
| YOLO26s | 0.762 | 0.506 | 9.95 | 5.10 |
| YOLO26s-P2 | 0.778 | 0.524 | 9.66 | 6.25 |
| **DAPR-YOLO26 (3-head)** | **0.783** | **0.532** | **7.18** | **6.67** |

---

## Installation

```bash
git clone https://github.com/Amin19910401/DAPR-YOLO26.git
cd DAPR-YOLO26
pip install -r requirements.txt
```

**Note:** The custom DAPR module is defined in `utils/dapr.py`. To use it, place this file inside the Ultralytics package directory as `dapr.py`:

```bash
# Linux / macOS
cp utils/dapr.py <python_env>/site-packages/ultralytics/nn/modules/dapr.py

# Windows
copy utils\dapr.py <python_env>\Lib\site-packages\ultralytics\nn\modules\dapr.py
```

Then import the DAPR classes in `ultralytics/nn/modules/__init__.py` and `ultralytics/nn/tasks.py` so that they can be resolved from the YAML configuration file.

---

## Dataset

Experiments are conducted on the EAGLE dataset. Download the dataset and organize it as follows:

```
Dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── data.yaml
```

Example `data.yaml`:

```yaml
path: /path/to/Dataset
train: images/train
val: images/val
test: images/test
nc: 2
names: ['car', 'truck']
```

---

## Training

Run the training script:

```bash
python src/train.py
```

**Configuration:**
- Input size: 640 × 640
- Batch size: 4
- Epochs: 100
- Optimizer: MuSGD (Auto)
- Learning rate: 0.01
- Random seeds: 7, 42, 123
- Mixed precision (AMP): enabled

---

## DAPR Configuration

| Parameter | Value |
|-----------|-------|
| Prototype dimension `D` | 48 |
| Classes `C` | 2 |
| Parts per class `P` | 4 |
| Total parts `K = C·P` | 8 |
| Temperature `τ` | 0.50 |
| Internal dimension `c_int` | min(64, max(c_half/4, 32)) |
| MLP_Δ hidden | 32 |
| Position MLP hidden | max(D/4, 8) = 12 |
| Prototype generator hidden | 32 |
| Confidence gate hidden | max(c_int/4, 16) |
| Base prototype init | N(0, 0.02²) |

---

## Repository Structure

```
DAPR-YOLO26/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── configs/
│   └── dapr_yolo26s_3head.yaml
├── src/
│   └── train.py
└── utils/
    └── dapr.py
```

---

## Ablation Variants

The `utils/dapr.py` file provides the following ablation variants:

- `DAPR_only_v3` — full model (main contribution)
- `DAPR_noDelta` — ablation without adaptive prototype offset
- `DAPR_noPosition` — ablation without positional embedding
- `DAPR_noGate` — ablation without confidence gate
- `DAPR_noClassPart` — ablation without class-part conditioning

---

## Citation

```bibtex
@article{doustali2026dapr,
  title   = {DAPR: Dynamic Adaptive Prototype-based Refinement for Efficient Small Object Detection in Aerial Images},
  author  = {Doustali, Amin and Hasanlou, Mahdi and Sharifi, Alireza},
  journal = {IEEE Transactions on Geoscience and Remote Sensing},
  year    = {2026}
}
```

---

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
