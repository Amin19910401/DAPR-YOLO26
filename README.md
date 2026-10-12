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
git clone https://github.com/<Amin19910401>/DAPR-YOLO26.git
cd DAPR-YOLO26
pip install -r requirements.txt