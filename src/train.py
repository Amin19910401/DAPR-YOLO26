# -*- coding: utf-8 -*-
"""
Training pipeline for DAPR-YOLO26 on the two-class EAGLE benchmark.
"""

import os
import sys
import json
import time
import random
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
import cv2
import yaml
import torch
from ultralytics import YOLO


RANDOM_SEED = 42
NEW_SEEDS = [RANDOM_SEED]
SELECTED_CLASSES = [0, 1]
NUM_CLASSES = 2

IMAGE_SIZES_CONFIG = [
    {
        'imgsz': 640,
        'batch': 4,
        'workers': 4,
        'lr0': 0.01,
        'mosaic': 0.8,
        'cache': 'disk',
        'patience': 15,
    },
]

PROJECT_ROOT_DIR = "D:/YOLO26_Training_Runs"
RESULTS_DIR = "D:/Training_Results_2classes"
DATASET_BASE = "D:/Doustali/Dataset"

ARCHITECTURES = [
    {
        'name': 'DAPR-YOLO26s-3head',
        'yaml': os.path.join(DATASET_BASE, 'dapr_yolo26s_3head.yaml'),
        'project_name': 'DAPR-YOLO26s-3head',
        'root_dir': 'D:/eagle',
    },
]


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def save_json(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def safe_metric_attr(obj, path, default=0.0):
    try:
        cur = obj
        for p in path.split("."):
            cur = getattr(cur, p)
        return float(cur)
    except Exception:
        return default


def get_model_params_and_flops(model_path, imgsz=640):
    print("\nComputing Params and GFLOPs...")
    model = YOLO(model_path)
    model.model.to('cuda')
    model.model.eval()
    total_params = sum(p.numel() for p in model.model.parameters())
    total_params_millions = total_params / 1_000_000
    try:
        from thop import profile
        dummy_input = torch.randn(1, 3, imgsz, imgsz).to('cuda')
        flops, _ = profile(model.model, inputs=(dummy_input,), verbose=False)
        gflops = flops / 1e9
    except ImportError:
        gflops = (total_params * 2 * imgsz * imgsz) / 1e9
    print(f"   [OK] Parameters: {total_params_millions:.2f}M, GFLOPs: {gflops:.2f}")
    return total_params_millions, gflops


def measure_fps_on_real_images(model_path, data_yaml, imgsz=640, num_samples=30):
    print("\nMeasuring FPS on real images...")
    model = YOLO(model_path)
    with open(data_yaml, 'r') as f:
        data = yaml.safe_load(f)
    test_dir = os.path.join(os.path.dirname(data_yaml), data['test'])
    if not os.path.exists(test_dir):
        return 0.0, 0.0
    images = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not images:
        return 0.0, 0.0
    if len(images) > num_samples:
        images = random.sample(images, num_samples)
    sample_images = []
    for img_name in images[:num_samples]:
        img = cv2.imread(os.path.join(test_dir, img_name))
        if img is not None:
            sample_images.append(img)
    if not sample_images:
        return 0.0, 0.0
    for _ in range(3):
        _ = model.predict(sample_images, imgsz=imgsz, device='cuda', verbose=False)
    torch.cuda.synchronize()
    times = []
    for _ in range(5):
        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = model.predict(sample_images, imgsz=imgsz, device='cuda', verbose=False)
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)
    avg_time = float(np.mean(times))
    std_time = float(np.std(times))
    fps = (len(sample_images) / avg_time) * 1000
    fps_std = fps * (std_time / avg_time)
    print(f"   [OK] FPS: {fps:.1f} +/- {fps_std:.1f}")
    return fps, fps_std


def train_model(arch_config, imgsz_config, results_dir, seed=42):
    imgsz = imgsz_config['imgsz']
    batch = imgsz_config['batch']
    workers = imgsz_config.get('workers', 4)
    lr0 = imgsz_config.get('lr0', 0.01)
    mosaic = imgsz_config.get('mosaic', 0.8)
    cache = imgsz_config.get('cache', False)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    project_name = f"{arch_config['project_name']}_{imgsz}_seed{seed}_{timestamp}"
    project_root_dir = arch_config['root_dir']

    print(f"\n{'='*70}\n Training: {arch_config['name']} (imgsz={imgsz})\n{'='*70}")
    start_time = datetime.now()

    try:
        model = YOLO(arch_config['yaml'])
        model.to('cuda')
        model.train(
            data=os.path.join(DATASET_BASE, "data.yaml"),
            epochs=100,
            patience=imgsz_config.get('patience', 15),
            batch=batch,
            imgsz=imgsz,
            device='cuda',
            workers=workers,
            save_period=10,
            plots=True,
            save=True,
            exist_ok=True,
            project=project_root_dir,
            name=project_name,
            seed=seed,
            deterministic=False,
            cache=cache,
            lr0=lr0,
            classes=SELECTED_CLASSES,
            optimizer='auto',
            verbose=True,
            pretrained=False,
            amp=True,
            hsv_h=0.01, hsv_s=0.6, hsv_v=0.4,
            translate=0.08, scale=0.5, fliplr=0.5,
            mosaic=mosaic, degrees=3.0, flipud=0.1,
            mixup=0.15, shear=2.0, copy_paste=0.08,
            box=7.5, cls=0.5, dfl=1.5,
        )

        best_model_path = f"{project_root_dir}/{project_name}/weights/best.pt"
        if not os.path.exists(best_model_path):
            return None

        test_results = model.val(
            data=os.path.join(DATASET_BASE, "data.yaml"),
            split='test', batch=batch, imgsz=imgsz, device='cuda',
            conf=0.25, iou=0.6, classes=SELECTED_CLASSES,
            plots=True, save_json=True, verbose=True,
        )

        test_map50 = safe_metric_attr(test_results, "box.map50", 0.0)
        test_map = safe_metric_attr(test_results, "box.map", 0.0)
        test_precision = safe_metric_attr(test_results, "box.mp", 0.0)
        test_recall = safe_metric_attr(test_results, "box.mr", 0.0)

        params_millions, gflops = get_model_params_and_flops(best_model_path, imgsz)
        fps_real, fps_real_std = measure_fps_on_real_images(
            best_model_path, os.path.join(DATASET_BASE, "data.yaml"), imgsz
        )

        duration = str(datetime.now() - start_time).split('.')[0]

        result = {
            'architecture': arch_config['name'],
            'imgsz': imgsz,
            'project_name': project_name,
            'project_path': f"{project_root_dir}/{project_name}",
            'test_map50': round(test_map50, 4),
            'test_map': round(test_map, 4),
            'test_precision': round(test_precision, 4),
            'test_recall': round(test_recall, 4),
            'params_millions': round(params_millions, 2),
            'gflops': round(gflops, 2),
            'fps_real': round(fps_real, 1),
            'fps_real_std': round(fps_real_std, 1),
            'duration': duration,
            'success': True,
        }

        save_json(os.path.join(results_dir, f'results_{project_name}.json'), result)
        return result

    except Exception as e:
        print(f"   [ERROR] {e}")
        traceback.print_exc()
        return None


def main():
    print("DAPR-YOLO26 Training on EAGLE (2 classes)")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    set_seed(RANDOM_SEED)

    if not torch.cuda.is_available():
        print("[ERROR] CUDA not available.")
        sys.exit(1)
    print(f"[OK] GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_results = []

    for imgsz_config in IMAGE_SIZES_CONFIG:
        for arch in ARCHITECTURES:
            for seed in NEW_SEEDS:
                result = train_model(arch, imgsz_config, RESULTS_DIR, seed=seed)
                if result:
                    all_results.append(result)

    if all_results:
        df = pd.DataFrame(all_results)
        csv_path = os.path.join(RESULTS_DIR, 'all_results.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"[OK] Results saved: {csv_path}")

    print("\nAll training runs completed.")


if __name__ == '__main__':
    main()